"""Record the frozen openFDA snapshot: `queries.all_queries(drug)` for every drug in a list.

Usage:
  uv run python scripts/snapshot.py data/snapshots/drugs.txt --version 2026-10-20
  uv run python scripts/snapshot.py data/snapshots/pilot.txt --version 2026-10-15 --budget 600
  uv run python scripts/snapshot.py --stats --version 2026-10-15

Resumable and idempotent: queries already on disk are skipped, so rerunning after a crash, a
429, or a spent budget picks up where it stopped. OPENFDA_API_KEY is read from the environment.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdagym.openfda import DiskSnapshot, OpenFDAClient, OpenFDAError
from fdagym.openfda.store import DEFAULT_ROOT
from fdagym.queries import all_queries

KEYLESS_DAILY_LIMIT = 1000
_QUOTED = re.compile(r'"([^"]+)"')


def read_drugs(path: Path) -> list[str]:
    drugs = []
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name and name not in drugs:
            drugs.append(name)
    return drugs


def record(drugs: list[str], snapshot: DiskSnapshot, budget: int) -> int:
    client = snapshot.client
    for i, drug in enumerate(drugs, 1):
        queries = all_queries(drug)
        todo = [q for q in queries if not snapshot.has(q)]
        if todo and client.requests_sent + len(todo) > budget:
            print(f"budget of {budget} requests reached before {drug!r}; rerun to resume")
            return 1
        for q in todo:
            try:
                snapshot.get(q.endpoint, search=q.search, count=q.count, limit=q.limit)
            except OpenFDAError as err:
                print(f"\nstopped on {drug!r}: {err}\nrerun the same command to resume")
                return 1
        print(
            f"[{i:>3}/{len(drugs)}] {drug:<28} fetched {len(todo):>2}  cached {len(queries) - len(todo):>2}"
            f"  requests so far {client.requests_sent}",
            flush=True,
        )
    return 0


def drug_name(search: str | None) -> str:
    if not search:
        return "?"
    found = _QUOTED.search(search)
    return found.group(1) if found else "?"


def stats(snapshot: DiskSnapshot) -> None:
    rows = snapshot.write_manifest()
    by_endpoint: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    by_drug: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    labels: dict[str, tuple[int, int | None]] = {}
    updated: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        kind = f"{row['endpoint']} {'count' if row['count'] else 'records'}"
        agg = by_endpoint[kind]
        agg[0] += 1
        agg[1] += row["n_results"]
        agg[2] += row["raw_bytes"]
        agg[3] += row["trimmed_bytes"]
        agg[4] += row["stored_bytes"]
        drug = drug_name(row.get("search"))
        bucket = by_drug[drug]
        bucket[0] += 1
        bucket[1] += row["raw_bytes"]
        bucket[2] += row["trimmed_bytes"]
        bucket[3] += row["stored_bytes"]
        if row["endpoint"] == "drug/label" and not row["count"]:
            labels[drug] = (row["n_results"], row["total"])
        if row.get("last_updated"):
            updated[row["endpoint"]].add(row["last_updated"])

    print(f"{'kind':<24}{'files':>7}{'records':>9}{'raw MB':>9}{'trim MB':>9}{'gz MB':>8}")
    total = [0, 0, 0, 0, 0]
    for kind, agg in sorted(by_endpoint.items()):
        total = [a + b for a, b in zip(total, agg, strict=True)]
        print(f"{kind:<24}{agg[0]:>7}{agg[1]:>9}{agg[2] / 1e6:>9.2f}{agg[3] / 1e6:>9.2f}{agg[4] / 1e6:>8.2f}")
    print(f"{'total':<24}{total[0]:>7}{total[1]:>9}{total[2] / 1e6:>9.2f}{total[3] / 1e6:>9.2f}{total[4] / 1e6:>8.2f}")

    print(f"\n{'drug':<28}{'queries':>8}{'raw MB':>9}{'trim MB':>9}{'gz MB':>8}{'labels':>8}{'total':>8}")
    for drug, bucket in sorted(by_drug.items(), key=lambda item: -item[1][2]):
        returned, reported = labels.get(drug, (0, None))
        reported_s = "" if reported is None else str(reported)
        print(
            f"{drug:<28}{bucket[0]:>8}{bucket[1] / 1e6:>9.2f}{bucket[2] / 1e6:>9.2f}"
            f"{bucket[3] / 1e6:>8.2f}{returned:>8}{reported_s:>8}"
        )

    # limit=1 event queries are report counts: the total is the answer, and the one
    # stored record is not meant to be the whole result set. Flag the queries that
    # asked for a page of records and still came up short (labels, approvals).
    flags = [
        row
        for row in rows
        if row["count"] is None
        and row["limit"] > 1
        and row["total"] is not None
        and row["total"] > row["n_results"]
    ]
    print()
    if not flags:
        print("truncated: none (every label and approval query returned all of its matches)")
    else:
        noun = "query" if len(flags) == 1 else "queries"
        print(f"truncated: {len(flags)} {noun} that asked for a page and did not get every match")
        for row in flags:
            print(
                f"  {drug_name(row.get('search'))} {row['endpoint']} "
                f"total={row['total']} returned={row['n_results']} limit={row['limit']}"
            )
    for endpoint, dates in sorted(updated.items()):
        print(f"last_updated {endpoint}: {', '.join(sorted(dates))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("drugs", nargs="?", type=Path, help="file with one generic name per line (# comments ok)")
    parser.add_argument("--version", required=True, help="snapshot version, e.g. the recording date 2026-10-20")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--budget", type=int, help="max network requests this run (default 1000 without a key)")
    parser.add_argument("--stats", action="store_true", help="rewrite manifest.jsonl and print sizes")
    args = parser.parse_args()

    if args.stats:
        stats(DiskSnapshot(args.version, args.root))
        return 0
    if args.drugs is None:
        parser.error("a drug list is required unless --stats")

    client = OpenFDAClient()
    budget = args.budget or (10**9 if client.api_key else KEYLESS_DAILY_LIMIT)
    snapshot = DiskSnapshot(args.version, args.root, client=client)
    drugs = read_drugs(args.drugs)
    print(
        f"{len(drugs)} drugs, {len(drugs) * len(all_queries('x'))} queries, version {args.version}, "
        f"api key {'yes' if client.api_key else 'no'}, budget {budget}",
        flush=True,
    )
    code = record(drugs, snapshot, budget)
    snapshot.write_manifest()
    return code


if __name__ == "__main__":
    sys.exit(main())
