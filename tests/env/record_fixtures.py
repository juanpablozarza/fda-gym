"""Record the small openFDA fixture the env tests run against (live API, no key, ~10 requests).

    uv run python tests/env/record_fixtures.py

Responses are keyed by the exact `fdagym.queries.Query` the tools issue, then trimmed so the
fixture stays small: fewer label records, shorter sections, fewer submissions. `meta` is kept as-is.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fdagym import queries
from fdagym.queries import Query

OUT = Path(__file__).parent / "fixtures" / "openfda.json"
BASE = "https://api.fda.gov/"
KEEP_LABEL_SETS = {"0cbce382-9c88-4f58-ae0f-532a841e8f95"}
LABEL_FIELDS = (
    "set_id", "id", "effective_time", "version", "boxed_warning", "indications_and_usage",
    "contraindications", "warnings_and_cautions", "warnings", "adverse_reactions", "drug_interactions",
    "pregnancy", "overdosage", "description",
)
SECTION_CHARS = 2500

RECORD = [
    queries.label_search("warfarin"),
    queries.approvals_search("semaglutide"),
    queries.faers_top_reactions("metformin"),
    queries.faers_top_reactions("metformin", serious=True, year=2025),
    queries.faers_report_count("metformin", serious=True, year=2025),
    queries.faers_top_reactions("warfarin"),
    queries.faers_report_count("lecanemab", year=2020),
    queries.faers_top_reactions("lecanemab", year=2020),
    queries.faers_report_count("lecanemab", year=2024),
]


def key(q: Query) -> str:
    return json.dumps(list(q))


def fetch(q: Query) -> dict:
    params = {"search": q.search, "limit": q.limit}
    if q.count:
        params["count"] = q.count
    url = f"{BASE}{q.endpoint}.json?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"meta": {"results": {"skip": 0, "limit": q.limit, "total": 0}}, "results": []}
        if e.code == 429:
            raise SystemExit("openFDA daily quota 429; stopped without retrying") from e
        raise


def trim(q: Query, body: dict) -> dict:
    results = body.get("results", [])
    if q.endpoint == "drug/label":
        keep = [r for r in results if r.get("set_id") in KEEP_LABEL_SETS]
        keep += [r for r in results if r.get("set_id") not in KEEP_LABEL_SETS][:4]
        results = []
        for r in keep:
            slim = {k: r[k] for k in LABEL_FIELDS if k in r}
            for k, v in slim.items():
                if isinstance(v, list):
                    slim[k] = [s[:SECTION_CHARS] for s in v]
            of = r.get("openfda", {})
            slim["openfda"] = {k: of[k] for k in ("brand_name", "generic_name", "manufacturer_name") if k in of}
            results.append(slim)
    elif q.endpoint == "drug/drugsfda":
        for r in results:
            brands: list[str] = []
            for product in r.get("products") or []:
                brand = product.get("brand_name")
                if isinstance(brand, str):
                    brands.append(brand)
                elif isinstance(brand, list):
                    brands.extend(str(b) for b in brand)
            # Many applications have no openfda block; the brand lives on the product.
            r["brand_names"] = brands
            r.pop("products", None)
            r["openfda"] = {k: r.get("openfda", {}).get(k) for k in ("brand_name", "generic_name") if k in r.get("openfda", {})}
            r["submissions"] = [
                {k: s.get(k) for k in ("submission_type", "submission_number", "submission_status",
                                       "submission_status_date", "submission_class_code_description") if k in s}
                for s in r.get("submissions", [])
            ]
    elif q.endpoint == "drug/event" and not q.count:
        results = [{"safetyreportid": r.get("safetyreportid"), "receivedate": r.get("receivedate")} for r in results]
    return {"meta": body.get("meta", {}), "results": results}


def main() -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    # Approvals already hold all 6 semaglutide applications. Only the limit in the key changed.
    old_approvals = json.dumps(list(queries.Query(
        "drug/drugsfda", 'products.active_ingredients.name:"semaglutide"', None, 100,
    )))
    new_approvals = key(queries.approvals_search("semaglutide"))
    if old_approvals in out and new_approvals not in out:
        out[new_approvals] = out.pop(old_approvals)
        print("rekeyed approvals to limit", queries.APPROVALS_LIMIT)
    for q in RECORD:
        k = key(q)
        if k in out:
            continue
        body = fetch(q)
        out[k] = trim(q, body)
        print(q.endpoint, q.search, q.count, "->", len(body.get("results", [])), body.get("meta", {}).get("results"))
        time.sleep(0.4)
    # Drop FAERS bodies recorded under the old generic_name-only search.
    keep = {key(q) for q in RECORD}
    stale = [k for k in out if json.loads(k)[0] == "drug/event" and k not in keep]
    for k in stale:
        del out[k]
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(out)} keys)")


if __name__ == "__main__":
    main()
