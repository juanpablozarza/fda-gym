"""Disk-backed implementation of the `Snapshot` protocol.

Layout of one frozen version:

    data/snapshots/<version>/
      manifest.jsonl                  one line per recorded query, sorted by key (regenerated, never edited)
      responses/<key[:2]>/<key>.json.gz

Each response file holds {"query": ..., "fetched_at": ..., "raw_bytes": ..., "body": <trimmed openFDA JSON>}.
`body` is not the byte-for-byte response: `trim_response` drops fields tasks never cite.
`raw_bytes` is the size openFDA actually sent, so the trim is auditable.
Gzip header mtime is 0, so the same record dict compresses to the same bytes.
A re-fetch would differ in `fetched_at` and is skipped when the file already exists.

Recording and manifest regeneration are single-writer operations per snapshot version. Unique
temporary files and atomic replacement prevent partial files, but concurrent recorders can still
choose different fetched bodies or regenerate a manifest before all in-flight responses are stored.
"""

import gzip
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from fdagym.openfda.client import OpenFDAClient
from fdagym.openfda.trim import trim_response
from fdagym.queries import Query
from fdagym.snapshot import Endpoint, SnapshotMiss

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "snapshots"


def query_key(endpoint: Endpoint, search: str | None, count: str | None, limit: int, skip: int = 0) -> str:
    canonical = json.dumps([endpoint, search, count, limit, skip], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def key_for(query: Query) -> str:
    return query_key(query.endpoint, query.search, query.count, query.limit)


def _richer_application(new: dict, old: dict) -> bool:
    """Prefer the copy that kept more of the application. Combination searches can repeat an id."""
    return (len(new.get("submissions") or []), len(new.get("products") or [])) > (
        len(old.get("submissions") or []),
        len(old.get("products") or []),
    )


@lru_cache(maxsize=2048)
def _read_text(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode()


class DiskSnapshot:
    """Offline by default. Pass a client to record misses (used only by scripts/snapshot.py)."""

    def __init__(self, version: str, root: Path = DEFAULT_ROOT, client: OpenFDAClient | None = None):
        self.version = version
        self.dir = Path(root) / version
        self.client = client

    def path(self, key: str) -> Path:
        return self.dir / "responses" / key[:2] / f"{key}.json.gz"

    def has(self, query: Query) -> bool:
        return self.path(key_for(query)).exists()

    def get(
        self,
        endpoint: Endpoint,
        *,
        search: str | None = None,
        count: str | None = None,
        limit: int = 1,
        skip: int = 0,
    ) -> dict:
        key = query_key(endpoint, search, count, limit, skip)
        path = self.path(key)
        if path.exists():
            return json.loads(_read_text(path))["body"]
        if self.client is None:
            raise SnapshotMiss(f"{endpoint} search={search!r} count={count!r} limit={limit} skip={skip} (key {key})")

        raw = self.client.get(endpoint, search=search, count=count, limit=limit, skip=skip)
        record = {
            "query": {"endpoint": endpoint, "search": search, "count": count, "limit": limit, "skip": skip},
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "raw_bytes": self.client.last_raw_bytes,
            "body": trim_response(endpoint, raw, is_count=count is not None),
        }
        self._write(path, record)
        return record["body"]

    def _write(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(json.dumps(record, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
        _atomic_write(path, data)

    def drugsfda_records(self) -> list[dict]:
        """Cached Drugs@FDA applications, one record per application number.

        The freeze stores approvals under a generic-name search. Looking up
        NDA/BLA/ANDA reads those bodies. It does not issue a new openFDA query.
        """
        cached = getattr(self, "_drugsfda_records", None)
        if cached is not None:
            return cached
        manifest = self.dir / "manifest.jsonl"
        if not manifest.exists():
            self._drugsfda_records = []
            return []
        best: dict[str, dict] = {}
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("endpoint") != "drug/drugsfda" or row.get("count"):
                continue
            path = self.path(row["key"])
            if not path.exists():
                continue
            body = json.loads(_read_text(path)).get("body")
            if not isinstance(body, dict):
                continue
            for record in body.get("results") or []:
                if not isinstance(record, dict):
                    continue
                number = record.get("application_number")
                if not isinstance(number, str) or not number:
                    continue
                key = number.upper()
                prev = best.get(key)
                if prev is None or _richer_application(record, prev):
                    best[key] = record
        self._drugsfda_records = list(best.values())
        return self._drugsfda_records

    def write_manifest(self) -> list[dict]:
        """Regenerate the manifest from stored responses after the single recorder finishes.

        Unique temporary files prevent partial-file corruption, but concurrent recorders for one
        version can still race to choose response bodies or observe an incomplete manifest.
        """
        rows = []
        for path in sorted((self.dir / "responses").glob("*/*.json.gz")):
            record = json.loads(_read_text(path))
            body = record["body"]
            rows.append(
                {
                    "key": path.name.removesuffix(".json.gz"),
                    **record["query"],
                    "total": body.get("meta", {}).get("results", {}).get("total"),
                    "n_results": len(body.get("results", [])),
                    "last_updated": body.get("meta", {}).get("last_updated"),
                    "fetched_at": record["fetched_at"],
                    "raw_bytes": record["raw_bytes"],
                    "trimmed_bytes": len(json.dumps(body, separators=(",", ":"))),
                    "stored_bytes": path.stat().st_size,
                }
            )
        manifest = self.dir / "manifest.jsonl"
        _atomic_write(manifest, "".join(json.dumps(row) + "\n" for row in rows).encode())
        return rows


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace a file atomically without sharing a temporary path between writers."""
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
