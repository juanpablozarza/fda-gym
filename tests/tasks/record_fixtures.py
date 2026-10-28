"""Record small live openFDA responses for the task-generator tests. Run by hand, never in CI.

    uv run python tests/tasks/record_fixtures.py warfarin metformin ...

Records exactly `fdagym.tasks.planned_queries(drug)` per drug, so the fake snapshot holds the same
canonical queries the real snapshot will. Bodies are trimmed to the fields the generator reads,
which keeps the fixtures small; the trimming is lossless for task generation.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fdagym.queries import Query
from fdagym.tasks import planned_queries

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://api.fda.gov"

LABEL_KEEP = ("set_id", "id", "version", "effective_time", "boxed_warning", "contraindications")
OPENFDA_KEEP = ("generic_name", "brand_name", "manufacturer_name", "product_type")
TEXT_CLIP = 2000


def _clip(value):
    if isinstance(value, str):
        return value[:TEXT_CLIP]
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


def _trim_label(r: dict) -> dict:
    out = {k: _clip(r[k]) for k in LABEL_KEEP if k in r}
    out["openfda"] = {k: v for k, v in r.get("openfda", {}).items() if k in OPENFDA_KEEP}
    return out


def _trim_approval(r: dict) -> dict:
    return {
        "application_number": r["application_number"],
        "sponsor_name": r.get("sponsor_name"),
        "openfda": {k: v for k, v in r.get("openfda", {}).items() if k in OPENFDA_KEEP},
        "products": [
            {k: p.get(k) for k in ("brand_name", "active_ingredients", "dosage_form", "marketing_status")}
            for p in r.get("products", [])
        ],
        "submissions": [
            {k: s.get(k) for k in ("submission_type", "submission_number", "submission_status", "submission_status_date")}
            for s in r.get("submissions", [])
        ],
    }


def _trim_event(r: dict) -> dict:
    return {k: r.get(k) for k in ("safetyreportid", "receivedate", "serious")}


def _trim(q: Query, body: dict) -> dict:
    if q.count:
        return body
    trim = {"drug/label": _trim_label, "drug/drugsfda": _trim_approval, "drug/event": _trim_event}[q.endpoint]
    return {**body, "results": [trim(r) for r in body.get("results", [])]}


def fetch(q: Query) -> dict:
    params = {"search": q.search, "limit": q.limit}
    if q.count:
        params["count"] = q.count
    url = f"{BASE}/{q.endpoint}.json?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        body = {"meta": {"results": {"skip": 0, "limit": q.limit, "total": 0}}, "results": []}
    body["meta"] = {"last_updated": body["meta"].get("last_updated"), "results": body["meta"].get("results")}
    return _trim(q, body)


def main(drugs: list[str]) -> None:
    """Re-fetch approvals and FAERS. Label records stay; their query keys did not change."""
    FIXTURES.mkdir(exist_ok=True)
    calls = 0
    for drug in drugs:
        path = FIXTURES / f"{drug.replace(' ', '_')}.json"
        old = json.loads(path.read_text()) if path.exists() else []
        labels = [record for record in old if record["query"]["endpoint"] == "drug/label"]
        if not labels:
            raise SystemExit(f"{drug}: no label fixture to keep")
        fresh = []
        for query in planned_queries(drug):
            if query.endpoint == "drug/label":
                continue
            fresh.append({"query": query._asdict(), "response": fetch(query)})
            calls += 1
            print(f"{drug} {query.endpoint} {calls}", flush=True)
            time.sleep(0.35)
        path.write_text(json.dumps(labels + fresh, separators=(",", ":"), sort_keys=True) + "\n")
        print(f"{drug}: kept {len(labels)} label, wrote {len(fresh)} new", flush=True)
    print(f"live calls: {calls}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
