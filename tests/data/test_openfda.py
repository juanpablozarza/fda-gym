"""Offline tests for the openFDA client, the trimmer, and DiskSnapshot.

Nothing in this file opens a socket. Live recording is scripts/snapshot.py.
"""

import gzip
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from fdagym.openfda import store as store_module
from fdagym.openfda.client import OpenFDAClient, OpenFDAError, RateLimiter
from fdagym.openfda.store import DiskSnapshot, query_key
from fdagym.openfda.trim import trim_response
from fdagym.queries import LABEL_LIMIT, label_search
from fdagym.snapshot import SnapshotMiss

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = '["drug/label","openfda.generic_name:\\"warfarin\\"",null,1000,0]'


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _client(handler, **kwargs) -> OpenFDAClient:
    clock = kwargs.pop("clock", None) or Clock()
    return OpenFDAClient(
        api_key=kwargs.pop("api_key", ""),
        requests_per_minute=kwargs.pop("requests_per_minute", 10**6),
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        clock=clock,
        **kwargs,
    )


def test_universe_is_150_and_excludes_zero_label_names():
    drugs = _names(ROOT / "data" / "snapshots" / "drugs.txt")
    assert len(drugs) == len(set(drugs)) == 150
    assert {"sotatercept", "crinecerfont", "vorasidenib"}.isdisjoint(drugs)
    assert "ibuprofen" not in drugs
    assert "diclofenac" in drugs
    assert "sertraline" in drugs


def test_full_snapshot_serves_a_non_pilot_label():
    snap = DiskSnapshot("2026-10-15")
    query = label_search("sertraline")
    body = snap.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
    assert body["meta"]["results"]["total"] == 156
    assert len(body["results"]) == 156


def test_pilot_covers_one_drug_from_each_axis():
    assert _names(ROOT / "data" / "snapshots" / "pilot.txt") == [
        "warfarin",
        "metformin",
        "semaglutide",
        "isotretinoin",
        "atorvastatin",
        "montelukast",
        "lisinopril",
        "xanomeline",
        "thioridazine",
        "adalimumab",
    ]


def test_query_key_is_the_sha256_of_compact_json():
    key = query_key("drug/label", 'openfda.generic_name:"warfarin"', None, 1000, 0)
    assert key == hashlib.sha256(CANONICAL.encode()).hexdigest()
    assert query_key("drug/label", 'openfda.generic_name:"warfarin"', None, 1000, 1) != key


def test_label_search_key_includes_the_contract_limit():
    query = label_search("Warfarin")
    assert query.limit == LABEL_LIMIT
    assert query_key(query.endpoint, query.search, query.count, query.limit) == query_key(
        "drug/label", 'openfda.generic_name:"warfarin"', None, LABEL_LIMIT, 0
    )


def test_rate_limiter_spaces_a_burst():
    clock = Clock()
    limiter = RateLimiter(60, clock=clock, sleep=clock.sleep)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert clock.t == pytest.approx(2.0)


def test_404_no_matches_becomes_an_empty_result():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}})

    body = _client(handler).get("drug/label", search='openfda.generic_name:"xanomeline"', limit=1000)
    assert body == {"meta": {"results": {"skip": 0, "limit": 1000, "total": 0}}, "results": []}


def test_404_that_is_not_no_matches_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "SERVER_ERROR"}})

    with pytest.raises(OpenFDAError) as err:
        _client(handler).get("drug/label", limit=1)
    assert err.value.status == 404


def test_daily_quota_429_stops_without_retry():
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(
            429,
            json={"error": {"code": "OVER_RATE_LIMIT", "message": "You have exceeded your daily quota."}},
        )

    client = _client(handler, max_attempts=4)
    with pytest.raises(OpenFDAError) as err:
        client.get("drug/event", search='patient.drug.openfda.generic_name:"warfarin"', limit=1)
    assert err.value.status == 429
    assert seen["n"] == 1
    assert client.requests_sent == 1


def test_retries_429_and_500_then_returns_the_body():
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(429, json={"error": {"code": "TOO_MANY_REQUESTS"}})
        if seen["n"] == 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"meta": {}, "results": [{"set_id": "ok"}]})

    client = _client(handler, backoff_base=2.0)
    body = client.get("drug/event", search='patient.drug.openfda.generic_name:"warfarin"', limit=1)
    assert body["results"] == [{"set_id": "ok"}]
    assert client.requests_sent == 3
    assert client.limiter.clock.t >= 2.0


def test_transport_errors_are_retried():
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"meta": {}, "results": []})

    body = _client(handler).get("drug/drugsfda", search='products.active_ingredients.name:"warfarin"', limit=100)
    assert body["results"] == []
    assert seen["n"] == 2


def test_gives_up_after_max_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="nope")

    client = _client(handler, max_attempts=3, backoff_base=1.0)
    with pytest.raises(OpenFDAError):
        client.get("drug/label", limit=1)
    assert client.requests_sent == 3


def test_httpx_encodes_spaces_as_plus_which_openfda_accepts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"meta": {}, "results": []})

    search = 'patient.drug.openfda.generic_name:"metformin" AND serious:1'
    _client(handler).get("drug/event", search=search, limit=1)
    assert "AND" in seen["url"]
    assert "+" in seen["url"]
    assert "%2BAND%2B" not in seen["url"]
    assert " " not in seen["url"].split("?", 1)[1]


def test_api_key_is_sent_but_stripped_from_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "secret-key"
        return httpx.Response(400, text="bad request")

    with pytest.raises(OpenFDAError) as err:
        _client(handler, api_key="secret-key").get("drug/label", limit=1)
    assert "secret-key" not in str(err.value)


def _fat_label() -> dict:
    return {
        "set_id": "set-1",
        "id": "id-1",
        "version": "7",
        "effective_time": "20240101",
        "openfda": {
            "generic_name": ["WARFARIN SODIUM"],
            "brand_name": ["COUMADIN"],
            "manufacturer_name": ["BRISTOL MYERS SQUIBB"],
            "application_number": ["NDA009218"],
            "product_type": ["HUMAN PRESCRIPTION DRUG"],
            "rxcui": ["11289"],
            "pharm_class_epc": ["Vitamin K Antagonist [EPC]"],
        },
        "boxed_warning": ["WARNING: BLEEDING RISK"],
        "indications_and_usage": ["indicated for prophylaxis"],
        "contraindications": ["pregnancy"],
        "warnings_and_cautions": ["tissue necrosis"],
        "warnings": ["legacy warning section"],
        "adverse_reactions": ["hemorrhage"],
        "dosage_and_administration": ["2 to 5 mg"],
        "clinical_studies": ["X" * 5000],
        "spl_medguide": ["Y" * 5000],
    }


def test_label_trim_keeps_fields_tasks_cite_and_drops_the_rest():
    body = trim_response("drug/label", {"meta": {"results": {"total": 1}}, "results": [_fat_label()]}, is_count=False)
    record = body["results"][0]
    assert record["set_id"] == "set-1"
    assert record["version"] == "7"
    assert record["effective_time"] == "20240101"
    assert record["openfda"]["generic_name"] == ["WARFARIN SODIUM"]
    assert record["openfda"]["brand_name"] == ["COUMADIN"]
    assert record["openfda"]["manufacturer_name"] == ["BRISTOL MYERS SQUIBB"]
    for field in (
        "boxed_warning",
        "indications_and_usage",
        "contraindications",
        "warnings_and_cautions",
        "adverse_reactions",
        "dosage_and_administration",
    ):
        assert field in record
    assert "rxcui" not in record["openfda"]
    assert "clinical_studies" not in record
    assert "spl_medguide" not in record
    assert body["meta"]["results"]["total"] == 1


def test_count_responses_pass_through_untrimmed():
    raw = {"results": [{"term": "NAUSEA", "count": 30164}]}
    assert trim_response("drug/event", raw, is_count=True) is raw


def test_snapshot_miss_when_offline(tmp_path: Path):
    snap = DiskSnapshot("2026-10-15", tmp_path)
    with pytest.raises(SnapshotMiss):
        snap.get("drug/label", search='openfda.generic_name:"warfarin"', limit=LABEL_LIMIT)


def test_disk_snapshot_records_trimmed_body_and_then_works_offline(tmp_path: Path):
    class FakeClient:
        last_raw_bytes = 9000

        def get(self, endpoint, **kwargs):
            if kwargs.get("count"):
                return {"results": [{"term": "NAUSEA", "count": 3}]}
            return {
                "meta": {"results": {"total": 1, "skip": 0, "limit": 1}, "last_updated": "2026-10-14"},
                "results": [_fat_label()],
            }

    snap = DiskSnapshot("2026-10-15", tmp_path, client=FakeClient())
    body = snap.get("drug/label", search='openfda.generic_name:"warfarin"', limit=1000)
    assert "clinical_studies" not in body["results"][0]
    assert body["results"][0]["boxed_warning"] == ["WARNING: BLEEDING RISK"]
    snap.get(
        "drug/event",
        search='patient.drug.openfda.generic_name:"warfarin"',
        count="patient.reaction.reactionmeddrapt.exact",
        limit=10,
    )

    class Exploding:
        def get(self, endpoint, **kwargs):
            raise AssertionError("offline read must not call the client")

    snap.client = Exploding()
    again = snap.get("drug/label", search='openfda.generic_name:"warfarin"', limit=1000)
    assert again == body
    rows = snap.write_manifest()
    count_row = next(row for row in rows if row["count"])
    assert count_row["total"] is None
    assert count_row["n_results"] == 1


def test_same_record_gzips_to_the_same_bytes(tmp_path: Path):
    snap = DiskSnapshot("v", tmp_path)
    record = {"body": {"results": []}, "fetched_at": "2026-10-15T00:00:00+00:00", "raw_bytes": 2}
    first = tmp_path / "a.json.gz"
    second = tmp_path / "b.json.gz"
    snap._write(first, record)
    snap._write(second, record)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(gzip.decompress(first.read_bytes()))
    assert payload["fetched_at"] == "2026-10-15T00:00:00+00:00"


def test_concurrent_snapshot_writes_use_distinct_atomic_temps(tmp_path: Path, monkeypatch):
    snap = DiskSnapshot("v", tmp_path)
    target = tmp_path / "v" / "responses" / "same.json.gz"
    barrier = threading.Barrier(2)
    original_tempfile = store_module.tempfile.NamedTemporaryFile
    temp_names = []

    def create_simultaneous_temp(*args, **kwargs):
        handle = original_tempfile(*args, **kwargs)
        temp_names.append(Path(handle.name))
        barrier.wait(timeout=5)
        return handle

    monkeypatch.setattr(store_module.tempfile, "NamedTemporaryFile", create_simultaneous_temp)
    records = [
        {"fetched_at": f"2026-10-{day}T00:00:00+00:00", "body": {"results": [day]}}
        for day in ("15", "16")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(snap._write, target, record) for record in records]
        for future in futures:
            future.result()

    assert len(temp_names) == 2
    assert len(set(temp_names)) == 2
    stored = json.loads(gzip.decompress(target.read_bytes()))
    assert stored in records
    assert list(target.parent.glob(".*.tmp")) == []
