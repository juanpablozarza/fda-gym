"""Scripted episodes on the 10-drug pilot. Skipped when that snapshot directory is absent."""

import time
from pathlib import Path

import pytest

from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.queries import label_search
from tests.env.support import cite, submit

PILOT = Path(__file__).resolve().parents[2] / "data" / "snapshots" / "2026-10-15"

pytestmark = pytest.mark.skipif(not PILOT.is_dir(), reason="pilot snapshot 2026-10-15 is absent")


def _features(min_tool_calls: int, **overrides) -> dict:
    base = {
        "sources": 1,
        "min_tool_calls": min_tool_calls,
        "needs_disambiguation": False,
        "needs_aggregation": False,
        "needs_normalization": False,
        "has_constraint": False,
    }
    base.update(overrides)
    return base


def _task(
    task_id: str,
    question: str,
    drug: str,
    source: str,
    gold: dict,
    citations: dict,
    *,
    answerable: bool,
    min_tool_calls: int,
    task_type: str,
) -> dict:
    return {
        "id": task_id,
        "type": task_type,
        "question": question,
        "sources": [source],
        "entity": {"generic_name": drug},
        "split": "test",
        "answerable": answerable,
        "difficulty": "easy",
        "difficulty_features": _features(min_tool_calls),
        "gold": gold,
        "gold_citations": [citations],
        "snapshot_version": "2026-10-15",
    }


def _episode(task: dict):
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task])
    env.reset(task["id"])
    return env


def test_pilot_warfarin_boxed_warning_scores_one():
    task = _task(
        "FG-PILOT-WARFARIN",
        "Does the FDA label for warfarin carry a boxed warning, and if so, about what risk?",
        "warfarin",
        "drug/label",
        {"kind": "boolean", "value": True, "keywords": ["bleeding"]},
        {"endpoint": "drug/label", "ids": [], "fields": ["boxed_warning"]},
        answerable=True,
        min_tool_calls=2,
        task_type="boxed_warning",
    )
    env = _episode(task)
    obs, _reward, done, _info = env.step(
        {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}}
    )
    assert done is False
    set_id = obs["output"]["results"][0]["set_id"]
    obs, _reward, done, _info = env.step(
        {"tool": "get_label_section", "arguments": {"set_id": set_id, "section": "boxed_warning"}}
    )
    text = obs["output"]["boxed_warning"]
    assert "bleeding" in text.lower()
    _obs, reward, done, info = env.step(
        submit(
            "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
            [cite("drug/label", set_id, "boxed_warning", text[:80])],
        )
    )
    assert done is True
    assert reward == 1.0
    assert info["reward"]["correctness"] == 1.0


def test_pilot_atorvastatin_absence_is_cited_from_sections():
    task = _task(
        "FG-PILOT-ATORVA",
        "Does atorvastatin's label carry a boxed warning?",
        "atorvastatin",
        "drug/label",
        {"kind": "boolean", "value": False},
        {"endpoint": "drug/label", "ids": [], "fields": ["sections"]},
        answerable=True,
        min_tool_calls=1,
        task_type="boxed_warning",
    )
    query = label_search("atorvastatin")
    body = DiskSnapshot("2026-10-15").get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
    records = [row for row in body["results"] if isinstance(row, dict)]
    assert len(records) == 259
    assert all("boxed_warning" not in row for row in records)

    env = _episode(task)
    obs, _reward, _done, _info = env.step(
        {"tool": "search_drug_label", "arguments": {"generic_name": "atorvastatin", "page": 1}}
    )
    shown = obs["output"]["results"]
    assert len(shown) == 10
    assert obs["output"]["meta"]["results"]["total"] == 259
    row = shown[0]
    assert "boxed_warning" not in row["sections"]
    quote = row["sections"][0]
    _obs, reward, _done, info = env.step(submit("No", [cite("drug/label", row["set_id"], "sections", quote)]))
    assert reward == 1.0
    assert info["reward"]["correctness"] == 1.0
    assert any("polarity match" in note for note in info["reward"]["notes"])

    yes = _episode(task)
    yes.step({"tool": "search_drug_label", "arguments": {"generic_name": "atorvastatin", "page": 1}})
    _obs, _reward, _done, info = yes.step(submit("Yes", [cite("drug/label", row["set_id"], "sections", quote)]))
    assert info["reward"]["correctness"] == 0.0
    assert any("does not start with no" in note for note in info["reward"]["notes"])

    page_ids = {item["set_id"] for item in shown}
    missing = next(record["set_id"] for record in records if record.get("set_id") not in page_ids)
    missed = _episode(task)
    missed.step({"tool": "search_drug_label", "arguments": {"generic_name": "atorvastatin", "page": 1}})
    _obs, _reward, _done, info = missed.step(submit("No", [cite("drug/label", missing, "sections", quote)]))
    assert info["reward"]["citation"] == 0.0
    assert info["reward"]["correctness"] == 0.0
    assert any("did not appear" in note for note in info["reward"]["notes"])


def test_pilot_xanomeline_2020_not_found_is_correct():
    task = _task(
        "FG-PILOT-XAN",
        "How many FAERS reports for xanomeline were received in 2020?",
        "xanomeline",
        "drug/event",
        {"kind": "exact", "value": "NOT_FOUND"},
        {"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]},
        answerable=False,
        min_tool_calls=1,
        task_type="faers_count",
    )
    env = _episode(task)
    obs, _reward, _done, _info = env.step(
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "xanomeline", "mode": "count", "year": 2020},
        }
    )
    assert obs["output"]["meta"]["results"]["total"] == 0
    _obs, reward, _done, info = env.step(
        submit("NOT_FOUND", [cite("drug/event", obs["output"]["id"], "meta.results.total", "0")])
    )
    assert reward == 1.0
    assert info["reward"]["correctness"] == 1.0


def test_pilot_metformin_label_search_pages_and_reports_latency():
    task = _task(
        "FG-PILOT-MET",
        "How many metformin labels are in the snapshot?",
        "metformin",
        "drug/label",
        {"kind": "number", "value": 365, "tolerance": 0},
        {"endpoint": "drug/label", "ids": [], "fields": ["meta.results.total"]},
        answerable=True,
        min_tool_calls=1,
        task_type="label_search",
    )
    env = _episode(task)
    started = time.perf_counter()
    obs, _reward, done, _info = env.step(
        {"tool": "search_drug_label", "arguments": {"generic_name": "metformin", "page": 1}}
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert done is False
    page1 = obs["output"]["results"]
    assert len(page1) == 10
    assert obs["output"]["meta"]["results"]["total"] == 365
    obs, _reward, _done, _info = env.step(
        {"tool": "search_drug_label", "arguments": {"generic_name": "metformin", "page": 2}}
    )
    page2 = obs["output"]["results"]
    assert len(page2) == 10
    assert {row["set_id"] for row in page1}.isdisjoint({row["set_id"] for row in page2})
    print(f"metformin search_drug_label page 1: {elapsed_ms:.0f} ms")
