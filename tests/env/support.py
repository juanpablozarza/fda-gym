"""Shared snapshot and tasks for env and reward tests. Not part of the package."""

import json
from pathlib import Path

from fdagym.env import FDAEnv
from fdagym.snapshot import SnapshotMiss

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "openfda.json"
WARFARIN_SET = "0cbce382-9c88-4f58-ae0f-532a841e8f95"
EXAMPLE_TASK = json.loads((ROOT / "schemas" / "examples" / "task.example.json").read_text())


class JsonSnapshot:
    def __init__(self, path: Path = FIXTURE):
        raw = json.loads(path.read_text())
        self._rows = {}
        for key, body in raw.items():
            endpoint, search, count, limit = json.loads(key)
            self._rows[(endpoint, search, count, limit)] = body
        self.version = "2026-10-15"

    def get(self, endpoint, *, search=None, count=None, limit=1, skip=0):
        if skip:
            raise SnapshotMiss((endpoint, search, count, limit, skip))
        key = (endpoint, search, count, limit)
        if key not in self._rows:
            raise SnapshotMiss(key)
        return self._rows[key]


class MapSnapshot:
    def __init__(self, rows: dict, version: str = "test"):
        self._rows = rows
        self.version = version
        self.calls: list[tuple] = []

    def get(self, endpoint, *, search=None, count=None, limit=1, skip=0):
        self.calls.append((endpoint, search, count, limit, skip))
        key = (endpoint, search, count, limit)
        if key not in self._rows:
            raise SnapshotMiss(key)
        return self._rows[key]


def _features(**overrides) -> dict:
    base = {
        "sources": 1,
        "min_tool_calls": 1,
        "needs_disambiguation": False,
        "needs_aggregation": False,
        "needs_normalization": False,
        "has_constraint": False,
    }
    base.update(overrides)
    return base


def task_nausea() -> dict:
    return {
        "id": "FG-0002",
        "type": "faers_top_reaction",
        "question": "What is the most-reported FAERS reaction for metformin?",
        "sources": ["drug/event"],
        "entity": {"generic_name": "metformin"},
        "split": "train",
        "answerable": True,
        "difficulty": "medium",
        "difficulty_features": _features(needs_aggregation=True, needs_normalization=True),
        "gold": {"kind": "exact", "value": "NAUSEA", "aliases": ["nausea"]},
        "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["results"]}],
        "oracle_calls": [
            {"tool": "query_adverse_events", "arguments": {"generic_name": "metformin", "mode": "top_reactions"}}
        ],
        "snapshot_version": "2026-10-15",
    }


def task_count() -> dict:
    return {
        "id": "FG-0003",
        "type": "faers_count",
        "question": "How many serious metformin FAERS reports were received in 2025?",
        "sources": ["drug/event"],
        "entity": {"generic_name": "metformin"},
        "split": "train",
        "answerable": True,
        "difficulty": "hard",
        "difficulty_features": _features(needs_aggregation=True, has_constraint=True),
        "gold": {"kind": "number", "value": 20254, "tolerance": 0},
        "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}],
        "oracle_calls": [
            {
                "tool": "query_adverse_events",
                "arguments": {"generic_name": "metformin", "mode": "count", "serious": True, "year": 2025},
            }
        ],
        "snapshot_version": "2026-10-15",
    }


def task_unanswerable() -> dict:
    return {
        "id": "FG-0004",
        "type": "faers_count",
        "question": "How many FAERS reports for lecanemab were received in 2020?",
        "sources": ["drug/event"],
        "entity": {"generic_name": "lecanemab"},
        "split": "test",
        "answerable": False,
        "difficulty": "hard",
        "difficulty_features": _features(needs_aggregation=True, has_constraint=True),
        "gold": {"kind": "exact", "value": "NOT_FOUND"},
        "gold_citations": [
            {"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}
        ],
        "oracle_calls": [
            {
                "tool": "query_adverse_events",
                "arguments": {"generic_name": "lecanemab", "mode": "count", "year": 2020},
            }
        ],
        "snapshot_version": "2026-10-15",
    }


def task_approval() -> dict:
    return {
        "id": "FG-0005",
        "type": "approval",
        "question": "What year was semaglutide's Ozempic application originally approved?",
        "sources": ["drug/drugsfda"],
        "entity": {"generic_name": "semaglutide", "brand_name": "Ozempic"},
        "split": "test",
        "answerable": True,
        "difficulty": "easy",
        "difficulty_features": _features(),
        "gold": {"kind": "exact", "value": "2017", "aliases": ["2017-12-05"]},
        "gold_citations": [
            {"endpoint": "drug/drugsfda", "ids": ["NDA209637"], "fields": ["submissions"]}
        ],
        "oracle_calls": [{"tool": "search_approvals", "arguments": {"generic_name": "semaglutide"}}],
        "snapshot_version": "2026-10-15",
    }


ALL_TASKS = [EXAMPLE_TASK, task_nausea(), task_count(), task_unanswerable(), task_approval()]


def drive(task_id: str, actions: list[dict]):
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset(task_id)
    observations = []
    info: dict = {}
    done = False
    for action in actions:
        obs, _reward, done, info = env.step(action)
        observations.append(obs)
    if not done:
        raise AssertionError("episode did not finish")
    return observations, info["reward"], env


def cite(endpoint: str, cid: str, field: str, quote: str) -> dict:
    return {"endpoint": endpoint, "id": cid, "field": field, "quote": quote}


def submit(answer, citations: list[dict]) -> dict:
    return {"tool": "submit_answer", "arguments": {"answer": answer, "citations": citations}}
