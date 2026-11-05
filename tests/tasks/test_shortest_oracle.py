"""Stored compare/approval oracles are the shortest path the current grader already accepts.

Generic-name summaries expose original_approval, so hops._visible_year does not
need a follow-up search_approvals(application_number=…) hop.
"""

from pathlib import Path

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.tasks.hard_solver import parse_question
from tests.integration.test_oracle import citations_from_outputs, gold_answer, play

ROOT = Path(__file__).resolve().parents[2]
EXAM = ROOT / "data" / "snapshots" / "2026-10-15"


def _detail(call: dict) -> bool:
    return call.get("tool") == "search_approvals" and "application_number" in (call.get("arguments") or {})


def _play_prefix(env: FDAEnv, task: dict, n: int) -> dict:
    env.reset(task["id"])
    for call in task["oracle_calls"][:n]:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None, f"{task['id']} {obs.get('error')}"
        assert done is False
    visible = _visible_gold(env.steps, task)
    citations = citations_from_outputs(env.steps, {**task, "gold_citations": visible}) if visible else []
    _obs, _reward, done, info = env.step(
        {"tool": "submit_answer", "arguments": {"answer": gold_answer(task), "citations": citations}}
    )
    assert done is True
    return info["reward"]


def _visible_gold(steps: list[dict], task: dict) -> list[dict]:
    """Gold citation rows whose field actually appeared, so a prefix can still submit."""
    kept = []
    for gold in task["gold_citations"]:
        try:
            citations_from_outputs(steps, {**task, "gold_citations": [gold]})
        except AssertionError:
            continue
        kept.append(gold)
    return kept


def test_public_earliest_boxed_oracle_is_the_shortest_proving_path():
    assert (EXAM / "manifest.jsonl").exists()
    tasks = [task for task in load_split("public") if parse_question(task["question"])["variant"] == "earliest_boxed"]
    assert len(tasks) == 16
    env = FDAEnv(DiskSnapshot("2026-10-15"), tasks, run_id="n2-public-oracle", tool_budget=8)
    for task in tasks:
        assert task["difficulty_features"]["min_tool_calls"] == 6
        assert len(task["oracle_calls"]) == 6
        assert not any(_detail(call) for call in task["oracle_calls"])
        full = play(env, task, gold_answer(task))
        assert full["total"] == 1.0, (task["id"], full["total"], full["notes"][:6])
        for n in range(len(task["oracle_calls"])):
            prefix = _play_prefix(env, task, n)
            assert prefix["total"] < 0.95, (task["id"], n, prefix["total"])


def test_fg9102_documents_that_the_old_detail_hop_was_unnecessary():
    """FG-9102 previously stored search_approvals(application_number=NDA016012) as an extra hop."""
    assert (EXAM / "manifest.jsonl").exists()
    task = next(item for item in load_split("public") if item["id"] == "FG-9102")
    assert parse_question(task["question"])["variant"] == "earliest_boxed"
    assert not any(_detail(call) for call in task["oracle_calls"])
    year_cite = next(row for row in task["gold_citations"] if row["endpoint"] == "drug/drugsfda")
    assert year_cite["fields"] == ["original_approval"]
    assert "NDA016012" in year_cite["ids"]
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="n2-fg9102", tool_budget=8)
    reward = play(env, task, gold_answer(task))
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0
    assert reward["efficiency"] == 1.0


def test_fg0031_keeps_one_submissions_hop_and_stays_labeled_hard():
    """Year questions still cite submissions, so one detail hop stays. Extra winner apps do not."""
    assert (EXAM / "manifest.jsonl").exists()
    task = next(item for item in load_split("test") if item["id"] == "FG-0031")
    assert task["type"] == "approval"
    # Label is left as stored (medium on this sheet). Only oracle/min change.
    assert task["difficulty"] == "medium"
    assert task["difficulty_features"]["min_tool_calls"] == 2
    assert len(task["oracle_calls"]) == 2
    assert not _detail(task["oracle_calls"][0])
    assert _detail(task["oracle_calls"][1])
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="n2-fg0031", tool_budget=8)
    full = play(env, task, gold_answer(task))
    assert full["total"] == 1.0
    prefix = _play_prefix(env, task, 1)
    assert prefix["total"] < 0.95
