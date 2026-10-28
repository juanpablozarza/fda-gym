"""The three unnamed sponsor questions. The oracle scores 1.0 on the freeze."""

import json
from pathlib import Path

import pytest

from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.rewards import score
from fdagym.rewards.hops import hop_score
from fdagym.rewards.version import DATA_VERSION, GRADER_VERSION
from fdagym.tasks.hard_solver import parse_question
from fdagym.tasks.validate import require_oracle_calls

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data" / "tasks" / "sponsor_lookup.jsonl"
pytestmark = pytest.mark.skipif(not TASKS.exists(), reason="sponsor_lookup.jsonl is not in this tree")
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15"


def _tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]


@pytest.mark.skipif(not (SNAPSHOT / "manifest.jsonl").exists(), reason="snapshot absent")
def test_sponsor_lookup_oracle_scores_one():
    tasks = _tasks()
    assert [task["id"] for task in tasks] == ["FG-8454", "FG-8455", "FG-8456"]
    taken = set()
    for name in ("hard.jsonl", "hard_train.jsonl", "hard_train_holdout.jsonl", "train.jsonl", "test.jsonl"):
        if not (ROOT / "data" / "tasks" / name).exists():
            continue
        for line in (ROOT / "data" / "tasks" / name).read_text().splitlines():
            if line.strip():
                taken.add(json.loads(line)["id"])
    assert taken.isdisjoint(task["id"] for task in tasks)
    snapshot = DiskSnapshot("2026-10-15")
    for task in tasks:
        require_oracle_calls(task)
        assert task["question"].startswith("The sponsor of application ")
        assert task["entity"]["generic_name"] not in task["question"]
        env = FDAEnv(snapshot, tasks, run_id="sponsor-lookup", seed=0)
        env.reset(task["id"])
        for call in task["oracle_calls"]:
            obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
            assert obs["error"] is None and done is False, obs["error"]
        names = [row["generic_name"] for row in env.steps[1]["output"]["results"]]
        assert task["entity"]["generic_name"] in names
        assert len(names) == 2
        cited = None
        allowed = set(task["gold_citations"][0]["ids"])
        for step in env.steps:
            if step["tool"] != "search_drug_label":
                continue
            if step["arguments"].get("generic_name") != task["entity"]["generic_name"]:
                continue
            for row in step["output"]["results"]:
                if row.get("set_id") in allowed and "boxed_warning" in row.get("sections", []):
                    cited = row["set_id"]
                    break
        assert cited
        env.step(
            {
                "tool": "submit_answer",
                "arguments": {
                    "answer": task["gold"]["value"],
                    "citations": [
                        {"endpoint": "drug/label", "id": cited, "field": "sections", "quote": "boxed_warning"}
                    ],
                },
            }
        )
        assert env.trajectory["reward"]["total"] == 1.0
        assert env.trajectory["reward"]["grader_version"] == GRADER_VERSION
        assert env.trajectory["reward"]["data_version"] == DATA_VERSION
        assert parse_question(task["question"])["variant"] == "sponsor_boxed"
        hop = hop_score(env.steps, task)
        assert hop.rate == 1.0, hop.notes


def _drive(task: dict, calls: list[dict]) -> FDAEnv:
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="shortcut", seed=0)
    env.reset(task["id"])
    yes = task["entity"]["generic_name"]
    allowed = set(task["gold_citations"][0]["ids"])
    cited = None
    for call in calls:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None and done is False, obs.get("error")
        if call["tool"] == "search_drug_label" and call["arguments"].get("generic_name") == yes:
            for row in obs["output"]["results"]:
                if row.get("set_id") in allowed and "boxed_warning" in row.get("sections", []):
                    cited = row["set_id"]
    assert cited
    env.step(
        {
            "tool": "submit_answer",
            "arguments": {
                "answer": yes,
                "citations": [{"endpoint": "drug/label", "id": cited, "field": "sections", "quote": "boxed_warning"}],
            },
        }
    )
    return env


def _withheld(reward: dict) -> None:
    assert reward["correctness"] == 0.0
    assert reward["citation"] == 1.0
    assert reward["format"] == 1.0
    assert reward["efficiency"] == 0.0
    assert reward["total"] == 0.35
    assert any("correctness needs the sponsor proof" in note for note in reward["notes"])


@pytest.mark.skipif(not (SNAPSHOT / "manifest.jsonl").exists(), reason="snapshot absent")
@pytest.mark.parametrize("task_id", ["FG-8454", "FG-8455", "FG-8456"])
def test_skipping_the_sponsor_hop_scores_citation_and_format_only(task_id):
    task = next(item for item in _tasks() if item["id"] == task_id)
    yes = task["entity"]["generic_name"]
    label = {"tool": "search_drug_label", "arguments": {"generic_name": yes, "has_section": "boxed_warning"}}
    other = next(
        call["arguments"]["generic_name"]
        for call in task["oracle_calls"]
        if call["tool"] == "search_drug_label" and call["arguments"]["generic_name"] != yes
    )
    application = task["oracle_calls"][0]["arguments"]["application_number"]

    shortcut = _drive(task, [label]).trajectory["reward"]
    _withheld(shortcut)
    assert any("1 distinct tool calls" in note for note in shortcut["notes"])

    repeated = _drive(task, [label, label, label, label]).trajectory["reward"]
    _withheld(repeated)
    assert any("1 distinct tool calls" in note for note in repeated["notes"])

    wrong_sponsor = _drive(
        task,
        [
            {"tool": "search_approvals", "arguments": {"application_number": application}},
            {"tool": "search_approvals", "arguments": {"sponsor_name": "NOT THE SPONSOR"}},
            label,
            {"tool": "search_drug_label", "arguments": {"generic_name": other, "has_section": "boxed_warning"}},
        ],
    ).trajectory["reward"]
    _withheld(wrong_sponsor)
    assert any("hop sponsor_list: miss" in note for note in wrong_sponsor["notes"])


@pytest.mark.skipif(not (SNAPSHOT / "manifest.jsonl").exists(), reason="snapshot absent")
@pytest.mark.parametrize("task_id", ["FG-8454", "FG-8455", "FG-8456"])
def test_one_boxed_search_and_three_plain_searches(task_id):
    task = next(item for item in _tasks() if item["id"] == task_id)
    yes = task["entity"]["generic_name"]
    env = _drive(
        task,
        [
            {"tool": "search_drug_label", "arguments": {"generic_name": yes, "has_section": "boxed_warning"}},
            {"tool": "search_drug_label", "arguments": {"generic_name": yes}},
            {"tool": "search_approvals", "arguments": {"generic_name": yes}},
            {"tool": "query_adverse_events", "arguments": {"generic_name": yes, "mode": "top_reactions"}},
        ],
    )
    _withheld(env.trajectory["reward"])
    assert any("4 distinct tool calls" in note for note in env.trajectory["reward"]["notes"])
    dense = score(
        {"steps": env.steps, "final_answer": env.final_answer},
        task,
        dense=True,
    )
    assert dense["correctness"] == 0.0
    assert dense["total"] == 0.35


@pytest.mark.skipif(not (SNAPSHOT / "manifest.jsonl").exists(), reason="snapshot absent")
@pytest.mark.parametrize("task_id", ["FG-8454", "FG-8455", "FG-8456"])
def test_repeating_an_oracle_call_still_scores_one(task_id):
    task = next(item for item in _tasks() if item["id"] == task_id)
    reward = _drive(task, [*task["oracle_calls"], task["oracle_calls"][0]]).trajectory["reward"]
    assert reward["correctness"] == 1.0
    assert reward["efficiency"] == 1.0
    assert reward["total"] == 1.0
    assert any("4 distinct tool calls" in note for note in reward["notes"])


def test_sponsor_name_is_exclusive():
    from fdagym.tools import EpisodeState, execute
    from tests.env.support import MapSnapshot

    state = EpisodeState(MapSnapshot({}))
    _output, err = execute(
        state,
        "search_approvals",
        {"application_number": "NDA022524", "sponsor_name": "AQUESTIVE"},
        step_index=1,
    )
    assert err and "exactly one" in err
