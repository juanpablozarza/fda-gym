import json

import pytest

from fdagym.agents import baseline
from fdagym.agents.baseline import run_baseline
from scripts import run_baseline as baseline_cli
from tests.agents.fake import ScriptedCompletion
from tests.agents.test_litellm_agent import _oracle_calls
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot, task_nausea
from tests.test_schemas import validator


def _scripted_run(tmp_path, tasks, snapshot, calls, *, run_id="offline", models=None, seeds=None, cost_cap=5.0, cost=0.0001):
    return run_baseline(
        split="test",
        models=models or ["scripted-oracle"],
        seeds=seeds or [1],
        run_id=run_id,
        results_dir=tmp_path,
        completion=ScriptedCompletion(calls, cost=cost),
        cost_cap=cost_cap,
        snapshot=snapshot,
        tasks=tasks,
    )


def test_run_baseline_writes_valid_jsonl_and_leaderboard(tmp_path):
    snapshot = JsonSnapshot()
    calls = _oracle_calls(snapshot, ALL_TASKS, EXAMPLE_TASK)
    summary = _scripted_run(tmp_path, [EXAMPLE_TASK], snapshot, calls)
    lines = (tmp_path / "offline.jsonl").read_text().splitlines()
    assert len(lines) == 1
    trajectory = json.loads(lines[0])
    validator("trajectory.schema.json").validate(trajectory)
    assert trajectory["reward"]["total"] == 1.0
    assert summary["models"]["scripted-oracle"]["n"] == 1
    assert summary["models"]["scripted-oracle"]["mean_reward"] == 1.0
    assert summary["models"]["scripted-oracle"]["pass_rate"] == 1.0
    board = json.loads((tmp_path / "offline" / "leaderboard.json").read_text())
    assert board[0]["name"] == "scripted-oracle"
    assert board[0]["kind"] == "baseline"
    assert board[0]["mean_reward"] == 1.0
    assert board[0]["pass"] == 1.0
    assert "citations" in board[0] and "tools" in board[0]
    assert trajectory["id"] == "offline/scripted-oracle/FG-0001/s1"
    assert trajectory["run_id"] == "offline"


def test_run_baseline_writes_a_task_sidecar_including_skipped_types(tmp_path):
    snapshot = JsonSnapshot()
    open_task = dict(EXAMPLE_TASK)
    open_task["id"] = "FG-0099"
    open_task["type"] = "label_open"
    tasks = [EXAMPLE_TASK, open_task]
    calls = _oracle_calls(snapshot, tasks, EXAMPLE_TASK)
    _scripted_run(tmp_path, tasks, snapshot, calls, run_id="sidecar")
    for path in (tmp_path / "sidecar" / "tasks.json", tmp_path / "tasks.json"):
        cards = json.loads(path.read_text())
        assert [row["id"] for row in cards] == ["FG-0001", "FG-0099"]
        assert set(cards[0]) == {"id", "question", "type", "sources", "difficulty", "entity"}
        assert cards[1]["type"] == "label_open"
        assert cards[0]["question"] == EXAMPLE_TASK["question"]


def test_run_baseline_skips_label_open_and_counts_it(tmp_path):
    snapshot = JsonSnapshot()
    open_task = dict(EXAMPLE_TASK)
    open_task["id"] = "FG-0099"
    open_task["type"] = "label_open"
    tasks = [EXAMPLE_TASK, open_task]
    calls = _oracle_calls(snapshot, tasks, EXAMPLE_TASK)
    summary = _scripted_run(tmp_path, tasks, snapshot, calls, run_id="skip-open")
    assert summary["skipped"]["label_open"] == 1
    assert summary["models"]["scripted-oracle"]["n"] == 1
    assert len((tmp_path / "skip-open.jsonl").read_text().splitlines()) == 1


def test_run_baseline_is_resume_safe(tmp_path):
    snapshot = JsonSnapshot()
    calls = _oracle_calls(snapshot, ALL_TASKS, EXAMPLE_TASK)
    _scripted_run(tmp_path, [EXAMPLE_TASK], snapshot, calls, run_id="resume")
    summary = _scripted_run(tmp_path, [EXAMPLE_TASK], snapshot, calls, run_id="resume")
    assert summary["skipped"]["already_done"] == 1
    assert len((tmp_path / "resume.jsonl").read_text().splitlines()) == 1


def test_run_baseline_aborts_at_cost_cap(tmp_path):
    snapshot = JsonSnapshot()
    nausea = task_nausea()
    tasks = [EXAMPLE_TASK, nausea]
    war_calls = _oracle_calls(snapshot, tasks, EXAMPLE_TASK)
    nausea_calls = _oracle_calls(snapshot, tasks, nausea)
    summary = run_baseline(
        split="test",
        models=["scripted-oracle"],
        seeds=[1],
        run_id="cap",
        results_dir=tmp_path,
        completion=ScriptedCompletion(war_calls + nausea_calls, cost=2.0),
        cost_cap=1.5,
        snapshot=snapshot,
        tasks=tasks,
    )
    assert summary["aborted"]
    assert summary["models"]["scripted-oracle"]["n"] == 1
    assert "cost cap" in summary["aborted"]


def test_run_baseline_persists_episode_failures_and_marks_run_incomplete(tmp_path, monkeypatch):
    class FailingAgent:
        def __init__(self, _model, *, completion=None):
            pass

        def run(self, _env, _task_id):
            raise RuntimeError("provider failed with secret token")

    monkeypatch.setattr(baseline, "LiteLLMAgent", FailingAgent)
    summary = run_baseline(
        split="test",
        models=["broken-model"],
        seeds=[7],
        run_id="failed-episode",
        results_dir=tmp_path,
        snapshot=JsonSnapshot(),
        tasks=[EXAMPLE_TASK],
    )

    failure_path = tmp_path / "failed-episode" / "failures.jsonl"
    failure = json.loads(failure_path.read_text().splitlines()[0])
    saved_summary = json.loads((tmp_path / "failed-episode" / "summary.json").read_text())
    assert failure["model"] == "broken-model"
    assert failure["task_id"] == EXAMPLE_TASK["id"]
    assert failure["seed"] == 7
    assert failure["error_type"] == "RuntimeError"
    assert "secret token" not in failure_path.read_text()
    assert summary["failure_count"] == saved_summary["failure_count"] == 1
    assert summary["complete"] is False
    assert summary["status"] == "incomplete"


def test_run_baseline_rejects_an_empty_task_split(tmp_path):
    with pytest.raises(ValueError, match="task split 'test' is empty"):
        run_baseline(
            split="test",
            models=["unused"],
            seeds=[1],
            run_id="empty",
            results_dir=tmp_path,
            snapshot=JsonSnapshot(),
            tasks=[],
        )


def test_baseline_cli_returns_nonzero_for_recorded_failures(monkeypatch):
    monkeypatch.setattr(
        baseline_cli,
        "run_baseline",
        lambda **_kwargs: {"failure_count": 1, "failures": [], "aborted": None},
    )
    assert baseline_cli.main(["--run-id", "incomplete"]) == 2
