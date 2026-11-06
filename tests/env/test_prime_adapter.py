"""Public verifiers wrapper: honest FG-9101 through FDAEnv into score(), extra-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fdagym.demo import HONEST_TOTAL_MIN, PUBLIC_ID_FIRST, PUBLIC_ID_LAST, DemoError, load_public_tasks
from fdagym.env import TOOL_BUDGET
from fdagym.prime.episode import (
    ADAPTER_TOOLS,
    DEFAULT_TASK_ID,
    grade_episode,
    make_public_env,
    play_public_adapter_episode,
    step_tool,
)
from fdagym.rewards.version import GRADER_VERSION
from fdagym.tools import TOOLS

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT = ROOT / "data" / "snapshots"


def test_adapter_tools_are_the_five_env_tools():
    names = tuple(item["function"]["name"] for item in TOOLS)
    assert ADAPTER_TOOLS == names
    assert ADAPTER_TOOLS == (
        "search_drug_label",
        "get_label_section",
        "query_adverse_events",
        "search_approvals",
        "submit_answer",
    )
    assert TOOL_BUDGET == 8


def test_fg9101_through_adapter_scores_one_via_score():
    before = set(sys.modules)
    reward = play_public_adapter_episode(
        DEFAULT_TASK_ID,
        public_path=PUBLIC,
        snapshot_root=SNAPSHOT,
    )
    added = sorted(set(sys.modules) - before)
    assert not any(name == "verifiers" or name.startswith("verifiers.") for name in added)
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0
    assert reward["citation"] == 1.0
    assert reward["format"] == 1.0
    assert reward["efficiency"] == 1.0
    assert reward["grader_version"] == GRADER_VERSION == "5"


def test_fg9101_adapter_episode_is_three_faers_queries_then_submit():
    env, task = make_public_env(DEFAULT_TASK_ID, public_path=PUBLIC, snapshot_root=SNAPSHOT)
    assert task["id"] == "FG-9101"
    hops = []
    for call in task["oracle_calls"]:
        _obs, _reward, done, _info = step_tool(env, call["tool"], dict(call["arguments"]))
        assert done is False
        hops.append((call["tool"], call["arguments"]["generic_name"]))
    assert hops == [
        ("query_adverse_events", "abacavir"),
        ("query_adverse_events", "pravastatin"),
        ("query_adverse_events", "aripiprazole"),
    ]
    from fdagym.demo import gold_submit

    _obs, _reward, done, info = step_tool(env, "submit_answer", gold_submit(task, env.steps))
    assert done is True
    graded = grade_episode(task, env)
    assert graded["total"] == 1.0
    assert info["reward"]["total"] == 1.0
    assert [step["tool"] for step in env.steps] == [
        "query_adverse_events",
        "query_adverse_events",
        "query_adverse_events",
        "submit_answer",
    ]


def test_every_public_task_replays_through_prime_adapter_to_one():
    assert GRADER_VERSION == "5"
    tasks = load_public_tasks(PUBLIC)
    expected = [f"FG-{n}" for n in range(PUBLIC_ID_FIRST, PUBLIC_ID_LAST + 1)]
    assert sorted(tasks) == expected
    failures = []
    for task_id in expected:
        reward = play_public_adapter_episode(
            task_id,
            public_path=PUBLIC,
            snapshot_root=SNAPSHOT,
        )
        if reward["total"] < HONEST_TOTAL_MIN or reward["grader_version"] != GRADER_VERSION:
            failures.append(f"{task_id} total={reward['total']} grader_version={reward['grader_version']}")
    assert failures == []


def test_adapter_rejects_ids_outside_the_public_sheet():
    with pytest.raises(DemoError, match="published public sheet"):
        make_public_env("FG-0001", public_path=PUBLIC, snapshot_root=SNAPSHOT)
    with pytest.raises(DemoError, match="published public sheet"):
        make_public_env("FG-9194", public_path=PUBLIC, snapshot_root=SNAPSHOT)


def test_importing_fdagym_prime_does_not_load_verifiers():
    from fdagym import prime

    assert prime.DEFAULT_TASK_ID == "FG-9101"
    assert not any(name == "verifiers" or name.startswith("verifiers.") for name in sys.modules)


def test_importing_fdagym_does_not_load_verifiers():
    code = (
        "import sys, fdagym; "
        "mods = [n for n in sys.modules if n == 'verifiers' or n.startswith('verifiers.')]; "
        "print(mods); raise SystemExit(1 if mods else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_taskset_wiring_exports_fdagym_plugin():
    pytest.importorskip("verifiers.v1")
    import fdagym as pkg
    from fdagym.prime.servers.tools import FDAGymToolset
    from fdagym.prime.taskset import FDAGymTaskset

    assert pkg.FDAGymTaskset is FDAGymTaskset
    assert "FDAGymTaskset" in pkg.__all__
    names = [name for name, fn in vars(FDAGymToolset).items() if callable(fn) and getattr(fn, "tool", False)]
    assert set(names) == set(ADAPTER_TOOLS)
