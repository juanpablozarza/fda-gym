"""The public demo is keyless, stays on the published sheet, and prints score keys."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fdagym.demo import (
    DEFAULT_TASK_ID,
    HONEST_TOTAL_MIN,
    PUBLIC_ID_FIRST,
    PUBLIC_ID_LAST,
    SCORE_KEYS,
    DemoError,
    is_public_task_id,
    load_public_tasks,
    play_public_task,
    resolve_task,
    run_episode,
    snapshot_for_demo,
)
from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.gold_cite import citations_from_outputs
from fdagym.rewards.version import GRADER_VERSION

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "demo_public_episode.py"
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT = ROOT / "data" / "snapshots"
SECRET_ENV = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENFDA_API_KEY",
    "WANDB_API_KEY",
)


def _clean_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in SECRET_ENV}
    env["PYTHONPATH"] = str(ROOT)
    return env


def test_default_task_id_is_the_short_public_compare():
    assert DEFAULT_TASK_ID == "FG-9101"
    assert is_public_task_id(DEFAULT_TASK_ID)
    assert not is_public_task_id("FG-0001")
    assert not is_public_task_id("FG-9100")
    assert not is_public_task_id("FG-9194")


def test_resolve_task_rejects_ids_outside_the_public_sheet():
    from fdagym.demo import load_public_tasks

    tasks = load_public_tasks(PUBLIC)
    with pytest.raises(DemoError, match="published public sheet"):
        resolve_task(tasks, "FG-0001")
    with pytest.raises(DemoError, match="published public sheet"):
        resolve_task(tasks, "FG-9194")


def test_play_default_public_task_scores_one_without_an_llm(capsys):
    before = set(sys.modules)
    reward = play_public_task(
        DEFAULT_TASK_ID,
        public_path=PUBLIC,
        snapshot_root=SNAPSHOT,
        write=print,
    )
    added = sorted(set(sys.modules) - before)
    llm_added = [
        name
        for name in added
        if name in {"litellm", "openai", "anthropic"}
        or name.startswith(("litellm.", "fdagym.agents.litellm", "fdagym.agents.chat"))
    ]
    assert llm_added == []
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0
    assert reward["citation"] == 1.0
    assert reward["format"] == 1.0
    assert reward["efficiency"] == 1.0
    out = capsys.readouterr().out
    for key in SCORE_KEYS:
        assert key in out
    assert "query_adverse_events" in out
    assert "submit_answer" in out
    assert out.count("score  ") == 4  # 3 hops + submit
    assert "tool_budget=8" in out or f"tool_budget={TOOL_BUDGET}" in out


def test_importing_fdagym_demo_does_not_load_an_llm_client():
    code = (
        "import sys; import fdagym.demo; "
        "mods = [n for n in sys.modules if n in {'litellm', 'openai'} "
        "or n.startswith('fdagym.agents.litellm') or n.startswith('fdagym.agents.chat')]; "
        "print(mods); raise SystemExit(1 if mods else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_demo_script_runs_keyless_on_the_default_public_task():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    for key in SCORE_KEYS:
        assert key in out
    assert "FG-9101" in out
    assert "correctness=1.0000" in out
    assert "total=1.0000" in out
    assert "grader_version=" in out
    assert "litellm" not in out.lower()
    assert "openrouter" not in out.lower()


def test_demo_script_rejects_a_non_public_task_id():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--task-id", "FG-0001"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    assert result.returncode == 2
    assert "published public sheet" in result.stderr


def test_importing_fdagym_demo_does_not_load_train():
    code = (
        "import sys; import fdagym.demo; "
        "mods = [n for n in sys.modules if n == 'fdagym.train' or n.startswith('fdagym.train.')]; "
        "print(mods); raise SystemExit(1 if mods else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_demo_and_trainer_share_citations_from_outputs():
    from fdagym.demo import gold_submit as demo_submit
    from fdagym.gold_cite import gold_submit as shared_submit
    from fdagym.train.gold import citations_from_outputs as train_cite
    from fdagym.train.gold import gold_submit as train_submit

    assert demo_submit is shared_submit
    assert train_submit is shared_submit
    assert train_cite is citations_from_outputs


def test_citations_skip_gold_rows_whose_endpoint_was_never_queried():
    steps = [
        {
            "tool": "search_approvals",
            "arguments": {"generic_name": "sumatriptan"},
            "output": {"application_number": "NDA007337", "sponsor_name": "Glaxo"},
        }
    ]
    task = {
        "id": "FG-9151",
        "gold_citations": [
            {"endpoint": "drug/drugsfda", "ids": ["NDA007337"], "fields": ["sponsor_name"]},
            {"endpoint": "drug/label", "ids": ["abc"], "fields": ["boxed_warning"]},
        ],
    }
    cites = citations_from_outputs(steps, task)
    assert [cite["field"] for cite in cites] == ["sponsor_name"]
    assert cites[0]["id"] == "NDA007337"


def test_citations_still_error_when_a_queried_gold_field_never_appeared():
    steps = [
        {
            "tool": "search_drug_label",
            "arguments": {"generic_name": "sumatriptan"},
            "output": {"set_id": "abc", "indications_and_usage": "migraine"},
        }
    ]
    task = {
        "id": "T",
        "gold_citations": [{"endpoint": "drug/label", "ids": ["abc"], "fields": ["boxed_warning"]}],
    }
    with pytest.raises(RuntimeError, match=r"T gold field \['boxed_warning'\] never appeared"):
        citations_from_outputs(steps, task)


def test_citations_error_when_nothing_remains_after_filtering():
    steps = [
        {
            "tool": "search_approvals",
            "arguments": {"generic_name": "sumatriptan"},
            "output": {"application_number": "NDA007337", "sponsor_name": "Glaxo"},
        }
    ]
    task = {
        "id": "T",
        "gold_citations": [{"endpoint": "drug/label", "ids": ["abc"], "fields": ["boxed_warning"]}],
    }
    with pytest.raises(RuntimeError, match="T no gold field appeared"):
        citations_from_outputs(steps, task)


def test_every_public_task_replays_through_demo_to_one():
    assert GRADER_VERSION == "5"
    tasks = load_public_tasks(PUBLIC)
    expected = [f"FG-{n}" for n in range(PUBLIC_ID_FIRST, PUBLIC_ID_LAST + 1)]
    assert expected[0] == "FG-9101"
    assert expected[-1] == "FG-9193"
    assert len(expected) == 93
    assert sorted(tasks) == expected
    env = FDAEnv(
        snapshot_for_demo(SNAPSHOT),
        list(tasks.values()),
        run_id="public-demo-all",
        tool_budget=TOOL_BUDGET,
    )
    failures = []
    for task_id in expected:
        reward = run_episode(env, tasks[task_id], write=lambda _line: None)
        if reward["total"] < HONEST_TOTAL_MIN or reward["grader_version"] != GRADER_VERSION:
            failures.append(f"{task_id} total={reward['total']} grader_version={reward['grader_version']}")
    assert failures == []
