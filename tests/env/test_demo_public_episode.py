"""The public demo is keyless, stays on the published sheet, and prints score keys."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fdagym.demo import (
    DEFAULT_TASK_ID,
    SCORE_KEYS,
    DemoError,
    is_public_task_id,
    play_public_task,
    resolve_task,
)
from fdagym.env import TOOL_BUDGET

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
