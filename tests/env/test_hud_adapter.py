"""HUD adapter in-process path: FDAEnv + score(), no hud-sdk required."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fdagym.demo import DEFAULT_TASK_ID, DemoError, gold_submit
from fdagym.env import TOOL_BUDGET
from fdagym.hud.session import FDA_TOOL_NAMES, PublicEpisode, public_task_ids
from fdagym.rewards.version import GRADER_VERSION
from fdagym.tools import TOOLS

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT = ROOT / "data" / "snapshots"


def test_adapter_tool_names_match_fdagym_tools():
    assert FDA_TOOL_NAMES == tuple(spec["function"]["name"] for spec in TOOLS)
    assert FDA_TOOL_NAMES == (
        "search_drug_label",
        "get_label_section",
        "query_adverse_events",
        "search_approvals",
        "submit_answer",
    )


def test_public_task_ids_are_the_published_sheet():
    ids = public_task_ids(public_path=PUBLIC)
    assert ids[0] == "FG-9101"
    assert ids[-1] == "FG-9193"
    assert len(ids) == 93


def test_adapter_rejects_ids_outside_the_public_sheet():
    with pytest.raises(DemoError, match="published public sheet"):
        PublicEpisode("FG-0001", public_path=PUBLIC, snapshot_root=SNAPSHOT)
    with pytest.raises(DemoError, match="published public sheet"):
        PublicEpisode("FG-9194", public_path=PUBLIC, snapshot_root=SNAPSHOT)


def test_fg9101_through_adapter_scores_one_via_score(monkeypatch):
    scored = []
    from fdagym.hud import session as hud_session

    real_score = hud_session.score

    def wrapped(trajectory, task, **kwargs):
        scored.append(trajectory)
        return real_score(trajectory, task, **kwargs)

    monkeypatch.setattr(hud_session, "score", wrapped)

    episode = PublicEpisode(DEFAULT_TASK_ID, public_path=PUBLIC, snapshot_root=SNAPSHOT)
    assert episode.env.tool_budget == TOOL_BUDGET
    hops = []
    for call in episode.task["oracle_calls"]:
        result = episode.call_tool(call["tool"], dict(call["arguments"]))
        hops.append(call["tool"])
        assert result["done"] is False
        assert result["observation"]["error"] is None
    assert hops == ["query_adverse_events", "query_adverse_events", "query_adverse_events"]
    drugs = [step["arguments"]["generic_name"] for step in episode.env.steps]
    assert drugs == ["abacavir", "pravastatin", "aripiprazole"]
    submit = gold_submit(episode.task, episode.env.steps)
    result = episode.call_tool("submit_answer", submit)
    assert result["done"] is True
    reward = episode.grade()
    assert scored, "grade() must call fdagym.rewards.score"
    assert reward["grader_version"] == GRADER_VERSION
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0
    assert reward["citation"] == 1.0
    assert reward["format"] == 1.0
    assert reward["efficiency"] == 1.0


def test_load_hud_environment_requires_the_extra():
    if importlib.util.find_spec("hud") is not None:
        pytest.skip("hud extra is installed")
    from fdagym.hud.sdk import load_hud_environment

    with pytest.raises(ImportError, match="extra"):
        load_hud_environment()


def test_importing_hud_adapter_does_not_load_hud_or_fastmcp():
    code = (
        "import sys; "
        "before = set(sys.modules); "
        "import fdagym.hud, fdagym.hud.session, fdagym.hud.sdk; "
        "added = set(sys.modules) - before; "
        "blocked = [n for n in added if n == 'hud' or n.startswith('hud.') "
        "or n == 'fastmcp' or n.startswith('fastmcp.')]; "
        "print(blocked); raise SystemExit(1 if blocked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "HUD_API_KEY",
            }
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
