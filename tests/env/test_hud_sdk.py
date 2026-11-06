"""HUD SDK wiring. Skips unless the optional [hud] extra is installed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("hud")
pytest.importorskip("fastmcp")

from hud.environment import Environment

from fdagym.env import TOOL_BUDGET
from fdagym.hud.sdk import (
    ENV_NAME,
    ENV_VERSION,
    MCP_CAPABILITY_NAME,
    bind_episode,
    build_mcp_server,
    load_hud_environment,
    mcp_tool_names,
    mint_public_tasks,
    unbind_episode,
)
from fdagym.hud.session import FDA_TOOL_NAMES, PublicEpisode

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT = ROOT / "data" / "snapshots"


def test_hud_environment_registers_public_template():
    env, public_task = load_hud_environment()
    assert isinstance(env, Environment)
    assert env.name == ENV_NAME
    assert env.version == ENV_VERSION
    assert "public" in env.tasks
    task = public_task(task_id="FG-9101")
    assert task.env == ENV_NAME
    assert task.id == "public"
    assert task.args == {"task_id": "FG-9101"}


def test_mint_one_hud_task_per_public_row():
    tasks = mint_public_tasks(public_path=PUBLIC)
    assert len(tasks) == 93
    assert tasks[0].slug == "FG-9101"
    assert tasks[-1].slug == "FG-9193"
    assert {task.id for task in tasks} == {"public"}
    assert all(task.agent_config == {"max_steps": TOOL_BUDGET + 4} for task in tasks)


def test_mcp_server_advertises_the_five_fda_tools():
    names = asyncio.run(mcp_tool_names())
    assert names == list(FDA_TOOL_NAMES)


def test_in_process_mcp_tools_wrap_fdaenv_not_http():
    episode = PublicEpisode("FG-9101", public_path=PUBLIC, snapshot_root=SNAPSHOT)
    previous = bind_episode(episode)
    try:
        server = build_mcp_server()
        result = asyncio.run(
            server.call_tool(
                "query_adverse_events",
                {"generic_name": "abacavir", "mode": "top_reactions"},
            )
        )
        assert result.is_error is False
        assert episode.env.steps[0]["tool"] == "query_adverse_events"
        assert episode.env.steps[0]["arguments"]["generic_name"] == "abacavir"
    finally:
        unbind_episode(previous)
    assert MCP_CAPABILITY_NAME == "fda-gym-tools"
