"""Public-sheet FDAEnv helpers for the verifiers v1 wrapper.

This module does not import ``verifiers``. Scripted tests and the Toolset both
step through ``FDAEnv`` and grade with ``fdagym.rewards.score``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fdagym.demo import (
    DEFAULT_TASK_ID,
    SNAPSHOT_VERSION,
    DemoError,
    gold_submit,
    load_public_tasks,
    resolve_task,
    snapshot_for_demo,
)
from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.openfda.store import (
    discover_snapshot_root,
    packaged_hub_root,
    packaged_hub_snapshot_root,
    resolve_snapshot_root,
)
from fdagym.rewards import score
from fdagym.tools import TOOLS

ADAPTER_TOOLS = tuple(item["function"]["name"] for item in TOOLS)
ADAPTER_SYSTEM_PROMPT = """\
You are an FDA-Gym agent. Answer only from tool outputs returned in THIS episode.
Do not use memorized FDA facts.
Tools: search_drug_label, get_label_section, query_adverse_events, search_approvals, submit_answer.
Call submit_answer when you are done. Search before submitting NOT_FOUND.
You get at most 8 tool calls besides submit_answer.
"""
PUBLIC_ENV = "FDAGYM_PUBLIC_JSONL"
PUBLIC_SHEET = Path("data") / "tasks" / "public.jsonl"


def packaged_hub_public_path() -> Path | None:
    """``public.jsonl`` shipped in the ``fda-gym`` Hub wheel, else an editable checkout."""
    pkg = packaged_hub_root()
    if pkg is None:
        return None
    bundled = pkg / "data" / "tasks" / "public.jsonl"
    if bundled.is_file():
        return bundled
    snap = packaged_hub_snapshot_root()
    if snap is None:
        return None
    path = snap.parent / "tasks" / "public.jsonl"
    return path if path.is_file() else None


def discover_public_path() -> Path | None:
    """``FDAGYM_PUBLIC_JSONL``, else ``data/tasks/public.jsonl`` from cwd, snapshot, or Hub wheel."""
    env = os.environ.get(PUBLIC_ENV)
    if env:
        return Path(env)
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        path = candidate / PUBLIC_SHEET
        if path.is_file():
            return path
    root = discover_snapshot_root()
    if root is not None:
        path = root.parent / "tasks" / "public.jsonl"
        if path.is_file():
            return path
    return packaged_hub_public_path()


def resolve_public_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    found = discover_public_path()
    if found is None:
        raise DemoError(f"published sheet not found: {PUBLIC_SHEET}")
    return found


def public_snapshot(*, snapshot_root: Path | str | None = None):
    """Committed ``2026-10-15`` DiskSnapshot. Same root rules as ``fdagym.demo``."""
    root = resolve_snapshot_root(snapshot_root)
    return snapshot_for_demo(root)


def make_public_env(
    task_id: str = DEFAULT_TASK_ID,
    *,
    public_path: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    run_id: str = "prime-public",
) -> tuple[FDAEnv, dict]:
    """Build an ``FDAEnv`` for one published id (FG-9101–FG-9193) and ``reset`` it."""
    public = resolve_public_path(public_path)
    tasks = load_public_tasks(public)
    task = resolve_task(tasks, task_id)
    snap = public_snapshot(snapshot_root=snapshot_root)
    env = FDAEnv(snap, [task], run_id=run_id, tool_budget=TOOL_BUDGET)
    env.reset(task["id"])
    return env, task


def step_tool(env: FDAEnv, tool: str, arguments: dict) -> tuple[dict, float, bool, dict]:
    """One hop through ``FDAEnv.step``. The Toolset and scripted tests share this."""
    return env.step({"tool": tool, "arguments": arguments})


def grade_episode(task: dict, env: FDAEnv) -> dict:
    """Grade the adapter episode with ``score()`` (GRADER_VERSION 5)."""
    return score({"steps": env.steps, "final_answer": env.final_answer}, task)


def grade_steps(task: dict, steps: list[dict], final_answer: dict | None) -> dict:
    return score({"steps": steps, "final_answer": final_answer}, task)


def play_public_adapter_episode(
    task_id: str = DEFAULT_TASK_ID,
    *,
    public_path: Path | str | None = None,
    snapshot_root: Path | str | None = None,
) -> dict:
    """Honest-oracle replay through the adapter: ``oracle_calls`` then gold submit."""
    env, task = make_public_env(task_id, public_path=public_path, snapshot_root=snapshot_root)
    for call in task["oracle_calls"]:
        _obs, _reward, done, _info = step_tool(env, call["tool"], dict(call["arguments"]))
        if done:
            raise RuntimeError(f"{task['id']} oracle ended the episode on {call['tool']}")
    _obs, _reward, done, _info = step_tool(env, "submit_answer", gold_submit(task, env.steps))
    if not done:
        raise RuntimeError(f"{task['id']} submit_answer did not end the episode")
    return grade_episode(task, env)


def dump_observation(obs: dict, *, done: bool) -> str:
    payload = {"output": obs.get("output"), "error": obs.get("error"), "done": done}
    return json.dumps(payload, ensure_ascii=False, default=str)


def drop_none(arguments: dict) -> dict:
    return {key: value for key, value in arguments.items() if value is not None}


def load_public_sheet(public_path: Path | str | None = None) -> dict[str, dict]:
    return load_public_tasks(resolve_public_path(public_path))


__all__ = [
    "ADAPTER_SYSTEM_PROMPT",
    "ADAPTER_TOOLS",
    "DEFAULT_TASK_ID",
    "PUBLIC_ENV",
    "PUBLIC_SHEET",
    "SNAPSHOT_VERSION",
    "TOOL_BUDGET",
    "DemoError",
    "discover_public_path",
    "drop_none",
    "dump_observation",
    "grade_episode",
    "grade_steps",
    "load_public_sheet",
    "make_public_env",
    "packaged_hub_public_path",
    "play_public_adapter_episode",
    "public_snapshot",
    "resolve_public_path",
    "step_tool",
]
