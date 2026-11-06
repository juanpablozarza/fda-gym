"""In-process FDA-Gym episode used by the HUD adapter and by tests without hud-sdk.

Wraps ``FDAEnv`` and grades only through ``fdagym.rewards.score``. Public-sheet
and snapshot resolution follow ``fdagym.demo`` (FG-9101–FG-9193, freeze 2026-10-15).
"""

from __future__ import annotations

import json
from pathlib import Path

from fdagym.demo import (
    SNAPSHOT_VERSION,
    DemoError,
    load_public_tasks,
    resolve_task,
    snapshot_for_demo,
)
from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.openfda.store import resolve_snapshot_root
from fdagym.rewards import score
from fdagym.snapshot import Snapshot
from fdagym.tools import TOOLS

FDA_TOOL_NAMES = tuple(spec["function"]["name"] for spec in TOOLS)
HUD_RUN_ID = "hud-public"


def default_public_path() -> Path:
    """Published sheet: checkout next to this package, else walk from cwd."""
    packaged = Path(__file__).resolve().parents[2] / "data" / "tasks" / "public.jsonl"
    if packaged.is_file():
        return packaged
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        path = candidate / "data" / "tasks" / "public.jsonl"
        if path.is_file():
            return path
    raise DemoError("published sheet not found: data/tasks/public.jsonl")


def public_task_ids(*, public_path: Path | None = None) -> list[str]:
    tasks = load_public_tasks(public_path or default_public_path())
    return sorted(tasks)


def coerce_submit_answer(answer) -> dict | None:
    """If *answer* looks like a ``submit_answer`` payload, return it; else None."""
    raw = answer
    if hasattr(answer, "content"):
        raw = answer.content
        if raw is None and hasattr(answer, "raw"):
            raw = answer.raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict) and "answer" in raw and "citations" in raw:
        return {"answer": raw["answer"], "citations": raw["citations"]}
    return None


class PublicEpisode:
    """One public.jsonl row on ``FDAEnv``. Tool dispatch is ``FDAEnv.step``."""

    def __init__(
        self,
        task_id: str,
        *,
        public_path: Path | None = None,
        snapshot_root: Path | None = None,
        snapshot: Snapshot | None = None,
    ):
        path = public_path or default_public_path()
        tasks = load_public_tasks(path)
        self.task = resolve_task(tasks, task_id)
        snap = snapshot
        if snap is None:
            root = snapshot_root if snapshot_root is not None else resolve_snapshot_root()
            snap = snapshot_for_demo(root)
        self.env = FDAEnv(snap, list(tasks.values()), run_id=HUD_RUN_ID, tool_budget=TOOL_BUDGET)
        self.reset_obs = self.env.reset(task_id)

    def prompt(self) -> str:
        names = ", ".join(FDA_TOOL_NAMES)
        return (
            f"You are solving FDA-Gym public task {self.task['id']} on the frozen "
            f"openFDA snapshot {SNAPSHOT_VERSION}.\n"
            f"Drive the fda-gym-tools MCP capability. Tools: {names}.\n"
            f"You may make at most {self.env.tool_budget} tool calls other than submit_answer.\n"
            "Finish with submit_answer: one answer (string, number, boolean, or NOT_FOUND) "
            "and up to 8 citations. Each citation needs endpoint, id, field, and a quote "
            "from a tool output you were shown.\n\n"
            f"Question: {self.task['question']}"
        )

    def call_tool(self, name: str, arguments: dict) -> dict:
        if self.env.terminated_by is not None:
            return {"error": "episode already finished", "tool": name}
        obs, _reward, done, info = self.env.step({"tool": name, "arguments": arguments})
        return {
            "observation": obs,
            "done": done,
            "tool_calls_used": info.get("tool_calls_used", self.env.tool_calls),
        }

    def grade(self, answer=None) -> dict:
        """Grade this episode with ``score()``. Reward is ``total`` (grader v5)."""
        if self.env.terminated_by is None:
            payload = coerce_submit_answer(answer)
            if payload is not None:
                self.env.step({"tool": "submit_answer", "arguments": payload})
        return score({"steps": self.env.steps, "final_answer": self.env.final_answer}, self.task)
