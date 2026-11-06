"""Keyless public-sheet episode demo. No LLM client, no live openFDA.

Replay one published task's ``oracle_calls`` through ``FDAEnv`` on the committed
``2026-10-15`` DiskSnapshot, then ``submit_answer`` with gold plus citations
taken from this episode's tool output. After every hop, ``score()`` is printed
honestly — partial totals before submit are expected to be 0.

Default task ``FG-9101`` is a 3-call FAERS compare (abacavir / pravastatin /
aripiprazole, most reports of FATIGUE). It is short, uses only
``query_adverse_events``, and is the same row the README uses for the one-call
guess that scores correctness 0. The honest three-drug replay should total 1.0.

This module must not import ``fdagym.train``, ``fdagym.agents``, litellm, or
any chat-completion client. Gold submit lives in ``fdagym.gold_cite`` so the
demo never loads the trainer path (see ``scripts/tool_rollout_dry.py``).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.gold_cite import gold_submit
from fdagym.openfda import DiskSnapshot
from fdagym.rewards import score
from fdagym.snapshot import Snapshot

DEFAULT_TASK_ID = "FG-9101"
PUBLIC_ID_FIRST = 9101
PUBLIC_ID_LAST = 9193
SNAPSHOT_VERSION = "2026-10-15"
OBS_LIMIT = 600
HONEST_TOTAL_MIN = 0.99
SCORE_KEYS = ("correctness", "citation", "format", "efficiency", "total", "grader_version")
_PUBLIC_ID = re.compile(r"^FG-(\d{4})$")
_LLM_MODULES = ("litellm", "openai", "anthropic")
_LLM_PREFIXES = ("fdagym.agents.litellm", "fdagym.agents.chat")


class DemoError(ValueError):
    """User-facing demo failure (bad id, missing sheet, missing snapshot)."""


def is_public_task_id(task_id: str) -> bool:
    match = _PUBLIC_ID.fullmatch(task_id)
    if match is None:
        return False
    return PUBLIC_ID_FIRST <= int(match.group(1)) <= PUBLIC_ID_LAST


def loaded_llm_clients() -> list[str]:
    found = [name for name in _LLM_MODULES if name in sys.modules]
    for name in sys.modules:
        if name.startswith(_LLM_PREFIXES):
            found.append(name)
    return found


def load_public_tasks(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise DemoError(f"published sheet not found: {path}")
    tasks: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("id")
        if not isinstance(task_id, str) or not is_public_task_id(task_id):
            raise DemoError(f"{path} contains a row outside FG-9101–FG-9193: {task_id!r}")
        if row.get("split") != "public":
            raise DemoError(f"{task_id} is not split=public")
        tasks[task_id] = row
    if not tasks:
        raise DemoError(f"{path} has no public tasks")
    return tasks


def resolve_task(tasks: dict[str, dict], task_id: str) -> dict:
    if not is_public_task_id(task_id) or task_id not in tasks:
        raise DemoError(f"{task_id} is not on the published public sheet (FG-9101–FG-9193)")
    return tasks[task_id]


def format_score(reward: dict) -> str:
    parts = []
    for key in SCORE_KEYS:
        value = reward[key]
        if key == "grader_version":
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={float(value):.4f}")
    return " ".join(parts)


def truncate_observation(value, *, limit: int = OBS_LIMIT) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def snapshot_for_demo(root: Path) -> DiskSnapshot:
    if not (root / SNAPSHOT_VERSION / "manifest.jsonl").is_file():
        raise DemoError(f"committed snapshot missing: {root / SNAPSHOT_VERSION}")
    return DiskSnapshot(SNAPSHOT_VERSION, root=root)


def run_episode(
    env: FDAEnv,
    task: dict,
    *,
    write: Callable[[str], None],
) -> dict:
    """Replay ``oracle_calls`` then gold ``submit_answer``. Print after every hop."""
    reset = env.reset(task["id"])
    write(f"task {task['id']}  type={task['type']}  snapshot={env.snapshot.version}  tool_budget={env.tool_budget}")
    write(f"question: {reset['question']}")
    write("")
    for index, call in enumerate(task["oracle_calls"], start=1):
        action = {"tool": call["tool"], "arguments": dict(call["arguments"])}
        obs, _reward, done, _info = env.step(action)
        if done:
            raise RuntimeError(f"{task['id']} oracle ended the episode on {call['tool']}")
        _print_hop(write, heading=f"hop {index}", action=action, obs=obs, task=task, env=env)
    submit = gold_submit(task, env.steps)
    action = {"tool": "submit_answer", "arguments": submit}
    obs, _reward, done, info = env.step(action)
    if not done:
        raise RuntimeError(f"{task['id']} submit_answer did not end the episode")
    _print_hop(write, heading="submit", action=action, obs=obs, task=task, env=env, reward=info["reward"])
    return info["reward"]


def play_public_task(
    task_id: str,
    *,
    public_path: Path,
    snapshot_root: Path,
    write: Callable[[str], None],
    snapshot: Snapshot | None = None,
) -> dict:
    tasks = load_public_tasks(public_path)
    task = resolve_task(tasks, task_id)
    snap = snapshot if snapshot is not None else snapshot_for_demo(snapshot_root)
    env = FDAEnv(snap, list(tasks.values()), run_id="public-demo", tool_budget=TOOL_BUDGET)
    return run_episode(env, task, write=write)


def _print_hop(
    write: Callable[[str], None],
    *,
    heading: str,
    action: dict,
    obs: dict,
    task: dict,
    env: FDAEnv,
    reward: dict | None = None,
) -> None:
    write(f"--- {heading} ---")
    write(f"tool: {action['tool']}")
    write(f"args: {truncate_observation(action['arguments'])}")
    observation = {"output": obs.get("output"), "error": obs.get("error")}
    write(f"observation: {truncate_observation(observation)}")
    breakdown = reward if reward is not None else score({"steps": env.steps, "final_answer": env.final_answer}, task)
    write(f"score  {format_score(breakdown)}")
    write("")
