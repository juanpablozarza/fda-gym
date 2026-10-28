"""Generate resumable OpenRouter teacher episodes for distillation SFT."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

from fdagym.agents.baseline import load_split
from fdagym.agents.litellm_agent import LiteLLMAgent
from fdagym.openfda import DiskSnapshot
from fdagym.train.distill import teacher_trajectory_validator
from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET, system_prompt

DEFAULT_TEACHERS = (
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/anthropic/claude-haiku-5.5",
)
DEFAULT_COST_CAP_USD = 20.0


def prefixed_model(model: str) -> str:
    model = model.strip()
    if not model:
        raise ValueError("teacher model cannot be empty")
    return model if model.startswith("openrouter/") else f"openrouter/{model}"


def trajectory_messages(task: dict, trajectory: dict, *, tool_budget: int) -> list[dict]:
    """Convert scored FDAEnv steps into the tool-action messages the SFT codec reads."""
    messages = [
        {"role": "system", "content": system_prompt(DEFAULT_PROMPT_VERSION, tool_budget)},
        {"role": "user", "content": task["question"]},
    ]
    used = 0
    for step in trajectory["steps"]:
        action = {"tool": step["tool"], "arguments": step.get("arguments") or {}}
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        if step["tool"] == "submit_answer":
            continue
        used += 1
        observation = {
            "tool": step["tool"],
            "output": step.get("output"),
            "error": step.get("error"),
            "tool_calls_remaining": max(0, tool_budget - used),
        }
        messages.append(
            {
                "role": "tool",
                "name": step["tool"],
                "content": json.dumps(observation, ensure_ascii=False, default=str),
            }
        )
    return messages


def teacher_row(task: dict, trajectory: dict, *, teacher_model: str, attempt_index: int, tool_budget: int) -> dict:
    reward = trajectory["reward"]
    row = {
        "id": f"{teacher_model}/{task['id']}/a{attempt_index}",
        "question_id": task["id"],
        "template": task.get("type") or task.get("template") or "unknown",
        "variant": task.get("variant") or "",
        "teacher_model": teacher_model,
        "attempt_index": attempt_index,
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "tool_budget": tool_budget,
        "messages": trajectory_messages(task, trajectory, tool_budget=tool_budget),
        "reward": reward,
        "passed": reward["correctness"] > 0,
        "usage": trajectory.get("usage", {}),
    }
    teacher_trajectory_validator().validate(row)
    return row


def _completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            completed.add((row["teacher_model"], row["question_id"], int(row["attempt_index"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid existing trajectory row: {exc}") from exc
    return completed


def generate_teacher_trajectories(
    *,
    tasks: list[dict],
    snapshot,
    models: list[str],
    output: Path,
    episode_runner: Callable,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    attempts: int = 1,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    pause_s: float = 0.0,
) -> dict:
    """Append validated rows as episodes finish; reruns skip completed keys."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if not math.isfinite(cost_cap_usd) or cost_cap_usd <= 0:
        raise ValueError("cost_cap_usd must be > 0")
    if not math.isfinite(pause_s) or pause_s < 0:
        raise ValueError("pause_s must be >= 0")
    models = list(dict.fromkeys(prefixed_model(model) for model in models))
    if not models:
        raise ValueError("at least one teacher model is required")
    if isinstance(tool_budget, bool) or not isinstance(tool_budget, int) or tool_budget < 1:
        raise ValueError("tool_budget must be an integer >= 1")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(output)
    spent = 0.0
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                spent += float(json.loads(line).get("usage", {}).get("cost_usd") or 0.0)

    added = passed = skipped = 0
    aborted = None
    for model in models:
        agent = LiteLLMAgent(
            model,
            temperature=0.2,
            system_prompt=system_prompt(DEFAULT_PROMPT_VERSION, tool_budget),
        )
        # Share one env per model; reset() isolates each episode and usage is reset by run_episode.
        from fdagym.env import FDAEnv

        env = FDAEnv(snapshot, tasks, run_id="teacher-distill", tool_budget=tool_budget)
        for task in tasks:
            for attempt_index in range(attempts):
                key = (model, task["id"], attempt_index)
                if key in completed:
                    skipped += 1
                    continue
                if spent >= cost_cap_usd:
                    aborted = f"cost cap ${cost_cap_usd:.2f} reached (spent ${spent:.4f})"
                    break
                try:
                    trajectory = episode_runner(agent, env, task["id"])
                    row = teacher_row(
                        task,
                        trajectory,
                        teacher_model=model,
                        attempt_index=attempt_index,
                        tool_budget=tool_budget,
                    )
                except Exception as exc:  # noqa: BLE001 — continue other questions on a single episode failure
                    print(f"FAIL {model} {task['id']} a{attempt_index} {type(exc).__name__}: {exc}", flush=True)
                    continue
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                completed.add(key)
                added += 1
                passed += int(row["passed"])
                spent += float(row.get("usage", {}).get("cost_usd") or 0.0)
                print(
                    f"{model} {task['id']} a{attempt_index} "
                    f"R={row['reward']['total']:.3f} passed={row['passed']} spent=${spent:.4f}",
                    flush=True,
                )
                if pause_s:
                    time.sleep(pause_s)
            if aborted:
                break
        if aborted:
            break
    return {
        "output": str(output),
        "added": added,
        "passed": passed,
        "skipped": skipped,
        "spent_usd": round(spent, 6),
        "aborted": aborted,
    }


def run_from_files(
    *,
    split: str,
    tasks_path: Path | None,
    snapshot_version: str | None,
    models: list[str],
    output: Path,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    attempts: int = 1,
    limit: int | None = None,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    pause_s: float = 0.0,
) -> dict:
    tasks = load_split(split) if tasks_path is None else [
        json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("task input is empty")
    versions = {task["snapshot_version"] for task in tasks if task.get("snapshot_version")}
    if snapshot_version is None:
        if len(versions) != 1:
            raise ValueError(f"task input must have one snapshot_version, found {sorted(versions)}")
        snapshot_version = next(iter(versions))
    elif versions and versions != {snapshot_version}:
        raise ValueError(f"task snapshot versions {sorted(versions)} do not match {snapshot_version}")
    return generate_teacher_trajectories(
        tasks=tasks,
        snapshot=DiskSnapshot(snapshot_version),
        models=models,
        output=output,
        episode_runner=lambda agent, env, task_id: agent.run(env, task_id),
        tool_budget=tool_budget,
        attempts=attempts,
        cost_cap_usd=cost_cap_usd,
        pause_s=pause_s,
    )
