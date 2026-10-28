"""Gym-style episode over a frozen snapshot. Tool calls are capped, then submit_answer is still allowed."""

import json
from datetime import UTC, datetime

from fdagym.rewards import score
from fdagym.rewards.format import answer_validator
from fdagym.snapshot import Snapshot
from fdagym.tools import TOOLS, EpisodeState, execute

# Baseline and any caller that does not pass ``tool_budget``. Tool-mode runs pass 8.
TOOL_BUDGET = 6


class FDAEnv:
    def __init__(
        self,
        snapshot: Snapshot,
        tasks: list[dict],
        *,
        run_id: str = "local",
        seed: int = 0,
        model=None,
        judge=None,
        tool_budget: int | None = None,
    ):
        self.snapshot = snapshot
        self.tasks = {task["id"]: task for task in tasks}
        self.run_id = run_id
        self.seed = seed
        self.model = model or {"name": "unspecified", "kind": "baseline", "checkpoint_step": None}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None, "latency_ms": 0}
        self.judge = judge
        budget = TOOL_BUDGET if tool_budget is None else tool_budget
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("tool_budget must be an integer >= 1")
        self.tool_budget = budget
        self._ready = False

    def reset(self, task_id: str) -> dict:
        try:
            self.task = self.tasks[task_id]
        except KeyError:
            raise KeyError(task_id) from None
        self.state = EpisodeState(self.snapshot)
        self.steps: list[dict] = []
        self.tool_calls = 0
        self.final_answer = None
        self.terminated_by: str | None = None
        self._reward: dict | None = None
        self._done = False
        self.started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._ready = True
        return {"question": self.task["question"], "tools": TOOLS}

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode already finished")
        parsed, error = _parse_action(action)
        if parsed is None:
            return self._end("error", tool="submit_answer", arguments={}, output=None, error=error, thought=None)
        name = parsed["tool"]
        arguments = parsed["arguments"]
        thought = parsed["thought"]
        if name == "submit_answer":
            return self._submit(arguments, thought)
        if self.tool_calls >= self.tool_budget:
            return self._end(
                "max_steps",
                tool=name,
                arguments=arguments,
                output={"error": "tool budget exhausted; call submit_answer"},
                error="tool budget exhausted",
                thought=thought,
            )
        self.tool_calls += 1
        index = len(self.steps) + 1
        output, err = execute(self.state, name, arguments, step_index=index)
        self.steps.append(_step(index, name, arguments, output, err, thought))
        return self._obs(name, output, err), 0.0, False, {"tool_calls_used": self.tool_calls}

    def _submit(self, arguments: dict, thought: str | None):
        index = len(self.steps) + 1
        if answer_validator.is_valid(arguments):
            self.final_answer = arguments
            self.steps.append(_step(index, "submit_answer", arguments, None, None, thought))
            return self._end_done("submit")
        self.steps.append(
            _step(index, "submit_answer", arguments, None, "submit_answer does not match the schema", thought)
        )
        return self._end_done("error")

    def _end(self, terminated_by: str, **step_fields):
        index = len(self.steps) + 1
        self.steps.append(_step(index, **step_fields))
        return self._end_done(terminated_by)

    def _end_done(self, terminated_by: str):
        self._reward = score({"steps": self.steps, "final_answer": self.final_answer}, self.task, judge=self.judge)
        self.terminated_by = terminated_by
        self._done = True
        last = self.steps[-1]
        info = {"reward": self._reward, "trajectory": self.trajectory, "tool_calls_used": self.tool_calls}
        return self._obs(last["tool"], last["output"], last["error"]), self._reward["total"], True, info

    @property
    def trajectory(self) -> dict:
        if not self._ready or self.terminated_by is None or self._reward is None:
            raise RuntimeError("trajectory is available after the episode ends")
        return {
            "id": f"{self.run_id}/{self.task['id']}/s{self.seed}",
            "run_id": self.run_id,
            "task_id": self.task["id"],
            "model": self.model,
            "seed": self.seed,
            "snapshot_version": self.snapshot.version,
            "started_at": self.started_at,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "terminated_by": self.terminated_by,
            "reward": self._reward,
            "usage": self.usage,
        }

    def _obs(self, tool: str, output, error) -> dict:
        return {
            "tool": tool,
            "output": output,
            "error": error,
            "tool_calls_used": self.tool_calls,
            "tool_calls_remaining": max(0, self.tool_budget - self.tool_calls),
        }


def _parse_action(action) -> tuple[dict | None, str | None]:
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
    if not isinstance(action, dict):
        return None, "action must be a JSON object"
    tool = action.get("tool")
    if not isinstance(tool, str) or not tool:
        return None, "action.tool must be a string"
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        return None, "action.arguments must be an object"
    thought = action.get("thought")
    if not isinstance(thought, str):
        thought = None
    return {"tool": tool, "arguments": arguments, "thought": thought}, None


def _step(index: int, tool: str, arguments: dict, output, error, thought: str | None) -> dict:
    return {
        "index": index,
        "thought": thought,
        "tool": tool,
        "arguments": arguments,
        "output": output,
        "error": error,
        "latency_ms": 0,
    }
