"""verifiers v1 Taskset for public FDA-Gym. Importing this module needs the prime extra."""

from __future__ import annotations

from typing import Any

import verifiers.v1 as vf

from fdagym.prime.episode import (
    ADAPTER_SYSTEM_PROMPT,
    grade_steps,
    load_public_sheet,
    public_snapshot,
)
from fdagym.prime.servers.tools import FDAGymState, FDAGymToolset, FDAGymToolsetConfig
from fdagym.rewards.combine import WEIGHTS


class FDAGymTaskData(vf.TaskData):
    sheet_row: dict[str, Any]


class FDAGymTaskConfig(vf.TaskConfig):
    tools: FDAGymToolsetConfig = FDAGymToolsetConfig()


class FDAGymTask(vf.Task[FDAGymTaskData, FDAGymState, FDAGymTaskConfig]):
    @classmethod
    def toolsets(cls, config: FDAGymTaskConfig) -> list[vf.Toolset]:
        return [FDAGymToolset(config.tools)]

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return bool(getattr(trace.state, "done", False))

    def _parts(self, trace: vf.Trace) -> dict[str, Any]:
        return _grade(self.data.sheet_row, trace.state)

    # One returned dict would take a single weight. Each term has its own.
    @vf.reward(weight=WEIGHTS["correctness"])
    async def correctness(self, trace: vf.Trace) -> float:
        return float(self._parts(trace)["correctness"])

    @vf.reward(weight=WEIGHTS["citation"])
    async def citation(self, trace: vf.Trace) -> float:
        return float(self._parts(trace)["citation"])

    @vf.reward(weight=WEIGHTS["format"])
    async def format(self, trace: vf.Trace) -> float:
        return float(self._parts(trace)["format"])

    @vf.reward(weight=WEIGHTS["efficiency"])
    async def efficiency(self, trace: vf.Trace) -> float:
        return float(self._parts(trace)["efficiency"])

    async def finalize(self, trace: vf.Trace, runtime) -> None:
        reward = _grade(self.data.sheet_row, trace.state)
        trace.info["fdagym"] = {
            "correctness": reward["correctness"],
            "citation": reward["citation"],
            "format": reward["format"],
            "efficiency": reward["efficiency"],
            "total": reward["total"],
            "grader_version": reward["grader_version"],
        }


class FDAGymConfig(vf.TasksetConfig):
    task: FDAGymTaskConfig = FDAGymTaskConfig()
    snapshot_dir: str | None = None
    public_path: str | None = None


class FDAGymTaskset(vf.Taskset[FDAGymTask, FDAGymConfig]):
    def load(self) -> list[FDAGymTask]:
        public_snapshot(snapshot_root=self.config.snapshot_dir)
        rows = load_public_sheet(self.config.public_path)
        tools = self.config.task.tools
        if tools.snapshot_dir is None:
            tools = tools.model_copy(update={"snapshot_dir": self.config.snapshot_dir})
        if tools.public_path is None:
            tools = tools.model_copy(update={"public_path": self.config.public_path})
        task_config = self.config.task.model_copy(update={"tools": tools})
        return [
            FDAGymTask(
                FDAGymTaskData(
                    id=task_id,
                    name=task_id,
                    prompt=row["question"],
                    system_prompt=ADAPTER_SYSTEM_PROMPT,
                    sheet_row=row,
                ),
                task_config,
            )
            for task_id, row in rows.items()
        ]


def _grade(sheet_row: dict, state: Any) -> dict:
    steps = list(getattr(state, "steps", None) or [])
    final_answer = getattr(state, "final_answer", None)
    return grade_steps(sheet_row, steps, final_answer)


__all__ = ["FDAGymTaskset"]
