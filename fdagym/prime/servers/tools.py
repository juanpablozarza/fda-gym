"""Per-rollout MCP Toolset that forwards the five FDA-Gym tools into FDAEnv."""

from __future__ import annotations

from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from fdagym.prime.episode import (
    drop_none,
    dump_observation,
    make_public_env,
    step_tool,
)


class FDAGymState(vf.State):
    done: bool = False
    steps: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: dict | None = None


class FDAGymToolsetConfig(vf.ToolsetConfig):
    snapshot_dir: str | None = None
    public_path: str | None = None


class FDAGymToolset(vf.Toolset[FDAGymToolsetConfig, FDAGymState]):
    TOOL_PREFIX = None

    def __init__(self, config: FDAGymToolsetConfig) -> None:
        super().__init__(config)
        self._env = None

    async def setup_task(self, task) -> None:
        sheet = _sheet_row(task)
        self._env, _row = make_public_env(
            sheet["id"],
            public_path=self.config.public_path,
            snapshot_root=self.config.snapshot_dir,
        )

    def _call(self, tool: str, arguments: dict) -> str:
        env = self._env
        if env is None:
            raise RuntimeError("FDAEnv was not set up for this rollout")
        try:
            obs, _reward, done, _info = step_tool(env, tool, arguments)
        except RuntimeError as exc:
            return dump_observation({"output": None, "error": str(exc)}, done=True)
        self.state.steps = list(env.steps)
        self.state.final_answer = env.final_answer
        self.state.done = done
        return dump_observation(obs, done=done)

    @vf.tool
    def search_drug_label(
        self,
        generic_name: str,
        has_section: str | None = None,
        page: int | None = None,
    ) -> str:
        """List drug-label summaries for one generic name (set_id, brand, manufacturer, effective_time, sections). At most 10 rows per page. has_section keeps labels that contain that section. Section text is not included; call get_label_section for it."""
        return self._call(
            "search_drug_label",
            drop_none({"generic_name": generic_name, "has_section": has_section, "page": page}),
        )

    @vf.tool
    def get_label_section(self, set_id: str, section: str) -> str:
        """Return one section of a label whose set_id was already returned by search_drug_label in this episode. Text longer than 1500 characters is cut, and the output says so."""
        return self._call("get_label_section", {"set_id": set_id, "section": section})

    @vf.tool
    def query_adverse_events(
        self,
        generic_name: str,
        mode: str,
        serious: bool | None = None,
        year: int | None = None,
    ) -> str:
        """FAERS reports for one generic name. mode top_reactions returns the top MedDRA terms; mode count returns how many reports match. serious may only be true (omit it for all reports)."""
        return self._call(
            "query_adverse_events",
            drop_none({"generic_name": generic_name, "mode": mode, "serious": serious, "year": year}),
        )

    @vf.tool
    def search_approvals(
        self,
        generic_name: str | None = None,
        application_number: str | None = None,
        sponsor_name: str | None = None,
        page: int | None = None,
    ) -> str:
        """Drugs@FDA applications. Pass exactly one of generic_name, application_number, or sponsor_name. page starts at 1."""
        return self._call(
            "search_approvals",
            drop_none(
                {
                    "generic_name": generic_name,
                    "application_number": application_number,
                    "sponsor_name": sponsor_name,
                    "page": page,
                }
            ),
        )

    @vf.tool
    def submit_answer(
        self,
        answer: str | float | bool,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        """Finish the episode. answer is one string, number, or boolean, or the string NOT_FOUND. citations quote a field you were shown. At most 8 citations."""
        return self._call("submit_answer", {"answer": answer, "citations": citations or []})


def _sheet_row(task: Any) -> dict:
    if isinstance(task, dict) and "id" in task:
        return task
    sheet = getattr(task, "sheet_row", None)
    if isinstance(sheet, dict):
        return sheet
    data = getattr(task, "data", None)
    sheet = getattr(data, "sheet_row", None)
    if isinstance(sheet, dict):
        return sheet
    raise TypeError("verifiers task is missing the public sheet row")


if __name__ == "__main__":
    FDAGymToolset.run()
