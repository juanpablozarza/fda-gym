"""v4 prompt, selectable v3, and a tool budget that the prompt and the env share."""

from pathlib import Path

import pytest

from fdagym import queries
from fdagym.tools import faers_modes
from fdagym.train.eval_tools import eval_tool_tasks
from fdagym.train.rollout import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_TOOL_BUDGET,
    TOOL_SYSTEM_PROMPT,
    QwenToolCodec,
    RolloutLimits,
    TurnGeneration,
    rollout_batch,
    system_prompt,
)
from fdagym.train.rollout_dry import CharTokenizer, OracleGenerator
from fdagym.train.sft_config import SFTConfig
from fdagym.train.tool_config import ToolTrainConfig
from tests.env.support import ALL_TASKS, JsonSnapshot

ROOT = Path(__file__).resolve().parents[2]


def test_v3_at_budget_6_is_the_historical_prompt():
    assert system_prompt("v3", 6) == TOOL_SYSTEM_PROMPT
    assert "query_adverse_events requires" not in TOOL_SYSTEM_PROMPT
    assert TOOL_SYSTEM_PROMPT.endswith("You get at most 6 tool calls, then you must submit_answer.")


def test_v4_adds_only_the_faers_line_and_rewrites_the_budget_number():
    historical = system_prompt("v3", 6)
    line = "- You get at most 6 tool calls, then you must submit_answer."
    prefix = historical[: -len(line)]
    v4_at_6 = system_prompt("v4", 6)
    added = v4_at_6[len(prefix) : -len(line)]
    assert prefix.endswith("\n")
    assert v4_at_6.startswith(prefix)
    assert added.startswith("- query_adverse_events requires")
    assert added.endswith("\n")
    assert v4_at_6.endswith(line)
    assert "- - " not in v4_at_6
    assert system_prompt("v3", 8) == historical.replace("at most 6 tool calls", "at most 8 tool calls", 1)
    rendered = system_prompt("v4", 8)
    lines = rendered.splitlines()
    assert lines[-1] == "- You get at most 8 tool calls, then you must submit_answer."
    assert lines[-2].startswith("- query_adverse_events requires")
    assert lines[-3].startswith("- NOT_FOUND")
    for mode in faers_modes():
        assert mode in added
    assert f"at most {queries.TOP_REACTIONS_LIMIT} MedDRA terms" in added
    assert f"from {min(queries.FAERS_YEARS)} through {max(queries.FAERS_YEARS)}" in added
    assert "meta.results.total" in added
    assert "leaves results empty" in added
    with pytest.raises(ValueError, match="prompt_version"):
        system_prompt("v5", 8)


def test_new_tool_configs_default_to_v4_and_budget_8_and_v3_stays_selectable():
    assert DEFAULT_TOOL_BUDGET == 8
    assert DEFAULT_PROMPT_VERSION == "v4"
    tool = ToolTrainConfig()
    assert tool.tool_budget == 8
    assert tool.prompt_version == "v4"
    assert tool.limits.tool_budget == 8
    assert tool.limits.prompt_version == "v4"
    assert tool.limits.max_turns == 9
    legacy = ToolTrainConfig(tool_budget=6, prompt_version="v3")
    assert legacy.limits.tool_budget == 6
    assert legacy.limits.prompt_version == "v3"
    assert legacy.limits.max_turns == 7
    assert system_prompt("v3", 6) in QwenToolCodec(prompt_version="v3", tool_budget=6).prompt("q")
    sft = SFTConfig()
    assert sft.tool_budget == 8
    assert sft.prompt_version == "v4"
    with pytest.raises(ValueError, match="tool_budget"):
        ToolTrainConfig(tool_budget=6, limits=RolloutLimits(temperature=0.65, prompt_version="v4"))
    with pytest.raises(ValueError, match="prompt_version"):
        ToolTrainConfig(prompt_version="v3", limits=RolloutLimits(temperature=0.65, tool_budget=8))


def test_rollout_prompt_names_the_budget_the_env_enforces():
    tokenizer = CharTokenizer()
    # Codec defaults are v4/8. The limits below must win, or the prompt would advertise 8.
    codec = QwenToolCodec()
    limits = RolloutLimits(tool_budget=2, prompt_version="v4", temperature=0.0)
    seen: dict[str, str] = {}

    class RepeatSearch:
        def generate(self, prompt_token_ids, *, max_tokens, temperature):
            seen["prompt"] = tokenizer.decode(prompt_token_ids[0])
            text = codec.action_text({"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}})
            ids = tuple(tokenizer.encode(text))
            return [TurnGeneration(ids, tuple(-0.1 for _ in ids)) for _ in prompt_token_ids]

    rollouts = rollout_batch(
        JsonSnapshot(),
        ALL_TASKS,
        ["FG-0001"],
        tokenizer,
        RepeatSearch(),
        codec,
        limits,
        run_id="budget",
        seed=0,
    )
    rollout = rollouts[0]
    system = seen["prompt"].split("# Tools", 1)[0]
    assert system_prompt("v4", 2) in system
    assert "at most 2 tool calls" in system
    assert "at most 8 tool calls" not in system
    assert rollout.terminated_by == "max_steps"
    assert rollout.trajectory["steps"][-1]["error"] == "tool budget exhausted"
    assert sum(1 for step in rollout.trajectory["steps"] if step.get("error") is None) == 2


def test_eval_summary_records_the_prompt_and_the_budget(tmp_path):
    snapshot = JsonSnapshot()
    tasks = [task for task in ALL_TASKS if task["id"] in {"FG-0001", "FG-0002"}]
    limits = RolloutLimits(temperature=0.0, tool_budget=6, prompt_version="v3")
    codec = QwenToolCodec(prompt_version="v3", tool_budget=6)
    tokenizer = CharTokenizer()
    summary = eval_tool_tasks(
        snapshot=snapshot,
        tasks=tasks,
        tokenizer=tokenizer,
        generator=OracleGenerator(tokenizer, codec, snapshot, tasks),
        model={"name": "oracle-dry", "kind": "trained", "checkpoint_step": None},
        run_id="prompt-budget",
        results_dir=tmp_path,
        codec=codec,
        limits=limits,
        split="fixture",
    )
    assert summary["prompt_version"] == "v3"
    assert summary["tool_budget"] == 6
    assert summary["models"]["oracle-dry"]["mean_reward"] == 1.0


def test_entrypoints_publish_the_same_defaults():
    modal = (ROOT / "modal_train.py").read_text()
    assert modal.count(f"tool_budget: int = {DEFAULT_TOOL_BUDGET}") >= 4
    assert modal.count(f'prompt_version: str = "{DEFAULT_PROMPT_VERSION}"') >= 4
    for relative in ("scripts/eval_tools.py", "scripts/grpo_tools.py"):
        text = (ROOT / relative).read_text()
        assert "--tool-budget" in text
        assert "--prompt-version" in text
        assert "DEFAULT_TOOL_BUDGET" in text
        assert "DEFAULT_PROMPT_VERSION" in text
    checker = (ROOT / "scripts" / "audit_hard.py").read_text()
    assert "--budget" in checker
    assert "DEFAULT_TOOL_BUDGET" in checker
    assert "tool_budget=budget" in checker
