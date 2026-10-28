"""Tool-call GRPO dry run. Does not start Modal or a GPU.

    uv run python scripts/grpo_tools.py --dry-run

The GPU job is ``scripts/run_gpu.py train-tools`` (or Modal ``train_tools``).
Both call ``fdagym.train.jobs.run_train_tools``. This script only checks the
CPU fixtures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.grpo_tools import build_tool_rows
from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET, PROMPT_VERSIONS, QwenToolCodec
from fdagym.train.rollout_dry import format_report, run_rollout_dry
from fdagym.train.tool_config import ToolTrainConfig
from fdagym.train.tool_score import training_reward

DEFAULT_TASKS = ("FG-0001", "FG-0002")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not args.dry_run:
        print(
            "Pass --dry-run for the CPU check. GPU training is `python scripts/run_gpu.py train-tools`. "
            "This script does not start Modal.",
            file=sys.stderr,
        )
        return 2
    return _dry(args)


def _tool_config(args) -> ToolTrainConfig:
    return ToolTrainConfig(tool_budget=args.tool_budget, prompt_version=args.prompt_version)


def _codec(tool: ToolTrainConfig) -> QwenToolCodec:
    return QwenToolCodec(prompt_version=tool.prompt_version, tool_budget=tool.tool_budget)


def _dry(args) -> int:
    from tests.env.support import ALL_TASKS, JsonSnapshot

    tool = _tool_config(args)
    cfg = tool.train
    codec = _codec(tool)
    print(
        "num_generations="
        f"{cfg.num_generations} gradient_checkpointing={cfg.gradient_checkpointing} "
        f"context={cfg.max_prompt_length + cfg.max_completion_length} "
        f"vllm_gpu_memory_utilization={cfg.vllm_gpu_memory_utilization} "
        f"tool_parse_bonus={tool.tool_parse_bonus} dense_reward={cfg.dense_reward} "
        f"force_submit={tool.limits.force_submit} "
        f"max_turns={tool.limits.max_turns} tool_budget={tool.tool_budget} "
        f"prompt_version={tool.prompt_version} max_steps={cfg.max_steps} temperature={cfg.temperature} "
        f"logprob_chunk_size={cfg.logprob_chunk_size} "
        f"type_mix={tool.type_mix} parse_stop={tool.parse_stop_min}@{tool.parse_stop_steps}"
    )
    rows, mix = build_tool_rows(ALL_TASKS, cfg, codec)
    print(f"fixture_rows={[row['task_id'] for row in rows]} mix={mix}")
    from fdagym.agents.baseline import load_split

    train_rows, train_mix = build_tool_rows(
        load_split("train"), cfg, codec, type_mix=tool.type_mix, mix_rows=tool.mix_rows, mix_seed=tool.mix_seed
    )
    print(f"train_type_mix rows={len(train_rows)} by_type={train_mix['rows']} unique={train_mix['unique']}")
    report = run_rollout_dry(
        JsonSnapshot(),
        ALL_TASKS,
        list(DEFAULT_TASKS),
        codec=codec,
        limits=tool.limits,
        dense=cfg.dense_reward,
        bonus=tool.tool_parse_bonus,
    )
    print(format_report(report))
    for episode in report["episodes"]:
        expected = training_reward(episode["dense_total"], episode["parsed_tool_calls"], tool.tool_parse_bonus)
        if episode["reward"]["total"] < 1.0 or abs(episode["training_reward"] - expected) > 1e-9:
            print(f"{episode['task_id']} shaped reward {episode['training_reward']} != {expected}", file=sys.stderr)
            return 1
        if episode["training_reward"] <= episode["reward"]["total"]:
            print(f"{episode['task_id']} parse bonus did not apply", file=sys.stderr)
            return 1
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU dry run for tool GRPO. Does not start Modal or a GPU.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=DEFAULT_TOOL_BUDGET,
        help="tool calls enforced in the env and stated in the system prompt (default 8; use 6 to reproduce v3)",
    )
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=DEFAULT_PROMPT_VERSION,
        help="system prompt (default v4; v3 is the previous text, with the budget number filled in)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
