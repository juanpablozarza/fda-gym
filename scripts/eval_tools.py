"""Greedy tool-call eval dry run. Does not start Modal.

    uv run python scripts/eval_tools.py --dry-run

``--dry-run`` plays two fixture tasks with the scripted oracle and writes a
schema-valid trajectory file under a temporary directory. ``--eval`` loads a
local Hugging Face model (base, or ``--adapter`` as a directory or a Hub repo
id. A private Hub repo needs ``HF_TOKEN``). ``--split public`` reads
``data/tasks/public.jsonl``. The rented job is the same protocol:

    HF_TOKEN=... modal run modal_train.py --action eval-tools --split public \\
        --adapter your-org/fdagym-adapter --run-id public-eval
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.eval_tools import TOOL_BASELINE_ID, TOOL_EVAL_ID, eval_tool_tasks, tool_eval_model
from fdagym.train.rollout import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_TOOL_BUDGET,
    PROMPT_VERSIONS,
    QwenToolCodec,
    RolloutLimits,
)
from fdagym.train.rollout_dry import CharTokenizer, OracleGenerator

DEFAULT_TASKS = ("FG-0001", "FG-0002")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.dry_run == args.eval:
        print(
            "Pass --dry-run for the CPU check, or --eval to score a local model. This script does not start Modal.",
            file=sys.stderr,
        )
        return 2
    if args.memorization and args.dry_run:
        print(
            "--memorization measures hard_train_holdout against hard. The fixture dry run does not do that.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return _dry(args)
    return _eval(args)


def _dry(args) -> int:
    from tests.env.support import ALL_TASKS, JsonSnapshot

    snapshot = JsonSnapshot()
    tasks = [task for task in ALL_TASKS if task["id"] in set(args.task_id)]
    tokenizer = CharTokenizer()
    limits = _limits(args)
    codec = QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    generator = OracleGenerator(tokenizer, codec, snapshot, tasks)
    model = {"name": "oracle-dry", "kind": "trained", "checkpoint_step": None}
    with tempfile.TemporaryDirectory() as tmp:
        summary = eval_tool_tasks(
            snapshot=snapshot,
            tasks=tasks,
            tokenizer=tokenizer,
            generator=generator,
            model=model,
            run_id="eval-tools-dry",
            results_dir=Path(tmp),
            codec=codec,
            limits=limits,
            split="fixture",
        )
    print(summary["models"])
    print(
        f"protocol={summary['protocol']} temperature={summary['eval_temperature']} "
        f"prompt_version={summary['prompt_version']} tool_budget={summary['tool_budget']}"
    )
    mean = next(iter(summary["models"].values()))["mean_reward"]
    if summary["protocol"] != "tool_rollout" or mean < 1.0:
        print(f"dry eval mean reward {mean}", file=sys.stderr)
        return 1
    return 0


def _eval(args) -> int:
    from fdagym.train.config import QWEN_3B
    from fdagym.train.eval_tools import eval_tools_from_disk
    from fdagym.train.hf_rollout import load_hf_rollout

    generator, tokenizer = load_hf_rollout(args.model or QWEN_3B, adapter=args.adapter)
    run_id = args.run_id or (TOOL_EVAL_ID if args.adapter else TOOL_BASELINE_ID)
    limits = _limits(args)
    summary = eval_tools_from_disk(
        tokenizer=tokenizer,
        generator=generator,
        model=tool_eval_model(args.adapter, base_name=args.model or QWEN_3B),
        run_id=run_id,
        split=args.split,
        results_dir=Path(args.results_dir),
        limit=args.limit,
        limits=limits,
        adapter=args.adapter,
    )
    print(summary["models"])
    print(f"prompt_version={summary['prompt_version']} tool_budget={summary['tool_budget']}")
    if args.memorization:
        from fdagym.agents.baseline import write_summaries
        from fdagym.openfda import DiskSnapshot
        from fdagym.train.eval_tools import eval_memorization

        report = eval_memorization(
            snapshot=DiskSnapshot("2026-10-15"),
            tokenizer=tokenizer,
            generator=generator,
            model=tool_eval_model(args.adapter, base_name=args.model or QWEN_3B),
            limits=limits,
            run_id=f"{run_id}-memorization",
        )
        summary["memorization"] = report
        write_summaries(Path(args.results_dir) / run_id, summary)
        from fdagym.agents.baseline import load_split, read_trajectories
        from fdagym.train.wandb_log import log_eval_summary

        log_eval_summary(
            run_id=run_id,
            summary=summary,
            tasks=load_split(args.split),
            trajectories=read_trajectories(Path(args.results_dir) / f"{run_id}.jsonl"),
            prompt_version=limits.prompt_version,
            tool_budget=limits.tool_budget,
            adapter=args.adapter,
            model=tool_eval_model(args.adapter, base_name=args.model or QWEN_3B)["name"],
        )
        print(json.dumps(report["flags"]))
    return 0


def _limits(args) -> RolloutLimits:
    return RolloutLimits(temperature=0.0, tool_budget=args.tool_budget, prompt_version=args.prompt_version)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU dry run or local tool eval. Does not start Modal.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval", action="store_true", help="score a local model; does not call Modal")
    parser.add_argument("--task-id", action="append", dest="task_id")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None, help="local adapter dir or a Hugging Face repo id")
    parser.add_argument("--split", default="test", help="task split jsonl name, such as test or hard")
    parser.add_argument(
        "--memorization",
        action="store_true",
        help="after --eval, measure known-drug holdout vs hard and write summary['memorization']",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--run-id",
        default=None,
        help="default baseline-qwen3b-tools, or grpo-qwen3b-tools-eval with --adapter",
    )
    parser.add_argument("--results-dir", default="results")
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
    args = parser.parse_args(argv)
    if not args.task_id:
        args.task_id = list(DEFAULT_TASKS)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
