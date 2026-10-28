"""Plan A GRPO + LoRA. Prefetched oracle context, reward = FDAEnv.score after submit_answer.

Dry run (no torch):

    uv run python scripts/grpo_singleturn.py --dry-run

This script does not rent a GPU. JP starts Modal or RunPod separately — see docs/train.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.agents.baseline import load_split
from fdagym.train.config import QWEN_3B, QWEN_7B, TrainConfig, for_7b
from fdagym.train.dry import format_report, run_dry
from fdagym.train.launch import run_grpo


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    snapshot, tasks = _load_data(args)
    config = _config(args)
    if args.dry_run:
        report = run_dry(snapshot, tasks, config)
        print(format_report(report))
        if report["gold"]["total"] < 1.0:
            print("gold completion did not score 1.0 — check oracle_calls vs rewards", file=sys.stderr)
            return 1
        return 0

    results_dir = Path(args.results_dir)
    try:
        summary = run_grpo(
            snapshot,
            tasks,
            config,
            run_id=args.run_id,
            results_dir=results_dir,
            resume=not args.no_resume,
        )
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(summary)
    return 0


def _config(args) -> TrainConfig:
    checkpointing = args.gradient_checkpointing
    if checkpointing is None:
        checkpointing = args.preset == "7b"
    extra = {}
    if args.vllm_gpu_memory_utilization is not None:
        extra["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.preset == "7b":
        return for_7b(
            model=args.model or QWEN_7B,
            stage=args.stage,
            max_steps=args.max_steps,
            num_generations=args.num_generations,
            learning_rate=args.lr,
            n_distractors=args.distractors,
            use_vllm=args.use_vllm,
            gradient_checkpointing=checkpointing,
            **extra,
        )
    return TrainConfig(
        model=args.model or QWEN_3B,
        stage=args.stage,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        n_distractors=args.distractors,
        use_vllm=args.use_vllm,
        gradient_checkpointing=checkpointing,
        **extra,
    )


def _load_data(args):
    if args.dry_run and not args.use_snapshot:
        from tests.env.support import ALL_TASKS, JsonSnapshot

        return JsonSnapshot(), ALL_TASKS
    from fdagym.openfda import DiskSnapshot

    return DiskSnapshot(args.snapshot), load_split("train")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan A GRPO + LoRA over FDA-Gym")
    parser.add_argument("--dry-run", action="store_true", help="score gold/junk on fixtures, no GPU")
    parser.add_argument("--use-snapshot", action="store_true", help="dry-run against the frozen DiskSnapshot")
    parser.add_argument("--run-id", default="grpo-dev")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--model", default=None, help=f"HF id (default {QWEN_3B}, or {QWEN_7B} with --preset 7b)")
    parser.add_argument("--preset", choices=("3b", "7b"), default="3b")
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--num-generations", type=int, default=TrainConfig.num_generations)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--distractors", type=int, default=TrainConfig.n_distractors)
    parser.add_argument("--snapshot", default="2026-10-15")
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default on for --preset 7b, off for 3b",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoint-* even if present")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
