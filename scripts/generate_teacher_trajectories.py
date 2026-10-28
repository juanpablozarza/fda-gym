"""Generate scored multi-turn OpenRouter teacher JSONL for distill SFT.

Examples:
    uv run python scripts/generate_teacher_trajectories.py --run-id teachers-01 --limit 5
    uv run python scripts/generate_teacher_trajectories.py --run-id teachers-pool --tasks data/tasks/distill_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fdagym  # noqa: F401 — loads .env for OPENROUTER_API_KEY
from fdagym.train.teacher_generation import (
    DEFAULT_COST_CAP_USD,
    DEFAULT_TEACHERS,
    prefixed_model,
    run_from_files,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "distill"


def _csv(value: str) -> list[str]:
    return [prefixed_model(item) for item in value.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Used in the output filename; choose a fresh name for new data.")
    parser.add_argument("--split", choices=("train", "test", "hard_train", "hard"), default="train")
    parser.add_argument("--tasks", type=Path, help="Optional task JSONL; overrides --split.")
    parser.add_argument("--snapshot-version", help="Override only when all tasks use this recorded snapshot.")
    parser.add_argument("--models", type=_csv, default=list(DEFAULT_TEACHERS))
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per teacher and question.")
    parser.add_argument("--limit", type=int, help="Limit tasks for a small pilot run.")
    parser.add_argument("--tool-budget", type=int, default=8)
    parser.add_argument("--cost-cap", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to wait after each completed episode.")
    parser.add_argument("--output", type=Path, help="Defaults to results/distill/<run-id>.jsonl.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id or args.run_id in {".", ".."}:
        raise SystemExit("--run-id must be a single path segment")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (copy the project .env or export it)")
    output = args.output or RESULTS / f"{args.run_id}.jsonl"
    result = run_from_files(
        split=args.split,
        tasks_path=args.tasks,
        snapshot_version=args.snapshot_version,
        models=args.models,
        output=output,
        tool_budget=args.tool_budget,
        attempts=args.attempts,
        limit=args.limit,
        cost_cap_usd=args.cost_cap,
        pause_s=args.pause,
    )
    print(json.dumps(result, indent=2))
    return 2 if result["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
