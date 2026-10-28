"""CPU dry-run for teacher-trajectory distill SFT. Does not start a GPU job.

    uv run python scripts/sft_distill.py --dry-run

Same pipeline as ``scripts/run_gpu.py distill-sft --dry-run``: load JSONL,
keep passing rows, nest subsets, tokenize with the char tokenizer, check
that loss is assistant-only. It never loads Qwen and never calls a teacher API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.distill import DEFAULT_RUN_ID, fixture_path


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    from fdagym.train.jobs import run_distill_sft

    result = run_distill_sft(
        runs_dir=args.runs_dir,
        trajectories=args.trajectories,
        run_id=args.run_id,
        sizes=args.sizes,
        seed=args.seed,
        teachers=args.teacher,
        templates=args.template,
        max_attempts_per_question=args.max_attempts_per_question,
        tool_budget=args.tool_budget,
        prompt_version=args.prompt_version,
        dry_run=True,
    )
    print(json.dumps({"run_id": result["run_id"], "dry_run": True, "filter": result["filter"]}, indent=2))
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET, PROMPT_VERSIONS

    parser = argparse.ArgumentParser(description="CPU dry-run for distill SFT. Does not load Qwen or start Modal.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--trajectories", action="append", type=Path, default=None)
    parser.add_argument("--sizes", default="3,5,all", help="nested sizes (fixture dry-run default 3,5,all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", action="append", default=None)
    parser.add_argument("--template", action="append", default=None)
    parser.add_argument("--max-attempts-per-question", type=int, default=None)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--tool-budget", type=int, default=DEFAULT_TOOL_BUDGET)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=DEFAULT_PROMPT_VERSION)
    args = parser.parse_args(argv)
    if not args.trajectories:
        args.trajectories = [fixture_path()]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
