"""SFT warm-up on oracle tool trajectories. Does not start Modal.

    uv run python scripts/sft_warmup.py --dry-run

``--dry-run`` plays two fixture tasks on the CPU and checks the loss mask.
``--train`` fits the warm-up on a GPU that is already in this process. It does
not call Modal. JP starts the rented job with ``modal run modal_train.py --action sft``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.rollout_dry import CharTokenizer
from fdagym.train.sft_tools import IGNORE_INDEX, build_sft_rows

DEFAULT_TASKS = ("FG-0001", "FG-0002")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.dry_run == args.train:
        print(
            "Pass --dry-run for the CPU check, or --train to fit on a local GPU. This script does not start Modal.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return _dry(args)
    return _train(args)


def _dry(args) -> int:
    snapshot, tasks = _fixtures()
    built = build_sft_rows(snapshot, tasks, list(args.task_id), CharTokenizer(), max_length=8192, run_id="sft-dry")
    print(f"kept={built['n_kept']} skipped={built['n_skipped']}")
    if built["n_kept"] != len(args.task_id) or built["n_skipped"]:
        print("oracle SFT rows were skipped", file=sys.stderr)
        return 1
    for row, rollout in zip(built["rows"], built["rollouts"], strict=True):
        prompt_labels = row["labels"][: len(rollout.prompt_ids)]
        if prompt_labels != [IGNORE_INDEX] * len(rollout.prompt_ids):
            print(f"{row['task_id']} prompt labels are not masked", file=sys.stderr)
            return 1
        supervised = sum(1 for label in row["labels"] if label != IGNORE_INDEX)
        if supervised != sum(rollout.loss_mask):
            print(f"{row['task_id']} supervised {supervised} != loss mask {sum(rollout.loss_mask)}", file=sys.stderr)
            return 1
        print(f"{row['task_id']} tokens={len(row['input_ids'])} supervised={supervised}")
    return 0


def _train(args) -> int:
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.train.sft_config import SFTConfig
    from fdagym.train.sft_launch import run_sft

    summary = run_sft(
        DiskSnapshot(args.snapshot),
        load_split("train"),
        SFTConfig(),
        run_id=args.run_id,
        results_dir=Path(args.results_dir),
    )
    print(summary)
    return 0


def _fixtures():
    from tests.env.support import ALL_TASKS, JsonSnapshot

    return JsonSnapshot(), ALL_TASKS


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU dry run or local SFT warm-up. Does not start Modal.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true", help="fit on the local GPU; does not call Modal")
    parser.add_argument("--task-id", action="append", dest="task_id")
    parser.add_argument("--snapshot", default="2026-10-15")
    parser.add_argument("--run-id", default="sft-qwen3b-tools")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    if not args.task_id:
        args.task_id = list(DEFAULT_TASKS)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
