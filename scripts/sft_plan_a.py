"""Plan A SFT warm-up. One LoRA epoch on train-split prompts whose gold is visible.

Dry build (no torch, no GPU):

    uv run python scripts/sft_plan_a.py --dry-build

Training is started by JP on Modal (`modal run modal_train.py --action sft-plan-a`).
This script does not rent a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.agents.baseline import load_split
from fdagym.openfda import DiskSnapshot
from fdagym.train.config import TrainConfig
from fdagym.train.sft import SFT_RUN_ID, build_sft_rows, run_sft


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    snapshot = DiskSnapshot(args.snapshot)
    tasks = load_split("train")
    config = TrainConfig()
    if args.dry_build:
        _rows, stats = build_sft_rows(snapshot, tasks, config)
        print(json.dumps(stats, indent=2))
        if stats["n_kept"] < 1:
            print("no SFT rows kept", file=sys.stderr)
            return 1
        return 0
    try:
        summary = run_sft(snapshot, tasks, config, output_dir=Path(args.output_dir))
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(summary)
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan A SFT warm-up over packed FDA-Gym prompts")
    parser.add_argument("--dry-build", action="store_true", help="count kept and skipped rows, no GPU")
    parser.add_argument("--snapshot", default="2026-10-15")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / SFT_RUN_ID),
        help="adapter is written to <output-dir>/adapter",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
