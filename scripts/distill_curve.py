"""Collect distill-SFT hard evals into CSV + SVG. No GPU.

    uv run python scripts/distill_curve.py \\
      --runs-dir /workspace/runs \\
      --prefix distill-sft-qwen3b \\
      --sizes 100,250,500,1000,all \\
      --out results/distill-curve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.distill import DEFAULT_RUN_ID, DEFAULT_SIZES
from fdagym.train.distill_curve import collect_curve, write_curve


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    records = collect_curve(args.runs_dir, args.prefix, args.sizes, split=args.split)
    written = write_curve(records, args.out)
    print(json.dumps({"n": written["n"], "csv": written["csv"], "svg": written["svg"]}, indent=2))
    overall = [row for row in records if not row.get("variant") and not row.get("missing")]
    for row in overall:
        print(f"n={row.get('size_n')} reward={row.get('mean_reward')} pass={row.get('pass_rate')} run={row['run_id']}")
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning-curve CSV/SVG from distill-sft eval-tools runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default=DEFAULT_RUN_ID)
    parser.add_argument("--sizes", default=",".join(DEFAULT_SIZES))
    parser.add_argument("--split", default="hard")
    parser.add_argument("--out", type=Path, default=Path("results/distill-curve"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
