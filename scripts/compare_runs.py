"""Print whether a trained Plan A eval beat its untrained baseline.

    uv run python scripts/compare_runs.py results/baseline-qwen3b results/grpo-qwen3b-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.compare import compare_summaries, format_compare, load_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("trained")
    args = parser.parse_args(argv)
    row = compare_summaries(load_summary(args.baseline), load_summary(args.trained))
    print(format_compare(row))
    print(json.dumps(row, indent=2))
    return 0 if row["beats_baseline"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
