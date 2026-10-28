"""Score an untrained (or LoRA) Qwen checkpoint on the held-out split, Plan A protocol.

Does not call OpenRouter. Needs a GPU for the real 3B/7B; tests inject a fake generate.

    uv run python scripts/eval_local.py --model Qwen/Qwen2.5-3B-Instruct --run-id baseline-qwen3b
    uv run python scripts/eval_local.py --model Qwen/Qwen2.5-7B-Instruct --run-id baseline-qwen7b
    uv run python scripts/eval_local.py --model Qwen/Qwen2.5-3B-Instruct --adapter results/grpo-qwen3b/adapter \\
        --run-id grpo-qwen3b-eval --kind trained
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.config import QWEN_3B
from fdagym.train.eval_local import eval_from_disk


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    from fdagym.train.local_lm import load_generator

    generate = load_generator(
        args.model,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seeds[0] if args.seeds else 0,
    )
    summary = eval_from_disk(
        generate=generate,
        model_name=args.model if not args.adapter else f"{args.model}+lora",
        run_id=args.run_id,
        split=args.split,
        results_dir=Path(args.results_dir),
        snapshot_version=args.snapshot,
        kind=args.kind,
        limit=args.limit,
        seeds=args.seeds,
    )
    print(json.dumps({k: summary[k] for k in ("run_id", "split", "models", "leaderboard") if k in summary}, indent=2))
    return 0


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=QWEN_3B)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory from a GRPO run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--snapshot", default="2026-10-15")
    parser.add_argument("--kind", default="baseline", choices=("baseline", "trained"))
    parser.add_argument("--limit", type=int, default=None, help="score only the first N tasks (smoke)")
    parser.add_argument("--seeds", default="1", type=_ints, help="repeat the split; pass 1,2,3 for a CI gate")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
