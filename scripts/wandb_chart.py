"""Pull one W&B run and write the training chart.

Reward, strict (``rewards/env/mean``), memorization gap, and flags
(0 ok, 1 warn, 2 stop and review). The API key is ``WANDB_API_KEY`` in the
environment. It is not an argument and it is not printed.

    uv run python scripts/wandb_chart.py grpo-qwen3b-tools
    uv run python scripts/wandb_chart.py grpo-qwen3b-tools --project bio-rl --entity my-lab --out /tmp/grpo.png

Project defaults to ``bio-rl``. Entity defaults to ``WANDB_ENTITY`` or the key's default entity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.wandb_chart import chart_series, fetch_history, render_chart
from fdagym.train.wandb_log import resolve_wandb_target, wandb_key_present


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a W&B training run to PNG")
    parser.add_argument("run_id")
    parser.add_argument("--project", default="", help="W&B project (default: env WANDB_PROJECT or bio-rl)")
    parser.add_argument("--entity", default="", help="W&B entity (default: env WANDB_ENTITY)")
    parser.add_argument("--out", type=Path, default=None, help="PNG path (default: results/<run_id>-chart.png)")
    args = parser.parse_args(argv)
    if not wandb_key_present():
        print("WANDB_API_KEY is not set. Export it, then rerun this script.", file=sys.stderr)
        return 2
    project, entity = resolve_wandb_target(args.project or None, args.entity or None)
    try:
        rows = fetch_history(args.run_id, project=project, entity=entity)
    except ImportError:
        print("wandb is not installed. Install it with: uv add wandb", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — report the failure type, never the key
        print(f"could not read the W&B run ({type(exc).__name__})", file=sys.stderr)
        return 1
    destination = args.out or (ROOT / "results" / f"{args.run_id}-chart.png")
    render_chart(chart_series(rows), destination, title=args.run_id)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
