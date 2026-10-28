"""CPU dry run of the multi-turn rollout in Qwen tool-call format. No torch, no vLLM, no Modal.

    uv run python scripts/tool_rollout_dry.py

Plays two fixture tasks through FDAEnv with a scripted oracle. Exits 1 if either
episode scores below 1.0. This does not rent a GPU. See docs/tool-rollouts.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.rollout_dry import format_report, resolve_task_ids, run_rollout_dry


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    snapshot, tasks = _load(args)
    try:
        task_ids = resolve_task_ids(tasks, args.task_id)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    report = run_rollout_dry(snapshot, tasks, task_ids)
    print(format_report(report))
    short = [episode for episode in report["episodes"] if episode["reward"]["total"] < 1.0]
    if short:
        print("scripted oracle rollout scored below 1.0", file=sys.stderr)
        return 1
    return 0


def _load(args):
    if args.use_snapshot:
        from fdagym.agents.baseline import load_split
        from fdagym.openfda import DiskSnapshot

        # Train first so the no-id default is a train row, not a TEST FG-0001.
        return DiskSnapshot(args.snapshot), load_split("train") + load_split("test")
    from tests.env.support import ALL_TASKS, JsonSnapshot

    return JsonSnapshot(), ALL_TASKS


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU dry run of live FDAEnv rollouts")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_id",
        help="repeatable; default is the first two gradable tasks in the loaded sheet",
    )
    parser.add_argument("--use-snapshot", action="store_true", help="read DiskSnapshot instead of fixtures")
    parser.add_argument("--snapshot", default="2026-10-15")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
