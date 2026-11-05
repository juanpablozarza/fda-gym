"""Keyless, offline demo of one published FDA-Gym episode.

    uv run python scripts/demo_public_episode.py
    uv run python scripts/demo_public_episode.py --task-id FG-9101

Replays one row from ``data/tasks/public.jsonl`` (FG-9101–FG-9193) through
``FDAEnv`` on the committed ``2026-10-15`` DiskSnapshot. After every tool hop
and after ``submit_answer`` it prints the tool, a truncated observation, and
the grader breakdown (correctness / citation / format / efficiency / total /
grader_version). No OpenRouter key, no GPU, no live openFDA.

Default ``FG-9101`` is a 3-call FAERS compare: of abacavir, pravastatin, and
aripiprazole, which has the most reports of FATIGUE? It is short, stays on one
tool, and is the same public row the README uses for the one-call guess that
scores correctness 0. The honest oracle (every stored ``oracle_call``, then
gold + gold citations from this episode) should total 1.0.

This is not the trainer Qwen-codec dry run. That entrypoint stays
``scripts/tool_rollout_dry.py`` (see docs/tool-rollouts.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.demo import (
    DEFAULT_TASK_ID,
    HONEST_TOTAL_MIN,
    DemoError,
    loaded_llm_clients,
    play_public_task,
)

PUBLIC_PATH = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT_ROOT = ROOT / "data" / "snapshots"


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    clients = loaded_llm_clients()
    if clients:
        print(f"demo must stay keyless; loaded {clients}", file=sys.stderr)
        return 2
    try:
        reward = play_public_task(
            args.task_id,
            public_path=PUBLIC_PATH,
            snapshot_root=SNAPSHOT_ROOT,
            write=print,
        )
    except DemoError as exc:
        print(exc, file=sys.stderr)
        return 2
    total = float(reward["total"])
    if total < HONEST_TOTAL_MIN:
        print(f"honest oracle total {total:.4f} is below {HONEST_TOTAL_MIN}", file=sys.stderr)
        return 1
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one public.jsonl task through the grader (offline, no LLM)")
    parser.add_argument(
        "--task-id",
        default=DEFAULT_TASK_ID,
        help=f"published id FG-9101–FG-9193 (default {DEFAULT_TASK_ID}, a 3-call FAERS compare)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
