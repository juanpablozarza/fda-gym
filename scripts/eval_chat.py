"""Score one model on one task split with a Chat Completions client.

    uv run python scripts/eval_chat.py --model <slug> --split hard --run-id <id>
"""

from __future__ import annotations

import argparse
import json

import fdagym  # noqa: F401  loads .env
from fdagym.agents.baseline import ROOT, SKIP_TYPES, append_trajectory, load_split
from fdagym.agents.chat import run_chat_episode
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot


def _complete(**kwargs):
    from litellm import completion

    return completion(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    tasks = load_split(args.split)
    runnable = [task for task in tasks if task["type"] not in SKIP_TYPES]
    if args.limit is not None:
        runnable = runnable[: args.limit]
    env = FDAEnv(DiskSnapshot(args.snapshot_version), tasks, run_id=args.run_id, seed=0)
    results = ROOT / "results"
    out = results / f"{args.run_id}.jsonl"
    if out.exists():
        out.unlink()
    rewards = []
    for task in runnable:
        trajectory = run_chat_episode(
            env,
            task["id"],
            _complete,
            model=args.model,
            temperature=args.temperature,
        )
        trajectory["id"] = f"{args.run_id}/{args.model}/{task['id']}/s0"
        trajectory["run_id"] = args.run_id
        append_trajectory(out, trajectory)
        reward = trajectory["reward"]
        rewards.append(reward)
        print(
            f"{task['id']} {reward['total']:.3f} {reward['correctness']:.3f} {' | '.join(reward['notes'])}",
            flush=True,
        )
    summary = _summary(args.model, args.split, rewards)
    run_dir = results / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"mean_reward={summary['mean_reward']:.4f} pass_rate={summary['pass_rate']:.4f}", flush=True)
    return 0


def _summary(model: str, split: str, rewards: list[dict]) -> dict:
    n = len(rewards)
    mean_reward = round(sum(row["total"] for row in rewards) / n, 4) if n else 0.0
    pass_rate = round(sum(1 for row in rewards if row["correctness"] > 0) / n, 4) if n else 0.0
    return {"model": model, "split": split, "n": n, "mean_reward": mean_reward, "pass_rate": pass_rate}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, choices=("train", "test", "hard", "public", "sponsor_lookup"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--snapshot-version", default="2026-10-15")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
