"""Re-score a stored run with the current grader and print what moved."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from fdagym.rewards import score
from fdagym.rewards.version import DATA_VERSION, GRADER_VERSION
from fdagym.tasks.hard_solver import Unsolvable, parse_question

ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = (
    "train.jsonl",
    "test.jsonl",
    "hard.jsonl",
    "hard_train.jsonl",
    "hard_train_holdout.jsonl",
    "sponsor_lookup.jsonl",
    "public.jsonl",
)


def load_tasks() -> dict[str, dict]:
    found: dict[str, dict] = {}
    source: dict[str, str] = {}
    directory = ROOT / "data" / "tasks"
    for name in TASK_FILES:
        path = directory / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            previous = source.get(task["id"])
            if previous is not None:
                raise SystemExit(f"{task['id']} is in {previous} and {name}")
            found[task["id"]] = task
            source[task["id"]] = name
    return found


def family_of(task: dict) -> str:
    try:
        return str(parse_question(task["question"])["variant"])
    except (Unsolvable, KeyError, TypeError, ValueError):
        return str(task.get("type") or "unknown")


def model_of(row: dict) -> str:
    model = row.get("model")
    if isinstance(model, dict):
        return str(model.get("name") or "unknown")
    if isinstance(model, str) and model:
        return model
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args(argv)
    tasks = load_tasks()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    moved = 0
    missing = 0
    n = 0
    families: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        n += 1
        task = tasks.get(row.get("task_id"))
        if task is None:
            missing += 1
            print(f"{row.get('task_id')} missing task")
            continue
        new = score({"steps": row.get("steps") or [], "final_answer": row.get("final_answer")}, task)
        old_reward = row["reward"]
        old = float(old_reward["total"])
        families[(model_of(row), family_of(task))].append(
            (old, new["total"], float(old_reward.get("correctness") or 0), float(new["correctness"]))
        )
        if abs(new["total"] - old) > 1e-9:
            moved += 1
            print(
                f"{row.get('task_id')} {old:.4f} -> {new['total']:.4f}"
                f"  correctness {old_reward.get('correctness')} -> {new['correctness']}"
                f"  efficiency {old_reward.get('efficiency')} -> {new['efficiency']}"
            )
    for (model, family), group in sorted(families.items()):
        count = len(group)
        old_mean = sum(item[0] for item in group) / count
        new_mean = sum(item[1] for item in group) / count
        old_pass = sum(1 for item in group if item[2] > 0)
        new_pass = sum(1 for item in group if item[3] > 0)
        print(f"{model} {family} n={count} mean {old_mean:.4f} -> {new_mean:.4f} pass {old_pass} -> {new_pass}")
    print(f"grader {GRADER_VERSION} data {DATA_VERSION}: {moved} of {n} rows moved, {missing} tasks missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
