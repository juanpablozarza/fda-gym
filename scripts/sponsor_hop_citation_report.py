"""Replay sponsor-hop trajectories, or say that none are stored.

Looks through results/, web/public/results/, and schemas/examples for a
trajectory whose task is sponsor_drug, sponsor_faers, or sponsor_count.
The committed eval jsonl files are the TEST-split baseline and a dashboard
fixture, not the hard set. When nothing matches, this prints the four
synthetic dense scores from tests/rewards/test_sponsor_hop_citation.py.

Does not log to W&B and does not start training.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = (
    ROOT / "data/tasks/hard.jsonl",
    ROOT / "data/tasks/hard_train.jsonl",
    ROOT / "data/tasks/hard_train_holdout.jsonl",
)
TRAJECTORY_GLOBS = (
    "results/**/*.jsonl",
    "web/public/results/**/*.jsonl",
    "schemas/examples/*.json",
    "schemas/examples/*.jsonl",
)


def _sponsor_tasks() -> dict[str, dict]:
    from fdagym.tasks.hard_solver import parse_question

    found = {}
    for path in TASK_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            if task.get("type") != "sponsor_hop":
                continue
            variant = parse_question(task["question"])["variant"]
            if variant in {"sponsor_drug", "sponsor_faers", "sponsor_count"}:
                found[task["id"]] = task
    return found


def _rows(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".json":
        body = json.loads(text)
        if isinstance(body, dict) and "steps" in body:
            return [body]
        if isinstance(body, list):
            return [row for row in body if isinstance(row, dict)]
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _trajectory_files() -> list[Path]:
    found = []
    for pattern in TRAJECTORY_GLOBS:
        found.extend(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(found)


def _load_cases():
    path = ROOT / "tests/rewards/test_sponsor_hop_citation.py"
    spec = importlib.util.spec_from_file_location("sponsor_hop_citation_cases", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    return module.ordering_cases()


def main() -> int:
    tasks = _sponsor_tasks()
    files = _trajectory_files()
    matched = []
    scanned = 0
    for path in files:
        for row in _rows(path):
            scanned += 1
            task_id = row.get("task_id")
            if task_id in tasks and isinstance(row.get("steps"), list):
                matched.append((path, row))

    print(f"sponsor-hop tasks in the hard splits: {len(tasks)}")
    print(f"trajectory files scanned: {len(files)}")
    print(f"trajectory rows scanned: {scanned}")
    if not matched:
        print("no sponsor-hop eval trajectories in the repo or fixtures.")
    else:
        from fdagym.rewards import score

        print(f"rescoring {len(matched)} stored sponsor-hop trajectories:")
        for path, row in matched:
            task = tasks[row["task_id"]]
            dense = score(
                {"steps": row.get("steps") or [], "final_answer": row.get("final_answer")},
                task,
                dense=True,
            )
            rel = path.relative_to(ROOT)
            print(f"  {rel} {row['task_id']} dense={dense['total']:.4f} citation={dense['citation']:.2f}")

    print("synthetic dense scores (tests/rewards/test_sponsor_hop_citation.py):")
    cases = _load_cases()
    order = ("full_two_hop", "two_hop_wrong", "honest_failure", "one_hop_guess")
    totals = []
    for name in order:
        reward = cases[name]
        totals.append(reward["total"])
        print(
            f"  {name:16} total={reward['total']:.4f}  correctness={reward['correctness']:.4f}"
            f"  citation={reward['citation']:.2f}  format={reward['format']:.2f}"
            f"  efficiency={reward['efficiency']:.2f}"
        )
    anchor = cases["anchor_only"]
    print(
        f"  {'anchor_only':16} total={anchor['total']:.4f}  correctness={anchor['correctness']:.4f}"
        f"  citation={anchor['citation']:.2f}  format={anchor['format']:.2f}"
        f"  efficiency={anchor['efficiency']:.2f}"
    )
    print("  anchor_only is the application lookup alone. one_hop_guess also searches the other named drugs.")
    if not (totals[0] > totals[1] >= totals[2] > totals[3]):
        print("ordering failed: expected full two-hop > two-hop wrong >= honest failure > one-hop guess")
        return 1
    print("ordering: full_two_hop > two_hop_wrong >= honest_failure > one_hop_guess")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
