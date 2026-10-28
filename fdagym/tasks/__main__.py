"""Write train/test JSONL and the review sheet from a snapshot loader.

    uv run python -m fdagym.tasks --snapshot fdagym.tasks.pilot:load --out data/tasks
"""

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

from fdagym.tasks.config import GenConfig
from fdagym.tasks.generators import generate
from fdagym.tasks.plan import load_universe
from fdagym.tasks.review import render_review


def load_snapshot(spec: str):
    module_name, factory = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, factory)()


def parse_drugs(arg: str | None) -> tuple[str, ...] | None:
    if not arg:
        return None
    path = Path(arg)
    if path.is_file():
        return load_universe(path)
    return tuple(part.strip() for part in arg.split(",") if part.strip())


def write_tasks(snapshot, out: Path, cfg: GenConfig | None = None, drugs=None) -> list[dict]:
    cfg = cfg or GenConfig()
    tasks = generate(snapshot, cfg, drugs)
    if any(task["entity"]["generic_name"].strip().lower() == "ibuprofen" for task in tasks):
        raise ValueError("ibuprofen is not in the universe")
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = [task for task in tasks if task["split"] == split]
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (out / "review.md").write_text(render_review(tasks, cfg))
    return tasks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate FDA-Gym tasks from a snapshot")
    parser.add_argument("--snapshot", required=True, help="import path of a zero-arg loader, as module:function")
    parser.add_argument("--out", type=Path, default=Path("data/tasks"))
    parser.add_argument(
        "--drugs",
        help="path to a name list (default data/snapshots/drugs.txt) or comma-separated generic names",
    )
    args = parser.parse_args(argv)
    tasks = write_tasks(load_snapshot(args.snapshot), args.out, drugs=parse_drugs(args.drugs))
    print(f"wrote {len(tasks)} tasks to {args.out}")
    counts: Counter[tuple[str, str]] = Counter((task["type"], task["difficulty"]) for task in tasks)
    for key in sorted(counts):
        print(f"  {key[0]:22} {key[1]:6} {counts[key]}")
    for split in ("train", "test"):
        rows = [task for task in tasks if task["split"] == split]
        if not rows:
            print(f"  {split}: 0")
            continue
        n_unans = sum(not task["answerable"] for task in rows)
        n_drugs = len({task["entity"]["generic_name"] for task in rows})
        print(f"  {split}: {len(rows)} tasks, {n_drugs} drugs, {n_unans} unanswerable ({100 * n_unans / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
