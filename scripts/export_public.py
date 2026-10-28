"""Write a clean public tree. No git history, and no private hard sheet.

    uv run python scripts/export_public.py /tmp/fda-gym-public

Copies the worktree, then removes:

- data/tasks/hard.jsonl
- data/tasks/review.md
- results/hard-*

.git, .env, and virtualenv or build directories are not copied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = (
    Path("data/tasks/hard.jsonl"),
    Path("data/tasks/review.md"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args(argv)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} is not empty", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            ".git",
            "--exclude",
            ".venv",
            "--exclude",
            ".env",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".pytest_cache",
            "--exclude",
            ".ruff_cache",
            "--exclude",
            "node_modules",
            "--exclude",
            ".next",
            "--exclude",
            ".DS_Store",
            "--exclude",
            "data/tasks/hard.jsonl",
            "--exclude",
            "data/tasks/review.md",
            "--exclude",
            "results/hard-*",
            "--exclude",
            "videos/hard-frontier",
            f"{ROOT}/",
            f"{dest}/",
        ],
        check=True,
    )
    leaked = _leaks(dest)
    if leaked:
        print("private paths still in the export:", file=sys.stderr)
        for item in leaked:
            print(f"  {item}", file=sys.stderr)
        return 1
    if not (dest / "data" / "tasks" / "public.jsonl").exists():
        print("data/tasks/public.jsonl is missing from the export", file=sys.stderr)
        return 1
    print(f"exported {dest}")
    return 0


def _leaks(dest: Path) -> list[str]:
    found = []
    for relative in PRIVATE:
        if (dest / relative).exists():
            found.append(str(relative))
    results = dest / "results"
    if results.exists():
        found.extend(str(path.relative_to(dest)) for path in results.glob("hard-*"))
    return found


if __name__ == "__main__":
    raise SystemExit(main())
