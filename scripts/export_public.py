"""Copy a clean tree without git history, env files, or build artifacts.

uv run python scripts/export_public.py /tmp/fda-gym-export
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
            f"{ROOT}/",
            f"{dest}/",
        ],
        check=True,
    )
    if not (dest / "data" / "tasks" / "public.jsonl").exists():
        print("data/tasks/public.jsonl is missing from the export", file=sys.stderr)
        return 1
    print(f"exported {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
