"""Refuse to build a GPU image if the frozen snapshot is not on disk."""

from pathlib import Path

SNAPSHOT_VERSION = "2026-10-15"


def snapshot_manifest(repo: Path | None = None) -> Path:
    root = repo or Path(__file__).resolve().parents[2]
    return root / "data" / "snapshots" / SNAPSHOT_VERSION / "manifest.jsonl"


def require_snapshot(repo: Path | None = None) -> Path:
    path = snapshot_manifest(repo)
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen snapshot missing at {path}. "
            "Copy data/snapshots/2026-10-15 from the main worktree before `modal run` or docker build."
        )
    return path
