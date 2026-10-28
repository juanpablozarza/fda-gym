"""Find a TRL/HF checkpoint directory so GRPO can resume."""

from pathlib import Path


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    found: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.split("-", 1)[-1]
        if not suffix.isdigit():
            continue
        found.append((int(suffix), path))
    if not found:
        return None
    return max(found)[1]
