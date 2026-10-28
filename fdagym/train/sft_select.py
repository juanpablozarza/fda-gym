"""Which tasks the tool-mode SFT warm-up may supervise.

The default is the current train split, the same pool as the v3 warm-up.
``include_hard_train`` appends ``hard_train`` oracle rows. The known-drug
holdout and the hard eval are never rows, even when that option is on.
"""

from __future__ import annotations

import hashlib

from fdagym.train.curriculum import hard_train_holdout_ids, stage_pool
from fdagym.train.hard_mix import file_ids


def task_ids_digest(task_ids: list[str] | tuple[str, ...]) -> str:
    """SHA-256 of the sorted ids joined by newlines. No trailing newline.

    ``scripts/build_warmup.py`` writes this same digest. A spawn passes it as
    ``task_ids_sha256`` so a swapped id file fails before the first step.
    """
    return hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest()


def sft_sources(
    tasks: list[dict],
    config,
    hard_tasks: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Stage-1 train pool, plus ``hard_train`` only when the config asks for it.

    ``config.task_ids`` replaces that pool: the rows are exactly those ids, looked
    up in ``tasks`` and ``hard_tasks``. Holdout and hard eval still fail the guard.
    """
    if config.task_ids is not None:
        selected = resolve_sft_task_ids(tasks, hard_tasks, config.task_ids, config.task_ids_sha256)
        # include_hard_train=True lets a hard_train id through. Holdout and hard eval still raise.
        assert_sft_rows_allowed(selected, include_hard_train=True)
        return selected, _explicit_mix(selected)
    selected, mix = stage_pool(tasks, config.stage)
    if config.include_hard_train:
        if not hard_tasks:
            raise RuntimeError("include_hard_train requires the hard_train split")
        selected = [*selected, *hard_tasks]
        mix = {**mix, "hard_train": len(hard_tasks)}
    assert_sft_rows_allowed(selected, include_hard_train=config.include_hard_train)
    return selected, mix


def resolve_sft_task_ids(
    tasks: list[dict],
    hard_tasks: list[dict] | None,
    task_ids: list[str] | tuple[str, ...],
    task_ids_sha256: str | None = None,
) -> list[dict]:
    """Rows for ``task_ids``, in that order, from train and hard_train only."""
    if not task_ids:
        raise RuntimeError("task_ids is empty")
    cleaned = [str(task_id).strip() for task_id in task_ids]
    if any(not task_id for task_id in cleaned):
        raise RuntimeError("task_ids contains a blank id")
    seen: set[str] = set()
    duplicates: list[str] = []
    for task_id in cleaned:
        if task_id in seen and task_id not in duplicates:
            duplicates.append(task_id)
        seen.add(task_id)
    if duplicates:
        shown = ", ".join(duplicates[:8])
        raise RuntimeError(f"duplicate task ids ({len(duplicates)}): {shown}")
    expected = (task_ids_sha256 or "").strip().lower()
    if expected:
        digest = task_ids_digest(cleaned)
        if digest != expected:
            raise RuntimeError(f"task_ids sha256 {digest} != {expected}")
    train_by: dict[str, dict] = {}
    for task in tasks:
        train_by.setdefault(str(task.get("id")), task)
    hard_by: dict[str, dict] = {}
    for task in hard_tasks or []:
        hard_by.setdefault(str(task.get("id")), task)
    both = [task_id for task_id in cleaned if task_id in train_by and task_id in hard_by]
    if both:
        shown = ", ".join(both[:8])
        raise RuntimeError(f"task id is in both train and hard_train ({len(both)}): {shown}")
    missing = [task_id for task_id in cleaned if task_id not in train_by and task_id not in hard_by]
    if missing:
        shown = ", ".join(missing[:8])
        raise RuntimeError(f"unknown task ids ({len(missing)}): {shown}")
    return [train_by[task_id] if task_id in train_by else hard_by[task_id] for task_id in cleaned]


def _explicit_mix(tasks: list[dict]) -> dict:
    by_split: dict[str, int] = {}
    for task in tasks:
        split = str(task.get("split") or "")
        by_split[split] = by_split.get(split, 0) + 1
    return {"task_ids": len(tasks), "by_split": by_split}


def assert_sft_rows_allowed(tasks: list[dict], *, include_hard_train: bool) -> None:
    """Raise if a warm-up row is holdout, hard eval, or an unexpected ``hard_train`` task."""
    holdout = hard_train_holdout_ids()
    hard_eval = file_ids("hard.jsonl") | file_ids("public.jsonl")
    blocked = []
    for task in tasks:
        task_id = str(task.get("id"))
        split = task.get("split")
        if task_id in holdout or split == "hard_train_holdout":
            blocked.append(f"{task_id} holdout")
        elif task_id in hard_eval or split in {"hard", "public"}:
            blocked.append(f"{task_id} hard eval")
        elif split == "hard_train" and not include_hard_train:
            blocked.append(f"{task_id} hard_train")
    if blocked:
        shown = ", ".join(blocked[:8])
        raise RuntimeError(
            f"SFT refuses holdout, hard eval, and hard_train rows unless include_hard_train is set "
            f"({len(blocked)}: {shown})"
        )
