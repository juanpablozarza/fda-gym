"""Pre-run filter for the hard GRPO training prompts.

The private eval sheet when it is present, the published sheet, TEST, the known-drug holdout, and
the memorization probes are not inputs here. ``filter_current_tasks`` only sees
the current train file, and it raises if one of those ids or splits shows up.
``hard_train`` is the other side of the 60/40 mix and is not filtered.

``base_pass_rate`` is null on the frozen tasks, so a rate of 0 or 1 drops a
prompt only when a rate is present (on the task, or in ``pass_rates``). With
no rate, non-protected current tasks whose structural ``difficulty`` is in
``drop_difficulties`` (default ``easy``) are dropped. ``boxed_warning`` is
protected: the hard eval needs that skill, and an easy boxed-warning row is
still that type. Composite is not protected here; the GRPO pool allowlist
drops it before this filter runs.

After the 60/40 draw, ``apply_protected_floor`` tops each protected type up to
``protected_floor`` of the current slice by replacing other current rows. The
hard-train count is unchanged.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from fdagym.train.curriculum import in_training_pool

# Splits that are measured, not trained. ``hard_train`` is refused here too:
# this filter only edits the current side of the mix.
FORBIDDEN_SPLITS = frozenset({"test", "hard", "hard_train", "hard_train_holdout"})
PROTECTED_TYPES = ("boxed_warning",)


@dataclass(frozen=True)
class PromptFilter:
    enabled: bool = False
    drop_solved: bool = True
    drop_failed: bool = True
    solved_at: float = 1.0
    failed_at: float = 0.0
    drop_difficulties: tuple[str, ...] = ("easy",)
    protect_types: tuple[str, ...] = PROTECTED_TYPES
    # Share of the *current* slice reserved for each protected type.
    protected_floor: float = 0.25
    pass_rates: dict[str, float] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.protected_floor < 0:
            raise ValueError("protected_floor must be >= 0")
        if self.protect_types and self.protected_floor * len(self.protect_types) > 1 + 1e-9:
            raise ValueError("protected floors must sum to at most 1 of the current slice")
        if self.solved_at < self.failed_at:
            raise ValueError("solved_at must be >= failed_at")


def hard_prompt_filter(**overrides) -> PromptFilter:
    """Filter used when a hard-mix run leaves ``difficulty_filter`` on its default."""
    return PromptFilter(enabled=True, **overrides)


def filter_current_tasks(
    tasks: list[dict],
    config: PromptFilter,
    *,
    blocked_ids: set[str] | frozenset[str] | None = None,
) -> tuple[list[dict], dict]:
    """Drop saturated current prompts. Protected types stay. Eval ids raise."""
    if not config.enabled:
        raise ValueError("filter_current_tasks requires an enabled PromptFilter")
    blocked = set(blocked_ids or ())
    kept: list[dict] = []
    dropped: list[dict] = []
    skipped_pool = 0
    for task in tasks:
        _refuse_held_out(task, blocked)
        if not in_training_pool(task):
            skipped_pool += 1
            continue
        if _drop(task, config):
            dropped.append(task)
        else:
            kept.append(task)
    if not kept:
        raise RuntimeError("difficulty filter removed every current training prompt")
    report = {
        "enabled": True,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_skipped_not_in_pool": skipped_pool,
        "dropped_by_type": _counts(dropped),
        "kept_by_type": _counts(kept),
        "protect_types": list(config.protect_types),
        "drop_difficulties": list(config.drop_difficulties),
    }
    return kept, report


def apply_protected_floor(
    rows: list[dict],
    current_pool: list[dict],
    *,
    config: PromptFilter,
    seed: int,
) -> tuple[list[dict], dict]:
    """Top protected types up to ``protected_floor`` of the current rows.

    Hard-train rows are not replaced. Current rows of other types are.
    """
    protected = set(config.protect_types)
    hard = [task for task in rows if task.get("split") == "hard_train"]
    current = [task for task in rows if task.get("split") != "hard_train"]
    n_current = len(current)
    target = {kind: int(config.protected_floor * n_current) for kind in config.protect_types}
    pools = {kind: _members(current_pool, kind) for kind in config.protect_types}
    missing = [kind for kind, members in pools.items() if not members and target.get(kind, 0) > 0]
    if missing:
        raise RuntimeError(f"protected types missing from the current pool: {', '.join(missing)}")
    rng = random.Random(seed)
    cursors = {kind: _cycle(members, rng) for kind, members in pools.items()}
    result = list(current)
    replaceable = [index for index, task in enumerate(result) if task.get("type") not in protected]
    rng.shuffle(replaceable)
    cursor = 0
    topped = {kind: 0 for kind in config.protect_types}
    short = {kind: 0 for kind in config.protect_types}
    for kind in config.protect_types:
        have = sum(1 for task in result if task.get("type") == kind)
        need = target[kind] - have
        if need <= 0:
            continue
        while need > 0 and cursor < len(replaceable):
            result[replaceable[cursor]] = next(cursors[kind])
            cursor += 1
            need -= 1
            topped[kind] += 1
        short[kind] = max(need, 0)
    out = hard + result
    rng.shuffle(out)
    if sum(1 for task in out if task.get("split") == "hard_train") != len(hard):
        raise RuntimeError("protected floor changed the hard-train count")
    report = {
        "target": target,
        "topped_up": topped,
        "short": short,
        "n_current": n_current,
        "n_hard": len(hard),
        "by_type": _counts(out),
    }
    return out, report


def blocked_eval_ids() -> frozenset[str]:
    """Ids that must never enter a training prompt or a resample draw.

    The private eval sheet when present, the published sheet, the known-drug holdout, and TEST.
    Memorization probes read the holdout on their own path and are not filtered here.
    """
    from fdagym.agents.baseline import TASKS_DIR, load_split

    found: set[str] = set()
    for name in ("hard", "public", "hard_train_holdout", "test"):
        if not (TASKS_DIR / f"{name}.jsonl").exists():
            continue
        found.update(task["id"] for task in load_split(name))
    return frozenset(found)


def count_by_type(tasks: list[dict]) -> dict:
    rows = _counts(tasks)
    unique = {kind: len({task["id"] for task in tasks if task.get("type") == kind}) for kind in rows}
    splits = dict(Counter(task.get("split") for task in tasks))
    return {"rows": rows, "unique": unique, "splits": splits}


def _drop(task: dict, config: PromptFilter) -> bool:
    if task.get("type") in config.protect_types:
        return False
    rate = _pass_rate(task, config.pass_rates)
    if rate is None:
        return task.get("difficulty") in config.drop_difficulties
    solved = config.drop_solved and rate >= config.solved_at
    failed = config.drop_failed and rate <= config.failed_at
    return bool(solved or failed)


def _pass_rate(task: dict, override: dict[str, float] | None) -> float | None:
    if override is not None and task.get("id") in override:
        return float(override[task["id"]])
    rate = task.get("base_pass_rate")
    if rate is None:
        return None
    return float(rate)


def _refuse_held_out(task: dict, blocked: set[str]) -> None:
    split = task.get("split")
    task_id = task.get("id")
    if split in FORBIDDEN_SPLITS or task_id in blocked:
        raise RuntimeError(
            f"difficulty filter refuses {task_id} (split {split}). "
            "It edits the current train pool only. Hard eval, TEST, holdout, and hard_train stay out."
        )


def _counts(tasks: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(task.get("type") for task in tasks).items()))


def _members(tasks: list[dict], kind: str) -> list[dict]:
    found = [task for task in tasks if task.get("type") == kind and in_training_pool(task)]
    found.sort(key=lambda task: task["id"])
    return found


def _cycle(members: list[dict], rng: random.Random):
    while True:
        batch = list(members)
        rng.shuffle(batch)
        yield from batch
