"""Question-level holdout inside the hard training drugs.

About 10% of ``hard_train`` questions, stratified by variant, become
``hard_train_holdout``. They keep the same drugs and sponsors. Those questions
never enter ``mix_hard_train`` or ``in_training_pool``.
"""

from __future__ import annotations

from collections import defaultdict

from fdagym.tasks.disjoint import ApprovalMaps, _sponsors, _surface_drugs
from fdagym.tasks.hard_solver import parse_question
from fdagym.tasks.plan import load_universe

HOLDOUT_RATE = 0.10


def holdout_quota(n: int, rate: float = HOLDOUT_RATE) -> int:
    """How many rows of a variant to hold out. At least one when the variant is non-empty."""
    if n <= 0:
        return 0
    return max(1, round(n * rate))


def carve_known_holdout(
    tasks: list[dict],
    *,
    rate: float = HOLDOUT_RATE,
    snapshot=None,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, holdout)`` copies. Selection is deterministic.

    A held-out question is accepted only when every drug it names, and every
    sponsor it cites, still appears on at least one question that stays in train.
    """
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        grouped[parse_question(task["question"])["variant"]].append(task)
    for rows in grouped.values():
        rows.sort(key=lambda task: task["id"])
    maps = None if snapshot is None else _maps(snapshot)
    universe = {drug.casefold() for drug in load_universe()}
    chosen: set[str] = set()
    by_id = {task["id"]: task for task in tasks}
    for variant in sorted(grouped):
        rows = grouped[variant]
        quota = holdout_quota(len(rows), rate)
        for task in _candidate_order(rows, quota):
            if sum(1 for task_id in chosen if parse_question(by_id[task_id]["question"])["variant"] == variant) >= quota:
                break
            if _can_hold_out(task, tasks, chosen, universe, maps):
                chosen.add(task["id"])
    train = [_with_split(task, "hard_train") for task in tasks if task["id"] not in chosen]
    holdout = [_with_split(task, "hard_train_holdout") for task in tasks if task["id"] in chosen]
    train.sort(key=lambda task: task["id"])
    holdout.sort(key=lambda task: task["id"])
    return train, holdout


def assert_question_holdout(train: list[dict], holdout: list[dict], snapshot) -> dict:
    """Holdout questions are new; their drugs and sponsors still occur in train."""
    if not holdout:
        raise RuntimeError("hard_train_holdout is empty")
    train_ids = {task["id"] for task in train}
    holdout_ids = {task["id"] for task in holdout}
    shared_ids = train_ids & holdout_ids
    if shared_ids:
        raise RuntimeError(f"holdout reuses training ids: {sorted(shared_ids)[:5]}")
    train_questions = {task["question"] for task in train}
    shared_questions = [task["id"] for task in holdout if task["question"] in train_questions]
    if shared_questions:
        raise RuntimeError(f"holdout reuses training questions: {shared_questions[:5]}")
    bad_split = [task["id"] for task in holdout if task.get("split") != "hard_train_holdout"]
    if bad_split:
        raise RuntimeError(f"holdout split is not hard_train_holdout: {bad_split[:5]}")
    maps = _maps(snapshot)
    universe = {drug.casefold() for drug in load_universe()}
    train_drugs, train_sponsors = _cover(train, universe, maps)
    hold_drugs, hold_sponsors = _cover(holdout, universe, maps)
    missing_drugs = hold_drugs - train_drugs
    missing_sponsors = hold_sponsors - train_sponsors
    if missing_drugs or missing_sponsors:
        raise RuntimeError(
            "holdout is not a known-drug slice — "
            f"drugs only in holdout: {sorted(missing_drugs)[:8]}; "
            f"sponsors only in holdout: {sorted(missing_sponsors)[:8]}"
        )
    return {
        "train": len(train),
        "holdout": len(holdout),
        "holdout_by_variant": _by_variant(holdout),
        "train_drugs": len(train_drugs),
        "holdout_drugs": len(hold_drugs),
        "train_sponsors": len(train_sponsors),
        "holdout_sponsors": len(hold_sponsors),
    }


def _maps(snapshot) -> ApprovalMaps:
    from fdagym.tasks.disjoint import approval_maps

    return approval_maps(snapshot)


def _with_split(task: dict, split: str) -> dict:
    copied = dict(task)
    copied["split"] = split
    return copied


def _candidate_order(rows: list[dict], quota: int) -> list[dict]:
    """Evenly spaced ids first, then the rest, so the holdout is not the first rows."""
    n = len(rows)
    if quota <= 0 or n == 0:
        return list(rows)
    used: set[int] = set()
    primary: list[dict] = []
    for slot in range(quota):
        index = min(n - 1, int((slot + 0.5) * n / quota))
        while index < n and index in used:
            index += 1
        if index >= n:
            break
        used.add(index)
        primary.append(rows[index])
    rest = [row for index, row in enumerate(rows) if index not in used]
    return primary + rest


def _can_hold_out(task: dict, tasks: list[dict], chosen: set[str], universe: set[str], maps) -> bool:
    trial = chosen | {task["id"]}
    staying = [row for row in tasks if row["id"] not in trial]
    drugs = _surface_drugs(task, universe)
    if drugs and not drugs <= set().union(*(_surface_drugs(row, universe) for row in staying)):
        return False
    if maps is None:
        return True
    sponsors = _sponsors(task, maps)
    if not sponsors:
        return True
    covered: set[str] = set()
    for row in staying:
        covered |= _sponsors(row, maps)
    return sponsors <= covered


def _cover(tasks: list[dict], universe: set[str], maps: ApprovalMaps) -> tuple[set[str], set[str]]:
    drugs: set[str] = set()
    sponsors: set[str] = set()
    for task in tasks:
        drugs |= _surface_drugs(task, universe)
        sponsors |= _sponsors(task, maps)
    return drugs, sponsors


def _by_variant(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        counts[parse_question(task["question"])["variant"]] += 1
    return dict(sorted(counts.items()))
