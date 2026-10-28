"""Stage a training pool from structural difficulty and optional base_pass_rate."""

import json
from functools import lru_cache
from pathlib import Path

SKIP_TYPES = frozenset({"label_open"})
# compare / sponsor_hop stay out of the default pool, including the hard-training file.
# Opt in with fdagym.train.hard_mix.mix_hard_train (60% hard_train / 40% current train).
# hard_train_holdout is the known-drug question holdout and is never a training row.
EVAL_ONLY_TYPES = frozenset({"compare", "sponsor_hop"})
EVAL_ONLY_SPLITS = frozenset({"hard", "hard_train", "hard_train_holdout", "public"})
_HOLDOUT_PATH = Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard_train_holdout.jsonl"


@lru_cache(maxsize=1)
def hard_train_holdout_ids() -> frozenset[str]:
    """Ids in the known-drug holdout. Empty when that file has not been carved yet."""
    if not _HOLDOUT_PATH.exists():
        return frozenset()
    found = []
    for line in _HOLDOUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.append(json.loads(line)["id"])
    return frozenset(found)


def in_training_pool(task: dict) -> bool:
    """Train rows only. Test, hard eval, hard_train, the question holdout, and hard-eval types stay out."""
    if task.get("id") in hard_train_holdout_ids():
        return False
    if task.get("type") in SKIP_TYPES or task.get("type") in EVAL_ONLY_TYPES:
        return False
    return task.get("split") != "test" and task.get("split") not in EVAL_ONLY_SPLITS


TIERS = ("easy", "medium", "hard")
STAGE_MIX = {
    1: {"easy": 0.6, "medium": 0.3, "hard": 0.1},
    2: {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    3: {"easy": 0.2, "medium": 0.4, "hard": 0.4},
}


def filter_pool(tasks: list[dict], stage: int) -> list[dict]:
    rows = [task for task in tasks if in_training_pool(task)]
    if not any(task.get("base_pass_rate") is not None for task in rows):
        return rows
    kept = []
    for task in rows:
        rate = task.get("base_pass_rate")
        if rate == 1:
            continue
        if stage < 3 and (rate is None or rate == 0):
            continue
        kept.append(task)
    return kept


def mix_counts(n: int, mix: dict[str, float]) -> dict[str, int]:
    raw = {tier: mix[tier] * n for tier in TIERS}
    counts = {tier: int(raw[tier]) for tier in TIERS}
    remainder = n - sum(counts.values())
    order = sorted(TIERS, key=lambda tier: (raw[tier] - counts[tier], -TIERS.index(tier)), reverse=True)
    for tier in order[:remainder]:
        counts[tier] += 1
    return counts


def stage_pool(tasks: list[dict], stage: int) -> tuple[list[dict], dict[str, int]]:
    """Unique tasks for this stage. Mix is a target; shortages fill from the next harder tier."""
    if stage not in STAGE_MIX:
        raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
    pool = filter_pool(tasks, stage)
    by_id = {task["id"]: task for task in pool}
    buckets = {tier: [] for tier in TIERS}
    for task in pool:
        buckets.get(task.get("difficulty"), buckets["medium"]).append(task["id"])
    for tier in TIERS:
        buckets[tier].sort(key=lambda tid: (-_rate(by_id[tid]), tid))
    n = len(pool)
    want = mix_counts(n, STAGE_MIX[stage])
    used: set[str] = set()
    picked: list[str] = []
    for index, tier in enumerate(TIERS):
        need = want[tier]
        take = _take(buckets, index, need, used)
        used.update(take)
        picked.extend(take)
    selected = [by_id[tid] for tid in picked]
    actual = {tier: 0 for tier in TIERS}
    for task in selected:
        actual[task.get("difficulty", "medium")] += 1
    return selected, actual


def _take(buckets: dict[str, list[str]], start: int, need: int, used: set[str]) -> list[str]:
    out = []
    for tier in TIERS[start:]:
        if len(out) >= need:
            break
        for tid in buckets[tier]:
            if tid in used or tid in out:
                continue
            out.append(tid)
            if len(out) >= need:
                break
    return out


def _rate(task: dict) -> float:
    rate = task.get("base_pass_rate")
    return -1.0 if rate is None else float(rate)


# Current-train types allowed in the GRPO prompt pool. Hard-train compare /
# sponsor_hop still enter through mix_hard_train, not this list.
# Composite is out: pass@8 on the distill adapter was 95% all-fail, and the
# hard eval is compare + sponsor_hop only. boxed_warning stays; that skill
# is on the hard eval. Relative shares match the 2026-10-19 mix with
# composite's 0.30 removed (type_mix_pool renormalizes the 0.70 remainder).
GRPO_POOL_TYPES = (
    "boxed_warning",
    "faers_top_reaction",
    "approval",
    "faers_count",
)
GRPO_POOL_TYPES_FLAG = "boxed_warning,faers_top_reaction,approval,faers_count"
TOOL_TYPE_MIX = {
    "boxed_warning": 0.30,
    "faers_top_reaction": 0.20,
    "approval": 0.15,
    "faers_count": 0.05,
}
TOOL_MIX_ROWS = 600


def parse_pool_types(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """CLI / config allowlist. ``None`` is the default GRPO pool (no composite)."""
    if value is None:
        return GRPO_POOL_TYPES
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        parts = tuple(str(item).strip() for item in value if str(item).strip())
    if not parts:
        raise ValueError("pool_types must name at least one task type")
    return parts


def filter_pool_types(
    tasks: list[dict], pool_types: str | tuple[str, ...] | list[str] | None
) -> tuple[list[dict], dict]:
    """Keep current-train rows whose type is on the GRPO allowlist."""
    from collections import Counter

    allowed = parse_pool_types(pool_types)
    kept = [task for task in tasks if task.get("type") in allowed]
    dropped = [task for task in tasks if task.get("type") not in allowed]
    report = {
        "allowlist": list(allowed),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped_by_type": dict(sorted(Counter(task.get("type") for task in dropped).items())),
        "kept_by_type": dict(sorted(Counter(task.get("type") for task in kept).items())),
    }
    if not kept:
        shown = ", ".join(allowed)
        raise RuntimeError(f"pool_types [{shown}] removed every current training prompt")
    return kept, report


def composite_flag(task: dict) -> str | None:
    """``"yes"`` or ``"no"`` from a composite gold like ``"8714; yes"``."""
    value = (task.get("gold") or {}).get("value")
    if task.get("type") != "composite" or not isinstance(value, str) or ";" not in value:
        return None
    flag = value.rsplit(";", 1)[1].strip().lower()
    return flag if flag in {"yes", "no"} else None


def type_mix_pool(
    tasks: list[dict],
    weights: dict[str, float] | None = None,
    *,
    n_rows: int = TOOL_MIX_ROWS,
    seed: int = 0,
    balance_composite: bool = True,
) -> tuple[list[dict], dict]:
    """Rows drawn by task type, then shuffled. Rows repeat when a type has fewer tasks than its share.

    Each type is cycled through a seeded shuffle, so every task of a type appears
    before any repeats. Composite rows split their share 50/50 between gold "yes"
    and gold "no" (the train split is 26 yes / 43 no; the RL model always said no).
    Types missing from ``tasks`` give their weight to the others in proportion.
    Same filters as ``filter_pool``: train split only, no ``label_open``, no hard-eval types.
    """
    import random

    weights = dict(TOOL_TYPE_MIX if weights is None else weights)
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and sum above 0")
    pool = [task for task in tasks if in_training_pool(task)]
    groups: dict[str, list[dict]] = {}
    for task in pool:
        kind = task.get("type")
        if kind not in weights or weights[kind] == 0:
            continue
        if kind == "composite" and balance_composite:
            flag = composite_flag(task)
            if flag is not None:
                kind = f"composite:{flag}"
        groups.setdefault(kind, []).append(task)
    shares: dict[str, float] = {}
    for kind, weight in weights.items():
        if weight == 0:
            continue
        if kind == "composite" and balance_composite:
            halves = [key for key in ("composite:yes", "composite:no", "composite") if groups.get(key)]
            for key in halves:
                shares[key] = weight / len(halves)
        elif groups.get(kind):
            shares[kind] = weight
    if not shares:
        raise RuntimeError("type mix pool is empty")
    total = sum(shares.values())
    raw = {kind: share / total * n_rows for kind, share in shares.items()}
    counts = {kind: int(value) for kind, value in raw.items()}
    for kind in sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)[: n_rows - sum(counts.values())]:
        counts[kind] += 1
    rng = random.Random(seed)
    rows: list[dict] = []
    for kind in sorted(counts):
        members = sorted(groups[kind], key=lambda task: task["id"])
        order: list[dict] = []
        while len(order) < counts[kind]:
            batch = list(members)
            rng.shuffle(batch)
            order.extend(batch)
        rows.extend(order[: counts[kind]])
    rng.shuffle(rows)
    unique = {kind: len(groups[kind]) for kind in counts}
    return rows, {"rows": counts, "unique": unique, "n_rows": len(rows), "seed": seed}
