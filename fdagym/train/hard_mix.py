"""Opt-in 60/40 mix of the hard training split and the current train pool.

``in_training_pool`` still rejects ``compare`` / ``sponsor_hop`` and splits ``hard``
and ``hard_train``. Default GRPO keeps loading ``data/tasks/train.jsonl`` only.
Pass the hard-training file here when a run should practice the multi-hop templates
without ever opening ``data/tasks/hard.jsonl`` as a training row.

A new hard RL run passes ``hard_train_mix=0.6``. The known-drug holdout and the
hard eval are loaded only to prove their ids are absent from that pool.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from fdagym.train.curriculum import TOOL_MIX_ROWS, hard_train_holdout_ids, in_training_pool

# Recorded RL mix: 60% hard-training tasks, 40% the current train pool.
HARD_TRAIN_MIX = {"hard": 0.60, "current": 0.40}


def mix_hard_train(
    current_tasks: list[dict],
    hard_tasks: list[dict],
    *,
    weights: dict[str, float] | None = None,
    n_rows: int = TOOL_MIX_ROWS,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Sample ``n_rows`` tasks. Rows cycle when a side has fewer tasks than its share.

    ``hard_tasks`` must already be the training split (``split == "hard_train"``).
    A row with ``split == "hard"`` or ``split == "hard_train_holdout"``, or an id from the
    known-drug holdout file, is refused even if the caller relabels the split.
    """
    weights = dict(HARD_TRAIN_MIX if weights is None else weights)
    if set(weights) != {"hard", "current"}:
        raise ValueError("hard_train mix weights need exactly the keys 'hard' and 'current'")
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and sum above 0")
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    _refuse_held_out(hard_tasks)
    hard_pool = [
        task
        for task in hard_tasks
        if task.get("split") == "hard_train" and task.get("type") in {"compare", "sponsor_hop"}
    ]
    if not hard_pool:
        raise RuntimeError("hard training pool is empty")
    hard_ids = {task["id"] for task in hard_pool}
    current_pool = [task for task in current_tasks if in_training_pool(task) and task["id"] not in hard_ids]
    if not current_pool:
        raise RuntimeError("current training pool is empty")
    total = sum(weights.values())
    raw = {key: weights[key] / total * n_rows for key in ("hard", "current")}
    counts = {key: int(raw[key]) for key in raw}
    remainder = n_rows - sum(counts.values())
    order = sorted(("hard", "current"), key=lambda key: (raw[key] - counts[key], key == "current"), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    rng = random.Random(seed)
    groups = {"hard": hard_pool, "current": current_pool}
    rows: list[dict] = []
    for key in ("current", "hard"):
        members = sorted(groups[key], key=lambda task: task["id"])
        picked: list[dict] = []
        while len(picked) < counts[key]:
            batch = list(members)
            rng.shuffle(batch)
            picked.extend(batch)
        rows.extend(picked[: counts[key]])
    rng.shuffle(rows)
    report = {
        "weights": {key: weights[key] / total for key in ("hard", "current")},
        "rows": counts,
        "unique": {"hard": len(hard_pool), "current": len(current_pool)},
        "n_rows": len(rows),
        "seed": seed,
        "hard_ids": sum(1 for task in rows if task["id"] in hard_ids),
    }
    return rows, report


def hard_mix_weights(hard_ratio: float) -> dict[str, float] | None:
    """Turn a CLI fraction into mix weights. ``0`` leaves the current type mix on.

    A new hard RL run passes ``0.6``, which is 60% ``hard_train`` and 40% current
    train. The holdout file is not a weight: ``mix_hard_train`` refuses those ids.
    """
    if isinstance(hard_ratio, bool) or not isinstance(hard_ratio, (int, float)):
        raise TypeError("hard_train_mix must be a number")
    if hard_ratio == 0:
        return None
    if hard_ratio <= 0 or hard_ratio >= 1:
        raise ValueError("hard_train_mix must be 0 (off) or a fraction strictly between 0 and 1")
    hard = float(hard_ratio)
    return {"hard": hard, "current": 1.0 - hard}


def pool_exclusion_proof(
    current_tasks: list[dict],
    hard_tasks: list[dict],
    holdout_tasks: list[dict],
    hard_eval_tasks: list[dict],
) -> dict:
    """Prove holdout and hard-eval ids are absent from the hard-RL training inputs.

    The check is on the lists that would be sampled, before ``in_training_pool``
    drops a row. A leaked id raises. Empty eval sets raise: there is nothing to
    prove against. The returned hashes cover the sorted eval ids.
    """
    holdout_ids = _task_ids(holdout_tasks, "holdout")
    hard_eval_ids = _task_ids(hard_eval_tasks, "hard eval")
    current_ids = _task_ids(current_tasks, "current train")
    hard_ids = _task_ids(hard_tasks, "hard_train")
    blocked = set(holdout_ids) | set(hard_eval_ids)
    leaked = sorted((set(current_ids) | set(hard_ids)) & blocked)
    hard_pool = {
        task["id"]
        for task in hard_tasks
        if task.get("split") == "hard_train" and task.get("type") in {"compare", "sponsor_hop"}
    }
    current_pool = {task["id"] for task in current_tasks if in_training_pool(task) and task["id"] not in hard_pool}
    pool = hard_pool | current_pool
    leaked_pool = sorted(pool & blocked)
    proof = {
        "n_holdout": len(holdout_ids),
        "n_hard_eval": len(hard_eval_ids),
        "n_hard_train": len(hard_pool),
        "n_current": len(current_pool),
        "n_pool": len(pool),
        "holdout_sha256": _id_hash(holdout_ids),
        "hard_eval_sha256": _id_hash(hard_eval_ids),
        "holdout_ids": sorted(holdout_ids),
        "hard_eval_ids": sorted(hard_eval_ids),
        "leaked": leaked or leaked_pool,
    }
    if leaked or leaked_pool:
        shown = ", ".join(proof["leaked"][:8])
        raise RuntimeError(
            f"holdout or hard-eval ids are in the training pool ({len(proof['leaked'])}: {shown}). "
            "hard_train.jsonl is the only hard file a hard RL run may sample."
        )
    return proof


def prove_training_pool(current_tasks: list[dict], hard_tasks: list[dict]) -> dict:
    """Load the known-drug holdout and the hard eval, then :func:`pool_exclusion_proof`.

    A checkout without ``hard.jsonl`` uses ``public.jsonl`` as the exclusion set.
    """
    from fdagym.agents.baseline import TASKS_DIR, load_split

    def _split(name: str) -> list[dict]:
        if not (TASKS_DIR / f"{name}.jsonl").exists():
            return []
        return load_split(name)

    return pool_exclusion_proof(
        current_tasks,
        hard_tasks,
        _split("hard_train_holdout"),
        _split("hard") or _split("public"),
    )


def format_pool_exclusion_log(proof: dict) -> str:
    """Startup lines: eval-set sizes, hashes, and the id lists those hashes cover."""
    sizes = {"n_holdout": proof["n_holdout"], "n_hard_eval": proof["n_hard_eval"]}
    brief = {
        "n_holdout": proof["n_holdout"],
        "n_hard_eval": proof["n_hard_eval"],
        "n_hard_train": proof["n_hard_train"],
        "n_current": proof["n_current"],
        "n_pool": proof["n_pool"],
        "holdout_sha256": proof["holdout_sha256"],
        "hard_eval_sha256": proof["hard_eval_sha256"],
    }
    return "\n".join(
        [
            "hard-rl eval set sizes " + json.dumps(sizes),
            "hard-rl pool exclusion " + json.dumps(brief),
            "hard-rl holdout ids " + ",".join(proof["holdout_ids"]),
            "hard-rl hard-eval ids " + ",".join(proof["hard_eval_ids"]),
        ]
    )


def _task_ids(tasks: list[dict], label: str) -> list[str]:
    if not tasks:
        raise RuntimeError(f"{label} is empty; refusing to start a hard RL run without that set")
    ids = []
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"{label} has a row without an id")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{label} has duplicate ids")
    return ids


def _id_hash(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


_TASK_DIR = Path(__file__).resolve().parents[2] / "data" / "tasks"


def file_ids(name: str) -> frozenset[str]:
    """Ids in ``data/tasks/<name>``. Empty when the file is missing."""
    path = _TASK_DIR / name
    if not path.exists():
        return frozenset()
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.append(json.loads(line)["id"])
    return frozenset(found)


def rollout_tasks(current_tasks: list[dict], hard_tasks: list[dict] | None) -> list[dict]:
    """Tasks ``FDAEnv`` may reset during a mixed GRPO run.

    Prompt sampling can pick ``hard_train`` ids. The trainer still has to own
    those rows: ``rollout_batch`` and ``by_id[task_id]`` look them up here, not
    in the prompt list. ``hard_tasks`` None (the default mix) returns ``current``
    unchanged. Holdout and hard-eval rows are refused.
    """
    if not hard_tasks:
        return list(current_tasks)
    prove_training_pool(current_tasks, hard_tasks)
    _refuse_held_out(hard_tasks)
    by_id = {task["id"]: task for task in current_tasks}
    for task in hard_tasks:
        if task.get("split") != "hard_train":
            continue
        by_id.setdefault(task["id"], task)
    return list(by_id.values())


def _refuse_held_out(hard_tasks: list[dict]) -> None:
    blocked = hard_train_holdout_ids()
    leaked = [
        task
        for task in hard_tasks
        if task.get("split") in {"hard", "hard_train_holdout"} or task.get("id") in blocked
    ]
    if not leaked:
        return
    shown = ", ".join(str(task.get("id")) for task in leaked[:5])
    splits = sorted({str(task.get("split")) for task in leaked})
    raise RuntimeError(
        f"refusing eval split {splits} in the hard-training mix ({len(leaked)} rows, e.g. {shown}). "
        "Load data/tasks/hard_train.jsonl. hard.jsonl and hard_train_holdout.jsonl stay out."
    )
