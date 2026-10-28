"""Oracle-trajectory rows for the SFT warm-up.

Rows are built by playing the scripted trajectory through ``rollout_batch``, so the
token ids and the loss mask are the same objects GRPO trains on. Boxed-warning and
composite label steps are answer-independent (see ``boxed_oracle``). This module
does not import torch.
"""

from __future__ import annotations

import random

from fdagym.snapshot import Snapshot
from fdagym.train.rollout import QwenToolCodec, Rollout, RolloutLimits, rollout_batch
from fdagym.train.rollout_dry import OracleGenerator

IGNORE_INDEX = -100
# Easy "yes" boxed tasks read the section the stored oracle skipped, so efficiency
# is 0.5 and the total is 0.975. Keep those. Drop anything the citation cannot carry.
MIN_SFT_REWARD = 0.97
VAL_DRUG_FRACTION = 0.05


def sft_example(rollout: Rollout) -> dict:
    """Prompt tokens and masked spans are ``-100``. Model tokens are the label."""
    labels = [IGNORE_INDEX] * len(rollout.prompt_ids)
    for token_id, bit in zip(rollout.completion_ids, rollout.loss_mask, strict=True):
        labels.append(int(token_id) if bit else IGNORE_INDEX)
    input_ids = list(rollout.prompt_ids) + list(rollout.completion_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "task_id": rollout.task_id,
    }


def split_sft_by_drug(
    tasks: list[dict],
    *,
    fraction: float = VAL_DRUG_FRACTION,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Hold out ``fraction`` of the drugs. Every task for a drug stays on one side.

    The draw is ``random.Random(seed).sample`` over the sorted generic names, so the
    same pool and seed always hold out the same drugs.
    """
    if fraction < 0 or fraction >= 1:
        raise ValueError("fraction must be in [0, 1)")
    drugs = sorted({(task.get("entity") or {}).get("generic_name") for task in tasks})
    drugs = [drug for drug in drugs if isinstance(drug, str) and drug]
    holdout = round(fraction * len(drugs))
    held = set(random.Random(seed).sample(drugs, holdout)) if holdout else set()
    train: list[dict] = []
    val: list[dict] = []
    for task in tasks:
        drug = (task.get("entity") or {}).get("generic_name")
        (val if drug in held else train).append(task)
    return train, val


def teacher_forced_tool_match(preds: list[int], labels: list[int]) -> dict[str, bool]:
    """Whether argmax tokens match the supervised spans. ``preds[i]`` predicts ``labels[i + 1]``.

    This is the eval forward pass, not a generation. ``first_tool_exact`` is the first
    supervised span (the first tool call, conditioned on the prompt and the gold prefix
    of that call). ``tool_seq_exact`` is every supervised span.
    """
    spans = _supervised_spans(labels)
    if not spans:
        return {"first_tool_exact": False, "tool_seq_exact": False}

    def matches(start: int, end: int) -> bool:
        for index in range(start, end):
            pred_at = index - 1
            if pred_at < 0 or pred_at >= len(preds) or preds[pred_at] != labels[index]:
                return False
        return True

    return {
        "first_tool_exact": matches(*spans[0]),
        "tool_seq_exact": all(matches(*span) for span in spans),
    }


def _supervised_spans(labels: list[int]) -> list[tuple[int, int]]:
    spans = []
    index = 0
    while index < len(labels):
        if labels[index] == IGNORE_INDEX:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] != IGNORE_INDEX:
            end += 1
        spans.append((index, end))
        index = end
    return spans


def build_sft_rows(
    snapshot: Snapshot,
    tasks: list[dict],
    task_ids: list[str],
    tokenizer,
    codec=None,
    limits: RolloutLimits | None = None,
    *,
    max_length: int = 8192,
    min_reward: float = MIN_SFT_REWARD,
    run_id: str = "sft",
) -> dict:
    """Keep episodes that score at least ``min_reward`` and fit ``max_length``."""
    codec = codec or QwenToolCodec()
    limits = limits or RolloutLimits(temperature=0.0)
    by_id = {task["id"]: task for task in tasks}
    chosen = [by_id[task_id] for task_id in task_ids]
    generator = OracleGenerator(tokenizer, codec, snapshot, chosen)
    skipped = [
        {"task_id": task_id, "reason": "script", "detail": detail} for task_id, detail in generator.failures.items()
    ]
    ok_ids = [task_id for task_id in task_ids if task_id not in generator.failures]
    rollouts = (
        rollout_batch(
            snapshot,
            tasks,
            ok_ids,
            tokenizer,
            generator,
            codec,
            limits,
            run_id=run_id,
            seed=0,
        )
        if ok_ids
        else []
    )
    rows = []
    kept_rollouts = []
    for rollout in rollouts:
        length = len(rollout.prompt_ids) + len(rollout.completion_ids)
        if rollout.reward["total"] + 1e-9 < min_reward:
            skipped.append(
                {"task_id": rollout.task_id, "reason": "reward", "total": rollout.reward["total"], "length": length}
            )
            continue
        if length > max_length:
            skipped.append(
                {"task_id": rollout.task_id, "reason": "length", "total": rollout.reward["total"], "length": length}
            )
            continue
        rows.append(sft_example(rollout))
        kept_rollouts.append(rollout)
    return {
        "rows": rows,
        "skipped": skipped,
        "rollouts": kept_rollouts,
        "n_kept": len(rows),
        "n_skipped": len(skipped),
    }
