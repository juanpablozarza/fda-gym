"""Zero-variance GRPO groups, and how many questions share an optimizer step.

TRL 0.21 does not sample again when a group's rewards are tied. That option was
proposed later (huggingface/trl#3758) and is not in this pin. Tool GRPO already
owns ``_generate_and_score_completions``, so the resample loop lives here and
the trainer calls it.

Four questions share an optimizer step by gradient accumulation, not by a wider
forward. Each micro-batch stays one group of ``num_generations`` sequences.
That is the forward that fits on one A100-80GB after log-prob chunking: the
unchunked ``[8, ~4k, 151936]`` fp32 window is about 17 GiB, and a chunk of 256
token rows is about 156 MiB. A 32-sequence forward would scale the checkpointed
hidden states by four and is not what this config does. ``generation_batch_size``
stays equal to ``num_generations``, so TRL's ``steps_per_generation`` stays 1
and vLLM's ``max_num_seqs`` stays 8.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Summed across the micro-batches of one optimizer step before W&B sees them.
# Fractions (``frac_reward_zero_std``, ``groups/initial_frac_zero_std``) stay means.
GROUP_COUNT_METRICS = (
    "groups/dropped",
    "groups/resampled",
    "groups/resample_rounds",
    "groups/masked",
    "groups/trained",
)

HARD_PROMPTS_PER_STEP = 4
HARD_MAX_RESAMPLE_ATTEMPTS = 3


def optimizer_batch(num_generations: int, prompts_per_step: int) -> dict:
    """One group per micro-batch, ``prompts_per_step`` micro-batches per optimizer step.

    ``generation_batch_size`` is the value ``TrainConfig.grpo_kwargs`` already
    writes (``num_generations``). Leaving it there keeps TRL from gathering
    every prompt in the optimizer step into one rollout.
    """
    if isinstance(prompts_per_step, bool) or not isinstance(prompts_per_step, int) or prompts_per_step < 1:
        raise ValueError("prompts_per_step must be an integer >= 1")
    if isinstance(num_generations, bool) or not isinstance(num_generations, int) or num_generations < 2:
        raise ValueError("num_generations must be an integer >= 2")
    return {
        "num_generations": num_generations,
        "prompts_per_step": prompts_per_step,
        "per_device_train_batch_size": num_generations,
        "gradient_accumulation_steps": prompts_per_step,
        "generation_batch_size": num_generations,
    }


def resolve_hard_signal(
    hard: bool,
    *,
    prompts_per_step: int | None,
    dynamic_sampling: bool | None,
    difficulty_filter: bool | None,
    max_resample_attempts: int | None,
) -> dict:
    """Defaults for a hard-mix run. ``None`` means "on for the hard mix, off otherwise".

    An unflagged tool run (``hard`` false and every override ``None``) stays one
    question per optimizer step, with no resampling and no prompt filter.
    """
    if prompts_per_step is not None and (
        isinstance(prompts_per_step, bool) or not isinstance(prompts_per_step, int) or prompts_per_step < 1
    ):
        raise ValueError("prompts_per_step must be an integer >= 1")
    if max_resample_attempts is not None and (
        isinstance(max_resample_attempts, bool)
        or not isinstance(max_resample_attempts, int)
        or max_resample_attempts < 0
    ):
        raise ValueError("max_resample_attempts must be an integer >= 0")
    return {
        "prompts_per_step": HARD_PROMPTS_PER_STEP if prompts_per_step is None and hard else (prompts_per_step or 1),
        "dynamic_sampling": hard if dynamic_sampling is None else bool(dynamic_sampling),
        "difficulty_filter": hard if difficulty_filter is None else bool(difficulty_filter),
        "max_resample_attempts": (
            HARD_MAX_RESAMPLE_ATTEMPTS if max_resample_attempts is None else max_resample_attempts
        ),
    }


def tied_group_flags(rewards: list[float], group_size: int, *, tol: float = 1e-9) -> list[bool]:
    """One flag per group. True when every reward in the group is the same."""
    if group_size < 1 or not rewards or len(rewards) % group_size != 0:
        raise ValueError("rewards length must be a positive multiple of group_size")
    flags = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        flags.append(max(group) - min(group) <= tol)
    return flags


def grouped_prompt_ids(task_ids: list[str], group_size: int) -> list[str]:
    """The prompt id of each contiguous group. TRL repeats one id ``group_size`` times."""
    if group_size < 1 or len(task_ids) % group_size != 0:
        raise ValueError("task id length must be a positive multiple of group_size")
    groups = []
    for start in range(0, len(task_ids), group_size):
        chunk = task_ids[start : start + group_size]
        if len(set(chunk)) != 1:
            shown = ", ".join(chunk[:4])
            raise ValueError(f"GRPO group is not one prompt ({shown})")
        groups.append(chunk[0])
    return groups


def expand_groups(group_ids: list[str], group_size: int) -> list[str]:
    return [task_id for task_id in group_ids for _ in range(group_size)]


def example_mask_for(tied: list[bool], group_size: int, *, enabled: bool) -> list[int]:
    """1 on rows that enter the loss. A still-tied group is 0 only when sampling is on."""
    mask = []
    for flag in tied:
        bit = 0 if enabled and flag else 1
        mask.extend([bit] * group_size)
    return mask


def weighted_mean(values: list[float], weights: list[float]) -> float:
    """Mean of ``values`` over positive weights. All-zero weights are 0.

    This is the reduction ``_compute_loss`` applies to per-sequence GRPO losses
    so a masked group does not shrink the other groups in the same micro-batch.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total


class PromptPool:
    """Training-prompt ids a resample may draw. Eval, holdout, and test ids are refused."""

    def __init__(self, task_ids: list[str], blocked: set[str] | frozenset[str]):
        if not task_ids:
            raise RuntimeError("resample pool is empty")
        leaked = sorted(set(task_ids) & set(blocked))
        if leaked:
            shown = ", ".join(leaked[:8])
            raise RuntimeError(
                f"resample pool includes {len(leaked)} eval, holdout, or test ids ({shown}). "
                "Dynamic sampling draws only from the training prompts."
            )
        self.ids = list(task_ids)

    def draw(self, rng: random.Random, *, avoid: set[str]) -> str:
        choices = [task_id for task_id in self.ids if task_id not in avoid]
        if not choices:
            choices = list(self.ids)
        return rng.choice(choices)


@dataclass(frozen=True)
class FillResult:
    payloads: list
    rewards: list[float]
    example_mask: list[int]
    task_ids: list[str]
    stats: dict


def fill_zero_std_groups(
    task_ids: list[str],
    group_size: int,
    rollout_fn,
    pool: PromptPool | None,
    *,
    enabled: bool,
    max_attempts: int,
    rng: random.Random,
    tol: float = 1e-9,
) -> FillResult:
    """Drop tied groups and roll replacements, up to ``max_attempts`` extra rounds.

    ``rollout_fn(ids, round_index)`` returns ``(payloads, rewards)`` aligned with
    ``ids``. Round 0 is the dataloader batch. Later rounds roll only the groups
    that were tied. A group that is still tied after the cap is masked out of
    the loss (``example_mask`` 0) so it is not a training example.

    ``stats["dropped"]`` is how many groups in the original batch were tied
    (those draws are not trained). ``stats["resampled"]`` is how many replacement
    groups were rolled. ``stats["masked"]`` is how many groups in the returned
    batch are still tied and excluded.
    """
    if enabled and pool is None:
        raise RuntimeError("dynamic sampling needs a training prompt pool")
    if max_attempts < 0:
        raise ValueError("max_attempts must be >= 0")
    grouped_prompt_ids(task_ids, group_size)
    payloads, rewards = rollout_fn(list(task_ids), 0)
    _check_aligned(task_ids, payloads, rewards)
    tied = tied_group_flags(rewards, group_size, tol=tol)
    initial = sum(tied)
    resampled = 0
    rounds = 0
    if enabled:
        while any(tied) and rounds < max_attempts:
            rounds += 1
            groups = grouped_prompt_ids(task_ids, group_size)
            avoid = {groups[index] for index, flag in enumerate(tied) if flag}
            new_groups = list(groups)
            n_tied = 0
            for index, flag in enumerate(tied):
                if not flag:
                    continue
                new_groups[index] = pool.draw(rng, avoid=avoid)
                avoid.add(new_groups[index])
                n_tied += 1
            resampled += n_tied
            task_ids = expand_groups(new_groups, group_size)
            reroll = []
            for index, flag in enumerate(tied):
                if flag:
                    reroll.extend(range(index * group_size, (index + 1) * group_size))
            subset = [task_ids[slot] for slot in reroll]
            sub_payloads, sub_rewards = rollout_fn(subset, rounds)
            _check_aligned(subset, sub_payloads, sub_rewards)
            for slot, payload, reward in zip(reroll, sub_payloads, sub_rewards):
                payloads[slot] = payload
                rewards[slot] = reward
            tied = tied_group_flags(rewards, group_size, tol=tol)
    masked = sum(tied) if enabled else 0
    n_groups = len(task_ids) // group_size
    stats = {
        "initial_zero_std": initial,
        "dropped": initial if enabled else 0,
        "resampled": resampled,
        "resample_rounds": rounds,
        "masked": masked,
        "trained": n_groups - masked,
        "n_groups": n_groups,
    }
    return FillResult(
        payloads=payloads,
        rewards=rewards,
        example_mask=example_mask_for(tied, group_size, enabled=enabled),
        task_ids=task_ids,
        stats=stats,
    )


def collapse_group_counts(metrics: dict) -> None:
    """Replace per-micro-batch counts with one sum, so a W&B step is the optimizer step.

    Hugging Face logs once per optimizer step and averages the list. Four
    questions append four counts; averaging would report a mean per question.
    A one-element list whose value is the sum survives that average.
    """
    for key in GROUP_COUNT_METRICS:
        values = metrics.get(key)
        if not values:
            continue
        metrics[key] = [float(sum(values))]


def _check_aligned(task_ids, payloads, rewards) -> None:
    if len(payloads) != len(task_ids) or len(rewards) != len(task_ids):
        raise RuntimeError(
            f"rollout returned {len(payloads)} payloads and {len(rewards)} rewards for {len(task_ids)} prompts"
        )
