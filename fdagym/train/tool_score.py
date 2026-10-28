"""Shaped training reward and the TRL 0.21 group advantage.

``trajectory.reward.total`` stays the env score. The schema caps that field at 1,
so the parse bonus lives only on the float GRPO normalizes.
"""

from __future__ import annotations


def score_dense(trajectory: dict, task: dict, snapshot) -> dict:
    """Partial citation and near-miss correctness. The caller passes ``dense=True`` here."""
    from fdagym.rewards import score

    return score(
        {"steps": trajectory.get("steps") or [], "final_answer": trajectory.get("final_answer")},
        task,
        dense=True,
        snapshot=snapshot,
    )


def hop_reward(trajectory: dict, task: dict, snapshot) -> float:
    """Hop-rate for logging. Credit already lives inside dense ``score()``.

    Do not add this onto the shaped total; that would double-count. Strict
    ``FDAEnv`` score and ``WEIGHTS`` stay put. Compare / sponsor_hop only.
    """
    if (task or {}).get("type") not in {"compare", "sponsor_hop"}:
        return 0.0
    return float(score_dense(trajectory, task, snapshot).get("hops") or 0.0)


def episode_scores(rollout, task, snapshot, *, dense: bool, bonus: float) -> tuple[float, float, float, float]:
    """Strict env total, dense/base, shaped training reward, and hop_rate.

    ``dense=True`` calls ``score(..., dense=True, snapshot=snapshot)``. ``FDAEnv``
    still stores the strict total on the trajectory. The bonus is not written there.

    Hop-by-hop credit is folded into dense correctness. ``hop_reward`` is not
    called here and is not an extra term beyond the parse bonus.
    """
    strict = float(rollout.reward["total"])
    hop_rate = 0.0
    if dense:
        dense_row = score_dense(rollout.trajectory, task, snapshot)
        base = float(dense_row["total"])
        hop_rate = float(dense_row.get("hops") or 0.0)
    else:
        base = strict
    return strict, base, training_reward(base, rollout.parsed_tool_calls, bonus), hop_rate


def training_reward(env_total: float, parsed_tool_calls: int, bonus: float = 0.05) -> float:
    """Add ``bonus`` once when any non-submit tool call parsed.

    A per-call bonus would stack (six parses is 0.30) and swamp efficiency.
    ``bonus`` is the whole extra, not a rate. Pass 0 to disable shaping.
    """
    if bonus < 0:
        raise ValueError("tool_parse_bonus must be >= 0")
    if parsed_tool_calls < 0:
        raise ValueError("parsed_tool_calls must be >= 0")
    total = float(env_total)
    if parsed_tool_calls > 0 and bonus > 0:
        total += float(bonus)
    return total


def trl_group_advantages(rewards: list[float], group_size: int, *, scale: bool = True) -> list[float]:
    """Match TRL 0.21: unbiased std, then ``(r - mean) / (std + 1e-4)`` when ``scale`` is on.

    ``torch.std`` divides by ``n - 1``. A group of one is undefined there, and GRPO
    always groups at least two completions. All-equal rewards stay exact zeros.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if len(rewards) % group_size != 0:
        raise ValueError("rewards length must be a multiple of group_size")
    out: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean = sum(group) / group_size
        var = sum((value - mean) ** 2 for value in group) / (group_size - 1)
        std = var**0.5
        for value in group:
            advantage = value - mean
            if scale:
                advantage = advantage / (std + 1e-4)
            out.append(advantage)
    return out


def pad_rows(rows: list[list[int]], *, pad_value: int, side: str) -> tuple[list[list[int]], list[list[int]]]:
    """Pad ragged rows. The second return is 1 on real positions and 0 on the pad."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if not rows:
        return [], []
    width = max(len(row) for row in rows)
    padded: list[list[int]] = []
    mask: list[list[int]] = []
    for row in rows:
        gap = width - len(row)
        pads = [pad_value] * gap
        zeros = [0] * gap
        ones = [1] * len(row)
        if side == "left":
            padded.append(pads + list(row))
            mask.append(zeros + ones)
        else:
            padded.append(list(row) + pads)
            mask.append(ones + zeros)
    return padded, mask


TERMINATIONS = ("submit", "error", "max_steps")


def frac_zero_std(rewards: list[float], group_size: int, *, tol: float = 1e-9) -> float:
    """Share of GRPO groups whose rewards are all equal (advantage 0, nothing to learn)."""
    if group_size < 1 or not rewards or len(rewards) % group_size != 0:
        raise ValueError("rewards length must be a positive multiple of group_size")
    groups = [rewards[start : start + group_size] for start in range(0, len(rewards), group_size)]
    tied = sum(1 for group in groups if max(group) - min(group) <= tol)
    return tied / len(groups)


def episode_metrics(rollouts) -> dict[str, float]:
    """Per-batch tool metrics copied onto the trainer log, which W&B records.

    ``rewards/parse_rate``: episodes with at least one parsed non-submit tool call.
    ``rewards/turn_parse_rate``: model turns that parsed into an action, submit included.
    ``episodes/terminated_*``: how the model ended the episode (env ``terminated_by``).
    ``episodes/closed_*``: the runner closed it (turn limit or context). Those are not
    counted under ``terminated_error`` even though the env records them as an error.
    """
    n = len(rollouts)
    if n == 0:
        return {}
    turns = sum(item.n_turns for item in rollouts)
    parsed = sum(getattr(item, "parsed_turns", 0) for item in rollouts)
    out = {
        "rewards/parse_rate": sum(1 for item in rollouts if item.parsed_tool_calls > 0) / n,
        "rewards/turn_parse_rate": parsed / turns if turns else 0.0,
        "episodes/tool_calls_mean": sum(item.parsed_tool_calls for item in rollouts) / n,
        "episodes/closed_turn_limit": sum(1 for item in rollouts if getattr(item, "closed_by", None) == "turn_limit") / n,
        "episodes/closed_context": sum(1 for item in rollouts if getattr(item, "closed_by", None) == "context") / n,
    }
    for reason in TERMINATIONS:
        out[f"episodes/terminated_{reason}"] = (
            sum(1 for item in rollouts if item.terminated_by == reason and getattr(item, "closed_by", None) is None) / n
        )
    return out
