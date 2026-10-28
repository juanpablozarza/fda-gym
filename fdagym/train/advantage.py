"""Group-relative advantage. TRL computes this inside GRPOTrainer; we keep a copy for tests."""


def advantages(rewards: list[float], *, eps: float = 1e-8) -> list[float]:
    """(R_i − mean) / std over one sibling group. All zeros when the group has no spread."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((value - mean) ** 2 for value in rewards) / n
    std = var**0.5
    if std < eps:
        return [0.0] * n
    return [(value - mean) / std for value in rewards]
