"""Early stop for tool GRPO: a bad warm-up shows up as malformed calls in the first steps.

Pure Python so the CPU tests cover it. ``run_tool_grpo`` wires it into ``ToolGRPOTrainer.log``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PARSE_KEYS = ("rewards/parse_rate", "rewards/turn_parse_rate")


class ParseRateStop(RuntimeError):
    """Raised after training stops on the parse-rate rule, so the pipeline records a failure."""


@dataclass
class ParseRateGuard:
    """Stop when the mean parse rate over steps 1..``window`` is below ``min_rate``.

    Checked once, at step ``window``, on both ``rewards/parse_rate`` (episodes that
    parsed a tool call) and ``rewards/turn_parse_rate`` (turns that parsed). A run
    resumed past ``window`` has no early history and is not checked again.
    """

    window: int = 20
    min_rate: float = 0.9
    seen: dict[str, list[float]] = field(default_factory=dict)
    checked: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if not 0 <= self.min_rate <= 1:
            raise ValueError("min_rate must be in [0, 1]")

    def observe(self, step: int, logs: dict) -> str | None:
        if self.checked or step < 1:
            return self.reason
        if step <= self.window:
            for key in PARSE_KEYS:
                value = logs.get(key)
                if isinstance(value, (int, float)):
                    self.seen.setdefault(key, []).append(float(value))
        if step < self.window:
            return None
        self.checked = True
        for key in PARSE_KEYS:
            values = self.seen.get(key) or []
            if not values:
                continue
            mean = sum(values) / len(values)
            if mean < self.min_rate:
                self.reason = (
                    f"{key} mean {mean:.3f} over the first {len(values)} logged steps is below {self.min_rate:.2f}"
                )
                return self.reason
        return None
