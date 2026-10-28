"""Efficiency reward. Counted only when correctness is already above 0.

A call is the pair (tool, arguments). Repeating that pair does not move the count.
"""

import json


def score_efficiency(steps: list[dict], task: dict, correctness: float) -> tuple[float, list[str]]:
    calls = _distinct_calls(steps)
    minimum = int(task["difficulty_features"]["min_tool_calls"])
    if correctness <= 0:
        return 0.0, [f"efficiency not counted: correctness is 0 ({calls} distinct tool calls)"]
    if calls < minimum:
        score = calls / minimum
        return score, [f"efficiency {score:.2f}: {calls} distinct tool calls under min_tool_calls {minimum}"]
    score = min(1.0, minimum / max(calls, 1))
    return score, [f"efficiency {score:.2f}: min_tool_calls {minimum} over {calls} distinct tool calls"]


def _distinct_calls(steps: list[dict]) -> int:
    seen: set[tuple[str, str]] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        if not isinstance(tool, str) or tool == "submit_answer":
            continue
        arguments = step.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        seen.add((tool, json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)))
    return len(seen)
