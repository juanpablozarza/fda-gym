"""Reject golds a reward function could not score: empty, tied, or multi-valued."""

from collections.abc import Iterable
from typing import Any


class TaskRejected(ValueError):
    """The snapshot does not determine one gold answer for this template."""


class OracleMismatch(ValueError):
    """oracle_calls is not the evidence-retrieving sequence difficulty_features claims."""


_ORACLE_TOOLS = frozenset(
    {"search_drug_label", "get_label_section", "query_adverse_events", "search_approvals"}
)


def require_oracle_calls(task: dict) -> None:
    """oracle_calls is the oracle's tool sequence, so its length is min_tool_calls."""
    calls = task.get("oracle_calls")
    minimum = int(task["difficulty_features"]["min_tool_calls"])
    if not isinstance(calls, list) or len(calls) != minimum:
        got = len(calls) if isinstance(calls, list) else None
        raise OracleMismatch(f"oracle_calls length {got} != min_tool_calls {minimum}")
    for call in calls:
        if not isinstance(call, dict) or call.get("tool") not in _ORACLE_TOOLS:
            raise OracleMismatch(f"oracle tool {call!r} is not an evidence tool")
        if not isinstance(call.get("arguments"), dict):
            raise OracleMismatch("oracle arguments must be an object")


def require_nonempty(value: Any, what: str) -> Any:
    if value is None or value == "" or value == [] or value == {}:
        raise TaskRejected(f"empty {what}")
    return value


def require_single(values: Iterable[Any], what: str) -> Any:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if not unique:
        raise TaskRejected(f"empty {what}")
    if len(unique) > 1:
        raise TaskRejected(f"multi-value {what}")
    return unique[0]


def unique_top_term(results: list[dict]) -> str:
    """The top FAERS reaction, or a rejection when the list is empty or tied."""
    if not results:
        raise TaskRejected("empty faers reactions")
    top = results[0]
    term = str(top.get("term") or "").strip()
    require_nonempty(term, "faers term")
    if len(results) > 1 and top.get("count") == results[1].get("count"):
        raise TaskRejected("tied faers top reaction")
    return term
