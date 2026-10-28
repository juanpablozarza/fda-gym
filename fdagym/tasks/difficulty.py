"""Layer 1 of docs/difficulty.md: the structural tier, computed from features and never hand-set."""

from typing import Literal, TypedDict

Tier = Literal["easy", "medium", "hard"]


class Features(TypedDict):
    sources: int
    min_tool_calls: int
    needs_disambiguation: bool
    needs_aggregation: bool
    needs_normalization: bool
    has_constraint: bool


def features(
    *,
    sources: int,
    min_tool_calls: int,
    needs_disambiguation: bool,
    needs_aggregation: bool,
    needs_normalization: bool,
    has_constraint: bool,
) -> Features:
    return Features(
        sources=sources,
        min_tool_calls=min_tool_calls,
        needs_disambiguation=needs_disambiguation,
        needs_aggregation=needs_aggregation,
        needs_normalization=needs_normalization,
        has_constraint=has_constraint,
    )


def tier(f: Features) -> Tier:
    """First match wins: hard, then easy, then medium."""
    if f["sources"] >= 2 or f["min_tool_calls"] >= 4 or (f["needs_aggregation"] and f["has_constraint"]):
        return "hard"
    if (
        f["sources"] == 1
        and f["min_tool_calls"] <= 2
        and not f["needs_disambiguation"]
        and not f["needs_aggregation"]
    ):
        return "easy"
    return "medium"
