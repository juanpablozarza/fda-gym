import pytest

from fdagym.tasks.validate import (
    OracleMismatch,
    TaskRejected,
    require_nonempty,
    require_oracle_calls,
    require_single,
    unique_top_term,
)


def test_unique_top_term_returns_the_leader():
    assert unique_top_term([{"term": "NAUSEA", "count": 10}, {"term": "FATIGUE", "count": 4}]) == "NAUSEA"


def test_tied_top_reaction_is_rejected():
    with pytest.raises(TaskRejected, match="tied"):
        unique_top_term([{"term": "NAUSEA", "count": 5}, {"term": "FATIGUE", "count": 5}])


def test_empty_reaction_list_is_rejected():
    with pytest.raises(TaskRejected, match="empty"):
        unique_top_term([])


def test_blank_gold_is_rejected():
    with pytest.raises(TaskRejected, match="empty"):
        require_nonempty("  ".strip(), "gold")


def test_multi_value_gold_is_rejected():
    with pytest.raises(TaskRejected, match="multi-value"):
        require_single(["yes", "no"], "boxed warning")


def test_single_value_passes():
    assert require_single(["2017", "2017"], "year") == "2017"


def test_oracle_calls_must_match_min_tool_calls():
    task = {
        "oracle_calls": [{"tool": "search_approvals", "arguments": {"generic_name": "warfarin"}}],
        "difficulty_features": {"min_tool_calls": 2},
    }
    with pytest.raises(OracleMismatch, match="min_tool_calls"):
        require_oracle_calls(task)
    task["difficulty_features"]["min_tool_calls"] = 1
    require_oracle_calls(task)
