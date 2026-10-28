import pytest

from fdagym.tasks.difficulty import features, tier


def _f(**overrides):
    base = {
        "sources": 1,
        "min_tool_calls": 1,
        "needs_disambiguation": False,
        "needs_aggregation": False,
        "needs_normalization": False,
        "has_constraint": False,
    }
    base.update(overrides)
    return features(**base)


def test_easy_is_one_source_two_calls_and_no_extra_demands():
    assert tier(_f(min_tool_calls=2)) == "easy"


def test_disambiguation_is_medium_not_easy():
    assert tier(_f(needs_disambiguation=True)) == "medium"


def test_aggregation_alone_is_medium():
    assert tier(_f(needs_aggregation=True)) == "medium"


def test_two_sources_is_hard_even_when_everything_else_is_easy():
    assert tier(_f(sources=2)) == "hard"


def test_four_tool_calls_is_hard():
    assert tier(_f(min_tool_calls=4)) == "hard"


def test_aggregation_plus_constraint_is_hard():
    assert tier(_f(needs_aggregation=True, has_constraint=True)) == "hard"


def test_constraint_alone_stays_easy():
    assert tier(_f(has_constraint=True)) == "easy"


@pytest.mark.parametrize(
    ("sources", "calls"),
    [(3, 6), (2, 1), (1, 4)],
)
def test_hard_rule_wins_before_the_easy_rule(sources, calls):
    assert tier(_f(sources=sources, min_tool_calls=calls)) == "hard"
