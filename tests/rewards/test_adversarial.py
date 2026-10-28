"""Attacks from learn/00-orientation.md §5b, plus the gates those attacks need."""

import json

import pytest

from fdagym.env import FDAEnv
from fdagym.rewards import score
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, WARFARIN_SET, JsonSnapshot, cite, drive, submit

QUOTE = "Warfarin sodium can cause major or fatal bleeding"


def _expect(reward: dict, total: float, **parts: float) -> None:
    assert reward["total"] == pytest.approx(total, abs=1e-9), reward
    for key, value in parts.items():
        assert reward[key] == pytest.approx(value, abs=1e-9), reward


def test_quitter_scores_format_only():
    _obs, reward, _env = drive("FG-0001", [submit("unknown", [])])
    _expect(reward, 0.10, correctness=0, citation=0, format=1, efficiency=0)


def test_irrelevant_field_spammer_scores_format_only():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    search = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    obs, _reward, done, _info = env.step(search)
    assert done is False
    rows = obs["output"]["results"]
    citations = [
        cite("drug/label", row["set_id"], "effective_time", str(row["effective_time"])) for row in rows
    ]
    citations.append(cite("drug/label", rows[0]["set_id"], "brand", str(rows[0]["brand"])))
    assert len(citations) == 6
    for _ in range(5):
        env.step(search)
    _obs, _reward, done, info = env.step(submit("No", citations))
    assert done is True
    assert env.tool_calls == 6
    _expect(info["reward"], 0.10, correctness=0, citation=0, format=1, efficiency=0)


def test_honest_searcher_with_the_right_evidence_and_the_wrong_answer():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    search = {
        "tool": "search_drug_label",
        "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"},
    }
    section = {"tool": "get_label_section", "arguments": {"set_id": WARFARIN_SET, "section": "boxed_warning"}}
    env.step(search)
    obs, _reward, _done, _info = env.step(section)
    quote = obs["output"]["boxed_warning"][:60]
    env.step(search)
    env.step(section)
    _obs, _reward, done, info = env.step(
        submit("No. The label discusses a different risk.", [cite("drug/label", WARFARIN_SET, "boxed_warning", quote)])
    )
    assert done is True
    assert env.tool_calls == 4
    _expect(info["reward"], 0.35, correctness=0, citation=1, format=1, efficiency=0)


def test_broken_json_scores_zero():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    search = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    for _ in range(3):
        env.step(search)
    _obs, reward, done, _info = env.step("{not json")
    assert done is True
    assert reward == 0.0
    assert env.trajectory["terminated_by"] == "error"
    assert env.trajectory["final_answer"] is None
    _expect(env.trajectory["reward"], 0.0, correctness=0, citation=0, format=0, efficiency=0)


def test_always_nausea_without_tools():
    _obs, reward, _env = drive("FG-0002", [submit("NAUSEA", [])])
    _expect(reward, 0.10, correctness=0, citation=0, format=1, efficiency=0)
    assert any("gated" in note for note in reward["notes"])


def test_memorized_set_id_never_retrieved_in_this_trajectory():
    _obs, reward, _env = drive(
        "FG-0001",
        [
            submit(
                "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
                [cite("drug/label", WARFARIN_SET, "boxed_warning", QUOTE)],
            )
        ],
    )
    _expect(reward, 0.10, correctness=0, citation=0, format=1, efficiency=0)
    assert any("did not appear" in note for note in reward["notes"])


def test_fabricated_quote_does_not_ground_a_correct_answer():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    env.step({"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}})
    env.step({"tool": "get_label_section", "arguments": {"set_id": WARFARIN_SET, "section": "boxed_warning"}})
    _obs, _reward, _done, info = env.step(
        submit(
            "Yes. bleeding",
            [cite("drug/label", WARFARIN_SET, "boxed_warning", "This sentence was not in the label at all.")],
        )
    )
    _expect(info["reward"], 0.10, correctness=0, citation=0, format=1, efficiency=0)
    assert any("quote fuzzy" in note for note in info["reward"]["notes"])


def test_answer_listing_alternatives_does_not_match():
    _obs, reward, _env = drive(
        "FG-0002",
        [
            {"tool": "query_adverse_events", "arguments": {"generic_name": "metformin", "mode": "top_reactions"}},
            submit("NAUSEA or DIARRHOEA", [cite("drug/event", "step:1", "results", "NAUSEA")]),
        ],
    )
    _expect(reward, 0.35, correctness=0, citation=1, format=1, efficiency=0)
    assert any("alternatives" in note for note in reward["notes"])


def test_abstain_on_an_answerable_task():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    env.step({"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}})
    obs, _reward, _done, _info = env.step(
        {"tool": "get_label_section", "arguments": {"set_id": WARFARIN_SET, "section": "boxed_warning"}}
    )
    quote = obs["output"]["boxed_warning"][:40]
    _obs, _reward, _done, info = env.step(submit("NOT_FOUND", [cite("drug/label", WARFARIN_SET, "boxed_warning", quote)]))
    _expect(info["reward"], 0.35, correctness=0, citation=1, format=1, efficiency=0)
    assert any("answerable" in note for note in info["reward"]["notes"])


def test_four_identical_searches_count_as_one_call():
    from fdagym.rewards.efficiency import score_efficiency

    step = {"tool": "search_drug_label", "arguments": {"generic_name": "diazepam", "has_section": "boxed_warning"}}
    task = {"difficulty_features": {"min_tool_calls": 4}}
    score, notes = score_efficiency([step, step, step, step], task, 1.0)
    assert score == 0.25
    assert notes == ["efficiency 0.25: 1 distinct tool calls under min_tool_calls 4"]
    distinct, distinct_notes = score_efficiency(
        [
            step,
            {"tool": "search_drug_label", "arguments": {"generic_name": "diazepam"}},
            {"tool": "search_approvals", "arguments": {"generic_name": "diazepam"}},
            {"tool": "query_adverse_events", "arguments": {"generic_name": "diazepam", "mode": "top_reactions"}},
        ],
        task,
        1.0,
    )
    assert distinct == 1.0
    assert distinct_notes == ["efficiency 1.00: min_tool_calls 4 over 4 distinct tool calls"]


def test_efficiency_not_counted_when_the_answer_is_wrong():
    _obs, reward, env = drive(
        "FG-0002",
        [
            {"tool": "query_adverse_events", "arguments": {"generic_name": "metformin", "mode": "top_reactions"}},
            submit("DIARRHOEA", [cite("drug/event", "step:1", "results", "NAUSEA")]),
        ],
    )
    assert env.tool_calls == 1
    assert env.task["difficulty_features"]["min_tool_calls"] == 1
    _expect(reward, 0.35, correctness=0, citation=1, format=1, efficiency=0)
    assert any("not counted" in note for note in reward["notes"])


def test_step_citation_for_a_different_drug_is_invalid():
    _obs, reward, _env = drive(
        "FG-0002",
        [
            {"tool": "query_adverse_events", "arguments": {"generic_name": "warfarin", "mode": "top_reactions"}},
            submit("NAUSEA", [cite("drug/event", "step:1", "results", "NAUSEA")]),
        ],
    )
    _expect(reward, 0.10, correctness=0, citation=0, format=1, efficiency=0)
    assert any("not about metformin" in note for note in reward["notes"])


def test_not_found_without_a_search_on_an_unanswerable_task():
    _obs, reward, _env = drive("FG-0004", [submit("NOT_FOUND", [])])
    _expect(reward, 0.10, correctness=0, citation=0, format=1, efficiency=0)


def test_worked_example_two_of_three_citations_and_three_calls():
    task = json.loads(json.dumps(EXAMPLE_TASK))
    task["gold_citations"] = [{"endpoint": "drug/label", "ids": ["aaa", "bbb"], "fields": ["boxed_warning"]}]
    task["difficulty_features"]["min_tool_calls"] = 2
    warning = {"boxed_warning": "WARNING: BLEEDING RISK"}
    steps = [
        {
            "tool": "search_drug_label",
            "arguments": {"generic_name": "warfarin"},
            "output": {"id": "step:1", "results": [{"set_id": "aaa"}, {"set_id": "bbb"}], "meta": {"results": {"total": 2}}},
        },
        {
            "tool": "get_label_section",
            "arguments": {"set_id": "aaa", "section": "boxed_warning"},
            "output": {"set_id": "aaa", **warning},
        },
        {
            "tool": "get_label_section",
            "arguments": {"set_id": "bbb", "section": "boxed_warning"},
            "output": {"set_id": "bbb", **warning},
        },
    ]
    citations = [
        cite("drug/label", "aaa", "boxed_warning", "BLEEDING RISK"),
        cite("drug/label", "bbb", "boxed_warning", "BLEEDING RISK"),
        cite("drug/label", "aaa", "indications_and_usage", "BLEEDING RISK"),
    ]
    reward = score({"steps": steps, "final_answer": {"answer": "Yes. bleeding", "citations": citations}}, task)
    _expect(reward, 0.90, correctness=1, citation=2 / 3, format=1, efficiency=2 / 3)
    assert reward["citations_valid"] == 2
    assert reward["citations_total"] == 3


def test_stuffed_boolean_keywords_name_too_many_topics():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    env.step({"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}})
    obs, _reward, _done, _info = env.step(
        {"tool": "get_label_section", "arguments": {"set_id": WARFARIN_SET, "section": "boxed_warning"}}
    )
    quote = obs["output"]["boxed_warning"][:40]
    answer = "Yes. bleeding nausea death cancer liver stroke infarct hemorrhage"
    _obs, _reward, _done, info = env.step(submit(answer, [cite("drug/label", WARFARIN_SET, "boxed_warning", quote)]))
    _expect(info["reward"], 0.35, correctness=0, citation=1, format=1, efficiency=0)
    assert "correctness: answer names 6 boxed-warning topics (max 2)" in info["reward"]["notes"]


def test_one_character_quote_does_not_validate():
    _obs, reward, _env = drive(
        "FG-0003",
        [
            {
                "tool": "query_adverse_events",
                "arguments": {"generic_name": "metformin", "mode": "count", "serious": True, "year": 2025},
            },
            submit(20254, [cite("drug/event", "step:1", "meta.results.total", "1")]),
        ],
    )
    _expect(reward, 0.10, correctness=0, citation=0, format=1, efficiency=0)
