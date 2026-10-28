import pytest

from fdagym.rewards import score
from fdagym.rewards.correctness import score_correctness
from fdagym.rewards.topics import build_topics
from fdagym.tasks.hard_solver import question_faers_year, question_none, question_sponsor_faers
from tests.env.support import EXAMPLE_TASK, JsonSnapshot, task_count, task_nausea


def _task_with(**gold):
    task = task_count()
    task["gold"].update(gold)
    return task


def test_exact_match_normalizes_case_and_rejects_alternatives():
    task = task_nausea()
    assert score_correctness("nausea", task, valid_citations=1, format_ok=True, empty_grounded=False)[0] == 1
    score, notes = score_correctness(
        "NAUSEA or DIARRHOEA", task, valid_citations=1, format_ok=True, empty_grounded=False
    )
    assert score == 0
    assert any("alternatives" in note for note in notes)


def test_number_is_exact_and_partial_inside_tolerance():
    exact = task_count()
    gold = exact["gold"]["value"]
    assert gold == 20254
    assert score_correctness(gold, exact, valid_citations=1, format_ok=True, empty_grounded=False)[0] == 1
    assert score_correctness("20,254", exact, valid_citations=1, format_ok=True, empty_grounded=False)[0] == 1
    assert score_correctness(18000, exact, valid_citations=1, format_ok=True, empty_grounded=False)[0] == 0

    partial = _task_with(tolerance=0.1)
    pred = 19241
    score, notes = score_correctness(pred, partial, valid_citations=1, format_ok=True, empty_grounded=False)
    rel = abs(pred - gold) / gold
    assert rel < 0.1
    assert score == pytest.approx(1 - rel / 0.1)
    assert any("within" in note for note in notes)
    # 18000 is about 11% low on the union count, outside the 0.1 window.
    assert score_correctness(18000, partial, valid_citations=1, format_ok=True, empty_grounded=False)[0] == 0
    assert score_correctness(1000, partial, valid_citations=1, format_ok=True, empty_grounded=False)[0] == 0


def test_topic_builder_reads_bleeding_risk_from_labels():
    topics = build_topics(JsonSnapshot(), ["warfarin"])
    bleeding = next(topic for topic in topics if topic["heading"] == "BLEEDING RISK")
    assert "hemorrhage" in bleeding["terms"]
    assert "haemorrhage" in bleeding["terms"]


def test_honest_boxed_warning_sentence_names_one_topic():
    answer = "Yes. Bleeding risk: warfarin can cause major or fatal bleeding."
    score, notes = score_correctness(answer, EXAMPLE_TASK, valid_citations=1, format_ok=True, empty_grounded=False)
    assert score == 1.0
    assert any("polarity" in note for note in notes)


def test_short_quotes_keep_whole_values_and_nausea():
    count = {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": "metformin", "mode": "count", "serious": True, "year": 2025},
        "output": {"id": "step:1", "results": [], "meta": {"results": {"total": 20254}}},
    }
    reactions = {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": "metformin", "mode": "top_reactions"},
        "output": {
            "id": "step:1",
            "results": [{"term": "NAUSEA", "count": 32496}],
            "meta": {"results": {"total": 1}},
        },
    }
    empty = {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": "lecanemab", "mode": "count", "year": 2020},
        "output": {"id": "step:1", "results": [], "meta": {"results": {"total": 0}}},
    }

    def cited(step, answer, field, quote, task):
        return score(
            {"steps": [step], "final_answer": {"answer": answer, "citations": [
                {"endpoint": "drug/event", "id": "step:1", "field": field, "quote": quote}
            ]}},
            task,
        )

    assert cited(count, 20254, "meta.results.total", "20254", task_count())["citation"] == 1
    assert cited(count, 20254, "meta.results.total", "1", task_count())["citation"] == 0
    assert cited(empty, "NOT_FOUND", "meta.results.total", "0", {
        **task_count(),
        "id": "FG-0004",
        "answerable": False,
        "entity": {"generic_name": "lecanemab"},
        "gold": {"kind": "exact", "value": "NOT_FOUND"},
    })["citation"] == 1
    nausea = task_nausea()
    assert cited(reactions, "NAUSEA", "results", "NAUSEA", nausea)["citation"] == 1


def _features() -> dict:
    return {"min_tool_calls": 3}


def _empty_label(index: int, drug: str, *, section: str = "boxed_warning", total: int = 0) -> dict:
    return {
        "tool": "search_drug_label",
        "arguments": {"generic_name": drug, "has_section": section},
        "output": {
            "id": f"step:{index}",
            "results": [],
            "meta": {"results": {"total": total, "page": 1, "page_size": 10}},
        },
    }


def test_boxed_none_accepts_an_empty_boxed_warning_search_for_any_named_drug():
    drugs = ["zorvetinib", "velmorastat", "pemradine"]
    task = {
        "id": "FG-9101",
        "type": "compare",
        "question": question_none(drugs),
        "entity": {"generic_name": "zorvetinib"},
        "answerable": True,
        "gold": {"kind": "boolean", "value": False},
        "gold_citations": [{"endpoint": "drug/label", "ids": ["abc"], "fields": ["sections"]}],
        "difficulty_features": _features(),
    }
    steps = [_empty_label(index, drug) for index, drug in enumerate(drugs, start=1)]
    final = {
        "answer": "no",
        "citations": [
            {"endpoint": "drug/label", "id": "step:2", "field": "meta.results.total", "quote": '"total": 0'}
        ],
    }
    reward = score({"steps": steps, "final_answer": final}, task)
    assert reward["citation"] == 1
    assert reward["correctness"] == 1
    assert any("empty boxed_warning search" in note for note in reward["notes"])
    listed = {
        "answer": "no",
        "citations": [{"endpoint": "drug/label", "id": "step:3", "field": "results", "quote": '"results": []'}],
    }
    reward = score({"steps": steps, "final_answer": listed}, task)
    assert reward["citation"] == 1
    assert reward["correctness"] == 1


def test_boxed_none_rejects_a_nonzero_total_and_a_drug_the_question_does_not_name():
    drugs = ["zorvetinib", "velmorastat", "pemradine"]
    task = {
        "id": "FG-9101",
        "type": "compare",
        "question": question_none(drugs),
        "entity": {"generic_name": "zorvetinib"},
        "answerable": True,
        "gold": {"kind": "boolean", "value": False},
        "gold_citations": [{"endpoint": "drug/label", "ids": ["abc"], "fields": ["sections"]}],
        "difficulty_features": _features(),
    }
    steps = [_empty_label(1, "zorvetinib"), _empty_label(2, "warfarin")]
    cite = {"endpoint": "drug/label", "id": "step:2", "field": "meta.results.total", "quote": '"total": 0'}
    reward = score({"steps": steps, "final_answer": {"answer": "no", "citations": [cite]}}, task)
    assert reward["citation"] == 0
    steps[0] = _empty_label(1, "zorvetinib", total=4)
    cite = {"endpoint": "drug/label", "id": "step:1", "field": "meta.results.total", "quote": '"total": 0'}
    reward = score({"steps": steps, "final_answer": {"answer": "no", "citations": [cite]}}, task)
    assert reward["citation"] == 0


def test_compare_step_citation_may_name_any_drug_in_the_question():
    task = {
        "id": "FG-9102",
        "type": "compare",
        "question": question_faers_year(["zorvetinib", "velmorastat", "pemradine"], 2024),
        "entity": {"generic_name": "zorvetinib"},
        "answerable": True,
        "gold": {"kind": "exact", "value": "ondansetron"},
        "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}],
        "difficulty_features": _features(),
    }
    steps = [
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "velmorastat", "mode": "count", "year": 2024, "serious": True},
            "output": {"id": "step:1", "results": [], "meta": {"results": {"total": 12}}},
        }
    ]
    cite = {"endpoint": "drug/event", "id": "step:1", "field": "meta.results.total", "quote": "12"}
    reward = score({"steps": steps, "final_answer": {"answer": "zorvetinib", "citations": [cite]}}, task)
    assert reward["citation"] == 1
    assert not any("not about" in note for note in reward["notes"])


def test_sponsor_step_citation_stays_about_the_answer_drug():
    task = {
        "id": "FG-9103",
        "type": "sponsor_hop",
        "question": question_sponsor_faers("NDA999999", ["zorvetinib", "velmorastat"], "toxicity"),
        "entity": {"generic_name": "zorvetinib"},
        "answerable": True,
        "gold": {"kind": "exact", "value": "HEADACHE"},
        "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["results"]}],
        "difficulty_features": _features(),
    }
    steps = [
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "velmorastat", "mode": "top_reactions"},
            "output": {"id": "step:1", "results": [{"term": "HEADACHE", "count": 10}], "meta": {"results": {"total": 1}}},
        }
    ]
    cite = {"endpoint": "drug/event", "id": "step:1", "field": "results", "quote": "HEADACHE"}
    reward = score({"steps": steps, "final_answer": {"answer": "HEADACHE", "citations": [cite]}}, task)
    assert reward["citation"] == 0
    assert any("not about zorvetinib" in note for note in reward["notes"])


def test_boolean_needs_the_polarity_first_and_one_keyword():
    task = EXAMPLE_TASK
    args = {"valid_citations": 1, "format_ok": True, "empty_grounded": False}
    assert score_correctness("Yes. Bleeding risk.", task, **args)[0] == 1
    assert score_correctness("Yes. liver toxicity", task, **args)[0] == 0
    assert score_correctness("bleeding, yes", task, **args)[0] == 0
    assert score_correctness("No. bleeding", task, **args)[0] == 0


def test_gates_zero_a_right_answer():
    task = EXAMPLE_TASK
    assert score_correctness("Yes. bleeding", task, valid_citations=0, format_ok=True, empty_grounded=False)[0] == 0
    assert score_correctness("Yes. bleeding", task, valid_citations=1, format_ok=False, empty_grounded=False)[0] == 0


def test_judge_stub_raises_and_a_plug_in_is_used():
    task = {
        **{k: v for k, v in EXAMPLE_TASK.items() if k != "gold"},
        "gold": {"kind": "judge", "value": "Mention bleeding.", "rubric": ["mentions bleeding"]},
    }
    with pytest.raises(NotImplementedError):
        score_correctness("Yes. bleeding", task, valid_citations=1, format_ok=True, empty_grounded=False)

    class Half:
        def score(self, *, question: str, answer: str, gold: dict) -> float:
            return 0.5

    score_value, notes = score_correctness(
        "Yes. bleeding", task, valid_citations=1, format_ok=True, empty_grounded=False, judge=Half()
    )
    assert score_value == 0.5
    assert any("judge" in note for note in notes)


def test_quote_longer_than_the_field_does_not_match():
    task = EXAMPLE_TASK
    steps = [
        {
            "tool": "get_label_section",
            "arguments": {"set_id": "aaa", "section": "boxed_warning"},
            "output": {"set_id": "aaa", "boxed_warning": "BLEEDING"},
        }
    ]
    # Copy the example's single accepted id so this short field is the one being cited.
    task = {
        **task,
        "gold_citations": [{"endpoint": "drug/label", "ids": ["aaa"], "fields": ["boxed_warning"]}],
    }
    reward = score(
        {
            "steps": steps,
            "final_answer": {
                "answer": "Yes. bleeding",
                "citations": [
                    {
                        "endpoint": "drug/label",
                        "id": "aaa",
                        "field": "boxed_warning",
                        "quote": "BLEEDING and an invented clause",
                    }
                ],
            },
        },
        task,
    )
    assert reward["citation"] == 0
    assert reward["correctness"] == 0
    assert any("longer than field text" in note for note in reward["notes"])


def test_duplicate_citations_count_once():
    steps = [
        {
            "tool": "get_label_section",
            "arguments": {"set_id": "aaa", "section": "boxed_warning"},
            "output": {"set_id": "aaa", "boxed_warning": "WARNING: BLEEDING RISK"},
        }
    ]
    valid = {"endpoint": "drug/label", "id": "aaa", "field": "boxed_warning", "quote": "BLEEDING RISK"}
    invalid = {"endpoint": "drug/label", "id": "aaa", "field": "description", "quote": "BLEEDING RISK"}
    task = {
        **EXAMPLE_TASK,
        "gold_citations": [{"endpoint": "drug/label", "ids": ["aaa"], "fields": ["boxed_warning"]}],
    }
    reward = score(
        {"steps": steps, "final_answer": {"answer": "Yes. bleeding", "citations": [valid, valid, valid, invalid]}},
        task,
    )
    assert reward["citations_valid"] == 1
    assert reward["citations_total"] == 2
    assert reward["citation"] == 0.5
    assert any("duplicate" in note for note in reward["notes"])


def test_sections_list_is_cited_only_when_that_set_id_was_returned():
    steps = [
        {
            "tool": "search_drug_label",
            "arguments": {"generic_name": "atorvastatin"},
            "output": {
                "id": "step:1",
                "results": [
                    {
                        "set_id": "returned",
                        "sections": ["indications_and_usage", "dosage_and_administration", "adverse_reactions"],
                    }
                ],
                "meta": {"results": {"total": 259}},
            },
        }
    ]
    task = {
        "question": "Does atorvastatin's label carry a boxed warning?",
        "answerable": True,
        "entity": {"generic_name": "atorvastatin"},
        "difficulty_features": {"min_tool_calls": 1},
        "gold": {"kind": "boolean", "value": False},
        "gold_citations": [{"endpoint": "drug/label", "ids": [], "fields": ["sections"]}],
    }
    quote = "indications_and_usage"

    def _scored(set_id: str) -> dict:
        return score(
            {
                "steps": steps,
                "final_answer": {
                    "answer": "No",
                    "citations": [{"endpoint": "drug/label", "id": set_id, "field": "sections", "quote": quote}],
                },
            },
            task,
        )

    grounded = _scored("returned")
    assert grounded["citation"] == 1.0
    assert grounded["correctness"] == 1.0
    assert any("polarity match" in note for note in grounded["notes"])

    invented = _scored("never-returned")
    assert invented["citation"] == 0
    assert invented["correctness"] == 0
    assert any("did not appear" in note for note in invented["notes"])
