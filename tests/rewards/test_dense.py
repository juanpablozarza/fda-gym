"""Dense training reward. The strict score stays the eval baseline."""

from fdagym.rewards import score
from fdagym.rewards.combine import WEIGHTS

_FEATURES = {
    "sources": 1,
    "min_tool_calls": 1,
    "needs_disambiguation": False,
    "needs_aggregation": False,
    "needs_normalization": False,
    "has_constraint": False,
}


def _approval_task():
    return {
        "id": "FG-0099",
        "type": "approval_year",
        "question": "In what year did the FDA approve the original NDA for ZIAGEN (abacavir)?",
        "sources": ["drug/drugsfda"],
        "entity": {"generic_name": "abacavir", "brand_name": "ZIAGEN"},
        "split": "train",
        "answerable": True,
        "difficulty": "medium",
        "difficulty_features": dict(_FEATURES),
        "gold": {"kind": "number", "value": 1998, "tolerance": 0},
        "gold_citations": [
            {"endpoint": "drug/drugsfda", "ids": ["NDA020977"], "fields": ["submissions"]}
        ],
        "oracle_calls": [],
        "snapshot_version": "2026-10-15",
    }


def _steps():
    return [
        {
            "tool": "search_approvals",
            "arguments": {"generic_name": "abacavir"},
            "output": {
                "id": "step:1",
                "results": [
                    {
                        "application_number": "NDA020977",
                        "products": [{"brand_name": "ZIAGEN", "active_ingredients": [{"name": "ABACAVIR"}]}],
                        "submissions": [
                            {"submission_type": "ORIG", "submission_status_date": "19981217"}
                        ],
                    }
                ],
            },
        }
    ]


def _submit(answer, *, field="submissions", quote="19981217"):
    return {
        "answer": answer,
        "citations": [
            {"endpoint": "drug/drugsfda", "id": "NDA020977", "field": field, "quote": quote}
        ],
    }


def test_strict_score_stays_format_only_when_the_quote_is_wrong():
    reward = score({"steps": _steps(), "final_answer": _submit(19981217, quote="nope")}, _approval_task())
    assert reward["total"] == 0.1
    assert reward["citation"] == 0.0
    assert reward["correctness"] == 0.0
    assert WEIGHTS["format"] == 0.10


def test_dense_year_and_id_only_citation_beat_format_only():
    task = _approval_task()
    final = _submit(19981217, quote="nope")
    strict = score({"steps": _steps(), "final_answer": final}, task)
    dense = score({"steps": _steps(), "final_answer": final}, task, dense=True)
    assert strict["total"] == 0.1
    assert dense["citation"] == 0.5
    assert dense["correctness"] == 0.7
    assert dense["total"] > strict["total"]
    assert dense["format"] == 1.0


def test_punctuation_and_synonym_are_partial_and_do_not_change_strict():
    task = {
        **_approval_task(),
        "gold": {"kind": "exact", "value": "ZIAGEN"},
        "gold_citations": [
            {"endpoint": "drug/drugsfda", "ids": ["NDA020977"], "fields": ["products"]}
        ],
    }
    steps = _steps()
    punct = _submit("ZIAGEN,", field="products", quote="nope")
    synonym = _submit("abacavir", field="products", quote="nope")
    strict = score({"steps": steps, "final_answer": punct}, task)
    dense_punct = score({"steps": steps, "final_answer": punct}, task, dense=True)
    dense_syn = score({"steps": steps, "final_answer": synonym}, task, dense=True)
    assert strict["correctness"] == 0.0
    assert dense_punct["correctness"] == 0.85
    assert dense_syn["correctness"] == 0.6
    assert dense_punct["total"] != dense_syn["total"] or dense_punct["correctness"] != dense_syn["correctness"]
    assert dense_punct["total"] > 0.1


def test_numeric_band_is_partial_and_years_are_not():
    task = {
        **_approval_task(),
        "type": "faers_count",
        "gold": {"kind": "number", "value": 1000, "tolerance": 0},
        "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}],
        "entity": {"generic_name": "metformin"},
    }
    steps = [
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "metformin", "mode": "count"},
            "output": {"id": "step:1", "results": [], "meta": {"results": {"total": 1000}}},
        }
    ]
    final = {
        "answer": 950,
        "citations": [
            {"endpoint": "drug/event", "id": "step:1", "field": "meta.results.total", "quote": "nope"}
        ],
    }
    dense = score({"steps": steps, "final_answer": final}, task, dense=True)
    assert 0 < dense["correctness"] < 1
    assert score({"steps": steps, "final_answer": final}, task)["correctness"] == 0.0

    year = _approval_task()
    off_by_one = score(
        {"steps": _steps(), "final_answer": _submit(1999, quote="nope")},
        year,
        dense=True,
    )
    assert off_by_one["correctness"] == 0.0


def test_distractor_id_earns_no_partial_credit():
    final = {
        "answer": 1998,
        "citations": [
            {"endpoint": "drug/drugsfda", "id": "NDA999999", "field": "submissions", "quote": "19981217"}
        ],
    }
    dense = score({"steps": _steps(), "final_answer": final}, _approval_task(), dense=True)
    assert dense["citation"] == 0.0
    assert dense["correctness"] == 0.0
    assert dense["total"] == 0.1
