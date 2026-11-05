"""Compare correctness needs the hop proof plus a citation of the deciding drug."""

from fdagym.rewards import score
from fdagym.rewards.hops import hop_score
from tests.rewards.test_hops import COMPARE_TASK


def _faers(drug, rows, index):
    return {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": drug, "mode": "top_reactions"},
        "output": {
            "id": f"step:{index}",
            "results": rows,
            "meta": {"results": {"total": len(rows)}},
        },
    }


def _cite(index, quote="DYSPNOEA"):
    return {"endpoint": "drug/event", "id": f"step:{index}", "field": "results", "quote": quote}


def _full_steps():
    return [
        _faers("aprocitentan", [{"term": "DYSPNOEA", "count": 1}, {"term": "PAIN", "count": 1}], 1),
        _faers("ondansetron", [{"term": "DYSPNOEA", "count": 9}], 2),
        _faers("sitagliptin", [{"term": "DYSPNOEA", "count": 3}], 3),
    ]


def test_one_call_and_a_generic_quote_does_not_pay_compare_correctness():
    steps = [_faers("aprocitentan", [{"term": "DYSPNOEA", "count": 1}, {"term": "PAIN", "count": 1}], 1)]
    assert not hop_score(steps, COMPARE_TASK).proves()
    reward = score(
        {"steps": steps, "final_answer": {"answer": "ondansetron", "citations": [_cite(1, "PAIN")]}},
        COMPARE_TASK,
    )
    assert reward["citation"] == 1.0
    assert reward["correctness"] == 0.0
    assert reward["efficiency"] == 0.0
    assert any("compare proof" in note for note in reward["notes"])


def test_every_drug_queried_still_needs_a_citation_of_the_answered_drug():
    steps = _full_steps()
    assert hop_score(steps, COMPARE_TASK).proves()
    reward = score(
        {"steps": steps, "final_answer": {"answer": "ondansetron", "citations": [_cite(1, "PAIN")]}},
        COMPARE_TASK,
    )
    assert reward["citation"] == 1.0
    assert reward["correctness"] == 0.0
    assert any("deciding compare evidence" in note for note in reward["notes"])


def test_honest_compare_trace_keeps_correctness():
    steps = _full_steps()
    reward = score(
        {"steps": steps, "final_answer": {"answer": "ondansetron", "citations": [_cite(2)]}},
        COMPARE_TASK,
    )
    assert reward["correctness"] == 1.0
    assert reward["citation"] == 1.0
    assert reward["format"] == 1.0
    assert reward["efficiency"] == 1.0
    assert reward["total"] == 1.0
