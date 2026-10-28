"""JP's rule: dense sponsor-hop citation credit needs a lookup of the hop-target drug.

The hop-target drug is ``entity.generic_name`` (the sponsor's other drug). Gold is
that name only for ``sponsor_drug``. A search of a different candidate does not
count. Strict eval is not on this path.
"""

from fdagym.rewards import score
from fdagym.rewards.combine import WEIGHTS
from fdagym.rewards.hops import HOP_CAP
from tests.rewards.test_dense import _approval_task, _steps
from tests.rewards.test_dense import _submit as approval_submit
from tests.rewards.test_hops import (
    COMPARE_TASK,
    SPONSOR_TASK,
    _approval,
    _events,
    _label,
    _section,
)

TARGET = "tirzepatide"
ANCHOR = "BLA205692"
JP_NOTE = "JP's rule: no sponsor-hop citation credit without a lookup of the hop-target drug"


def _cite(quote="Eli Lilly"):
    return {
        "endpoint": "drug/drugsfda",
        "id": ANCHOR,
        "field": "sponsor_name",
        "quote": quote,
    }


def _final(answer, quote="Eli Lilly"):
    return {"answer": answer, "citations": [_cite(quote)]}


def _anchor():
    return _approval(ANCHOR, sponsor="Eli Lilly")


def _drug(name, sponsor):
    extra = {"application_number": "NDA215866"} if name == TARGET else None
    return _approval(drug=name, sponsor=sponsor, extra=extra)


def _dense(steps, answer, quote="Eli Lilly", task=None):
    final = _final(answer, quote) if quote is not None else {"answer": answer, "citations": []}
    return score({"steps": steps, "final_answer": final}, task or SPONSOR_TASK, dense=True)


def _full_steps():
    return [
        _anchor(),
        _drug("aprocitentan", "Idorsia"),
        _drug("atrasentan", "AbbVie"),
        _drug(TARGET, "Eli Lilly"),
        _label(TARGET),
        _section("set-tirz"),
    ]


def _honest_steps():
    """Anchor plus every candidate approval page. No label, so the last hop misses."""
    return [
        _anchor(),
        _drug("aprocitentan", "Idorsia"),
        _drug("atrasentan", "AbbVie"),
        _drug(TARGET, "Eli Lilly"),
    ]


def _one_hop_steps():
    """Anchor, then the two distractors. The hop-target drug is never looked up."""
    return [
        _anchor(),
        _drug("aprocitentan", "Idorsia"),
        _drug("atrasentan", "AbbVie"),
    ]


def ordering_cases():
    """Dense scores for the report. Full two-hop > two-hop wrong >= honest failure > one-hop guess."""
    return {
        "full_two_hop": _dense(_full_steps(), TARGET),
        "two_hop_wrong": _dense([_anchor(), _drug(TARGET, "Eli Lilly")], "aprocitentan"),
        "honest_failure": _dense(_honest_steps(), "aprocitentan", quote=None),
        "one_hop_guess": _dense(_one_hop_steps(), "aprocitentan"),
        "anchor_only": _dense([_anchor()], "aprocitentan"),
    }


def test_one_hop_right_citation_earns_no_partial_credit():
    """(a) Anchor lookup plus a right citation of that application. No second-drug lookup."""
    steps = [_anchor()]
    wrong = _dense(steps, "aprocitentan")
    assert wrong["citation"] == 0.0
    assert wrong["citations_valid"] == 0
    assert JP_NOTE in wrong["notes"]
    assert wrong["correctness"] == HOP_CAP * (1 / 6)
    assert wrong["format"] == 1.0
    assert wrong["efficiency"] == 0.0
    assert wrong["total"] == 0.6 * wrong["correctness"] + 0.1
    # The withheld term is exactly the citation weight. Hop, format, and efficiency stay.
    assert wrong["total"] + WEIGHTS["citation"] == 0.4

    poked = _dense(_one_hop_steps(), "aprocitentan")
    assert poked["citation"] == 0.0
    assert poked["total"] == 0.25
    assert poked["total"] + WEIGHTS["citation"] == 0.5

    lucky = _dense(steps, TARGET)
    assert lucky["citation"] == 0.0
    assert lucky["correctness"] == HOP_CAP * (1 / 6)
    assert lucky["correctness"] < 1.0
    assert lucky["efficiency"] == 0.0
    assert lucky["total"] == wrong["total"]

    id_only = _dense(steps, TARGET, quote="nope")
    assert id_only["citation"] == 0.0
    assert id_only["correctness"] < 1.0
    assert JP_NOTE in id_only["notes"]

    failed = {
        "tool": "search_approvals",
        "arguments": {"generic_name": TARGET},
        "output": {"error": "no snapshot data"},
        "error": "no snapshot data",
    }
    assert _dense([*steps, failed], "aprocitentan")["citation"] == 0.0

    strict = score({"steps": steps, "final_answer": _final("aprocitentan")}, SPONSOR_TASK)
    assert strict["citation"] == 1.0
    assert strict["correctness"] == 0.0
    assert strict["total"] == 0.35
    assert JP_NOTE not in strict["notes"]
    strict_lucky = score({"steps": steps, "final_answer": _final(TARGET)}, SPONSOR_TASK)
    assert strict_lucky["correctness"] == 0.0
    assert strict_lucky["citation"] == 1.0
    assert strict_lucky["efficiency"] == 0.0
    assert strict_lucky["total"] == 0.35
    assert any("correctness needs the sponsor proof" in note for note in strict_lucky["notes"])


def test_two_hop_wrong_answer_keeps_partial_credit():
    """(b) The hop-target drug was looked up. A wrong answer still keeps citation credit."""
    steps = [_anchor(), _drug(TARGET, "Eli Lilly")]
    cited = _dense(steps, "aprocitentan")
    assert cited["citation"] == 1.0
    assert cited["citations_valid"] == 1
    assert JP_NOTE not in cited["notes"]
    assert cited["correctness"] == HOP_CAP * (2 / 6)
    assert cited["efficiency"] == 0.0
    assert cited["total"] == round(0.6 * cited["correctness"] + 0.25 + 0.1, 10)

    partial = _dense(steps, "aprocitentan", quote="nope")
    assert partial["citation"] == 0.5
    assert partial["citations_valid"] == 0
    assert JP_NOTE not in partial["notes"]
    assert partial["correctness"] == cited["correctness"]
    assert partial["total"] == round(0.6 * partial["correctness"] + 0.25 * 0.5 + 0.1, 10)

    label_only = _dense([_anchor(), _label(TARGET)], "aprocitentan")
    assert label_only["citation"] == 1.0
    faers_only = _dense(
        [_anchor(), _events(TARGET, [{"term": "NAUSEA", "count": 4}])],
        "aprocitentan",
    )
    assert faers_only["citation"] == 1.0


def test_full_correct_two_hop_score_is_unchanged():
    """(c) Both hops, right answer, right citation. Dense and strict stay at 1."""
    steps = _full_steps()
    final = _final(TARGET)
    dense = score({"steps": steps, "final_answer": final}, SPONSOR_TASK, dense=True)
    strict = score({"steps": steps, "final_answer": final}, SPONSOR_TASK)
    assert dense["correctness"] == 1.0
    assert dense["citation"] == 1.0
    assert dense["format"] == 1.0
    assert dense["efficiency"] == 1.0
    assert dense["hops"] == 1.0
    assert dense["total"] == 1.0
    assert strict["total"] == 1.0
    assert JP_NOTE not in dense["notes"]


def test_honest_failure_is_unchanged():
    """(d) Searched the target, wrong answer, no citation. Still the hop-plus-format score."""
    steps = _honest_steps()
    final = {"answer": "aprocitentan", "citations": []}
    dense = score({"steps": steps, "final_answer": final}, SPONSOR_TASK, dense=True)
    strict = score({"steps": steps, "final_answer": final}, SPONSOR_TASK)
    assert dense["citation"] == 0.0
    assert dense["citations_valid"] == 0
    assert dense["correctness"] == HOP_CAP * (5 / 6)
    assert dense["efficiency"] == 0.0
    assert dense["total"] == 0.35
    assert strict["total"] == 0.1
    assert JP_NOTE not in dense["notes"]


def test_non_sponsor_questions_are_unchanged():
    """(e) One-shot partial citation and compare hop credit stay on the old formula."""
    year = score(
        {"steps": _steps(), "final_answer": approval_submit(19981217, quote="nope")},
        _approval_task(),
        dense=True,
    )
    assert year["citation"] == 0.5
    assert year["correctness"] == 0.7
    assert year["efficiency"] == 1.0
    assert year["total"] == 0.6 * 0.7 + 0.25 * 0.5 + 0.1 + 0.05
    assert JP_NOTE not in year["notes"]

    compare_steps = [
        _events("aprocitentan", [{"term": "DYSPNOEA", "count": 1}]),
        _events("ondansetron", [{"term": "DYSPNOEA", "count": 9}]),
        _events("sitagliptin", [{"term": "DYSPNOEA", "count": 3}]),
    ]
    compare = score(
        {"steps": compare_steps, "final_answer": {"answer": "sitagliptin", "citations": []}},
        COMPARE_TASK,
        dense=True,
    )
    assert compare["hops"] == 1.0
    assert compare["correctness"] == HOP_CAP
    assert compare["citation"] == 0.0
    assert compare["total"] == 0.6 * HOP_CAP + 0.1
    assert JP_NOTE not in compare["notes"]


def test_hop_target_is_the_entity_drug_not_the_gold_string():
    """sponsor_faers gold is a reaction and sponsor_count gold is a number. The lookup is the entity drug."""
    faers = {
        **SPONSOR_TASK,
        "question": (
            "Application BLA205692 is held by a sponsor that also markets one of these drugs: "
            "aprocitentan or tirzepatide. That other drug has a boxed warning mentioning thyroid. "
            "What is that drug's most-reported FAERS reaction?"
        ),
        "gold": {"kind": "exact", "value": "NAUSEA"},
        "entity": {"generic_name": TARGET},
    }
    gold_as_name = _dense([_anchor(), _drug("NAUSEA", "Nobody")], "aprocitentan", task=faers)
    assert gold_as_name["citation"] == 0.0
    looked_up = _dense([_anchor(), _drug(TARGET, "Eli Lilly")], "aprocitentan", task=faers)
    assert looked_up["citation"] == 1.0

    count = {
        **SPONSOR_TASK,
        "question": (
            "Application BLA205692 is held by a sponsor that also markets one of these drugs: "
            "aprocitentan, atrasentan, or tirzepatide. How many serious FAERS reports did that drug receive in 2023?"
        ),
        "gold": {"kind": "number", "value": 12},
        "entity": {"generic_name": TARGET},
    }
    missed = _dense([_anchor()], "4", task=count)
    assert missed["citation"] == 0.0
    found = _dense([_anchor(), _events(TARGET, [])], "4", task=count)
    assert found["citation"] == 1.0


def test_score_ordering():
    cases = ordering_cases()
    assert cases["full_two_hop"]["total"] > cases["two_hop_wrong"]["total"]
    assert cases["two_hop_wrong"]["total"] >= cases["honest_failure"]["total"]
    assert cases["honest_failure"]["total"] > cases["one_hop_guess"]["total"]
    assert cases["full_two_hop"]["total"] == 1.0
    assert cases["two_hop_wrong"]["total"] == 0.45
    assert cases["honest_failure"]["total"] == 0.35
    assert cases["one_hop_guess"]["total"] == 0.25
