"""Process hops on compare / sponsor_hop. Strict eval stays format-only on a miss."""

from fdagym.rewards import score
from fdagym.rewards.hops import HOP_CAP, hop_score
from fdagym.tasks.hard_solver import question_earliest
from tests.rewards.test_dense import _approval_task, _steps
from tests.rewards.test_dense import _submit as approval_submit

SPONSOR_TASK = {
    "id": "FG-8051",
    "type": "sponsor_hop",
    "question": (
        "Application BLA205692 is held by a sponsor that also markets one of these drugs: "
        "aprocitentan, atrasentan, or tirzepatide. That other drug has a boxed warning mentioning thyroid. "
        "What is that drug?"
    ),
    "sources": ["drug/drugsfda", "drug/label"],
    "entity": {"generic_name": "tirzepatide"},
    "split": "hard",
    "answerable": True,
    "difficulty": "hard",
    "difficulty_features": {
        "sources": 2,
        "min_tool_calls": 6,
        "needs_disambiguation": True,
        "needs_aggregation": False,
        "needs_normalization": False,
        "has_constraint": True,
    },
    "gold": {"kind": "exact", "value": "tirzepatide"},
    "gold_citations": [{"endpoint": "drug/drugsfda", "ids": ["BLA205692"], "fields": ["sponsor_name"]}],
    "oracle_calls": [{"tool": "search_approvals", "arguments": {"application_number": "BLA205692"}}],
    "snapshot_version": "2026-10-15",
}

COMPARE_TASK = {
    "id": "FG-8005",
    "type": "compare",
    "question": "Of aprocitentan, ondansetron, and sitagliptin, which drug has the most FAERS reports of DYSPNOEA?",
    "sources": ["drug/event"],
    "entity": {"generic_name": "ondansetron"},
    "split": "hard",
    "answerable": True,
    "difficulty": "hard",
    "difficulty_features": {
        "sources": 1,
        "min_tool_calls": 3,
        "needs_disambiguation": True,
        "needs_aggregation": True,
        "needs_normalization": False,
        "has_constraint": True,
    },
    "gold": {"kind": "exact", "value": "ondansetron"},
    "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["results"]}],
    "oracle_calls": [
        {"tool": "query_adverse_events", "arguments": {"generic_name": "aprocitentan", "mode": "top_reactions"}}
    ],
    "snapshot_version": "2026-10-15",
}


def _submit(answer="nope"):
    return {"answer": answer, "citations": []}


def _approval(app=None, drug=None, sponsor="Eli Lilly", extra=None):
    arguments = {}
    if app:
        arguments["application_number"] = app
    if drug:
        arguments["generic_name"] = drug
    row = {"application_number": app or "NDA215866", "sponsor_name": sponsor}
    if extra:
        row.update(extra)
    return {"tool": "search_approvals", "arguments": arguments, "output": {"id": "step:1", "results": [row]}}


def _label(drug, set_id="set-tirz", boxed=True, text="Risk of thyroid C-cell tumors"):
    row = {"set_id": set_id, "sections": ["boxed_warning"] if boxed else ["indications"]}
    return {
        "tool": "search_drug_label",
        "arguments": {"generic_name": drug, "has_section": "boxed_warning"},
        "output": {"id": "step:2", "results": [row]},
    }


def _section(set_id, text="Risk of thyroid C-cell tumors"):
    return {
        "tool": "get_label_section",
        "arguments": {"set_id": set_id, "section": "boxed_warning"},
        "output": {"set_id": set_id, "boxed_warning": text},
    }


def _events(drug, rows):
    return {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": drug, "mode": "top_reactions"},
        "output": {"id": "step:1", "results": rows, "meta": {"results": {"total": len(rows)}}},
    }


def test_strict_score_ignores_hops():
    steps = [_approval("BLA205692", sponsor="Eli Lilly")]
    strict = score({"steps": steps, "final_answer": _submit()}, SPONSOR_TASK)
    assert strict["total"] == 0.1
    assert strict["correctness"] == 0.0
    assert "hops" not in strict


def test_anchor_only_is_one_sixth_on_sponsor_drug():
    steps = [_approval("BLA205692", sponsor="Eli Lilly")]
    hops = hop_score(steps, SPONSOR_TASK)
    assert hops.applies
    assert hops.hits == 1
    assert hops.total == 6
    dense = score({"steps": steps, "final_answer": _submit()}, SPONSOR_TASK, dense=True)
    assert dense["hops"] == hops.rate
    assert dense["correctness"] == HOP_CAP * hops.rate
    assert dense["efficiency"] == 0.0
    assert dense["total"] == 0.6 * dense["correctness"] + 0.1


def test_found_one_sponsor_without_the_second_drug():
    steps = [
        _approval("BLA205692", sponsor="Eli Lilly"),
        _approval(drug="aprocitentan", sponsor="Idorsia"),
    ]
    hops = hop_score(steps, SPONSOR_TASK)
    assert hops.hits == 2
    assert hops.total == 6
    assert "hop cover:aprocitentan" in " ".join(hops.notes)
    assert any("match: miss" in note for note in hops.notes)
    dense = score({"steps": steps, "final_answer": _submit("aprocitentan")}, SPONSOR_TASK, dense=True)
    assert dense["correctness"] == HOP_CAP * (2 / 6)
    assert dense["efficiency"] == 0.0


def test_full_hops_wrong_answer_caps_at_half_correctness():
    steps = [
        _approval("BLA205692", sponsor="Eli Lilly"),
        _approval(drug="aprocitentan", sponsor="Idorsia"),
        _approval(drug="atrasentan", sponsor="AbbVie"),
        _approval(drug="tirzepatide", sponsor="Eli Lilly", extra={"application_number": "NDA215866"}),
        _label("tirzepatide"),
        _section("set-tirz"),
    ]
    hops = hop_score(steps, SPONSOR_TASK)
    assert hops.hits == hops.total == 6
    dense = score({"steps": steps, "final_answer": _submit("aprocitentan")}, SPONSOR_TASK, dense=True)
    assert dense["correctness"] == HOP_CAP
    assert dense["answer_correctness"] == 0.0
    assert dense["efficiency"] == 0.0
    assert dense["total"] == 0.6 * HOP_CAP + 0.1
    strict = score({"steps": steps, "final_answer": _submit("aprocitentan")}, SPONSOR_TASK)
    assert strict["correctness"] == 0.0
    assert strict["total"] == 0.1


def test_right_answer_is_not_capped_by_hops():
    steps = [
        _approval("BLA205692", sponsor="Eli Lilly"),
        _approval(drug="aprocitentan", sponsor="Idorsia"),
        _approval(drug="atrasentan", sponsor="AbbVie"),
        _approval(drug="tirzepatide", sponsor="Eli Lilly", extra={"application_number": "NDA215866"}),
        _label("tirzepatide"),
        _section("set-tirz"),
    ]
    final = {
        "answer": "tirzepatide",
        "citations": [
            {
                "endpoint": "drug/drugsfda",
                "id": "BLA205692",
                "field": "sponsor_name",
                "quote": "Eli Lilly",
            }
        ],
    }
    dense = score({"steps": steps, "final_answer": final}, SPONSOR_TASK, dense=True)
    assert dense["correctness"] == 1.0
    assert dense["hops"] == 1.0
    assert dense["efficiency"] > 0


def test_match_requires_every_candidate_page():
    steps = [
        _approval("BLA205692", sponsor="Eli Lilly"),
        _approval(drug="tirzepatide", sponsor="Eli Lilly"),
    ]
    hops = hop_score(steps, SPONSOR_TASK)
    assert any("match: miss" in note for note in hops.notes)
    assert hops.hits == 2


def test_sponsor_covers_without_anchor_are_zero_hops():
    steps = [
        _approval(drug="aprocitentan", sponsor="Idorsia"),
        _approval(drug="atrasentan", sponsor="AbbVie"),
        _approval(drug="tirzepatide", sponsor="Eli Lilly"),
        _label("tirzepatide"),
        _events("tirzepatide", [{"term": "NAUSEA", "count": 9}]),
    ]
    hops = hop_score(steps, SPONSOR_TASK)
    assert hops.hits == 0
    assert hops.rate == 0.0
    dense = score({"steps": steps, "final_answer": _submit("tirzepatide")}, SPONSOR_TASK, dense=True)
    assert dense["correctness"] == 0.0
    assert dense["total"] == 0.1


def test_compare_unique_max_is_the_last_hop():
    steps = [
        _events("aprocitentan", [{"term": "DYSPNOEA", "count": 1}]),
        _events("ondansetron", [{"term": "DYSPNOEA", "count": 9}]),
        _events("sitagliptin", [{"term": "DYSPNOEA", "count": 3}]),
    ]
    hops = hop_score(steps, COMPARE_TASK)
    assert hops.hits == hops.total == 4
    two = hop_score(steps[:2], COMPARE_TASK)
    assert two.hits == 2
    assert any("unique: miss" in note for note in two.notes)


def test_one_shot_tasks_do_not_get_hop_fallback():
    dense = score({"steps": _steps(), "final_answer": approval_submit(1999, quote="nope")}, _approval_task(), dense=True)
    assert dense["hops"] == 0.0
    assert dense["correctness"] == 0.0


_EARLIEST_DRUGS = ("ondansetron", "sitagliptin", "aprocitentan")

EARLIEST_TASK = {
    **COMPARE_TASK,
    "id": "FG-8001",
    "question": question_earliest(list(_EARLIEST_DRUGS)),
    "sources": ["drug/drugsfda", "drug/label"],
    "entity": {"generic_name": "ondansetron"},
    "gold": {"kind": "exact", "value": "ondansetron"},
    "gold_citations": [{"endpoint": "drug/drugsfda", "ids": ["NDA020007"], "fields": ["submissions"]}],
    "oracle_calls": [{"tool": "search_approvals", "arguments": {"generic_name": "ondansetron"}}],
    "difficulty_features": {
        "sources": 2,
        "min_tool_calls": 6,
        "needs_disambiguation": True,
        "needs_aggregation": True,
        "needs_normalization": False,
        "has_constraint": False,
    },
}


def _orig(drug: str, year: int, app: str):
    return {
        "tool": "search_approvals",
        "arguments": {"generic_name": drug},
        "output": {
            "id": "step:1",
            "results": [
                {
                    "application_number": app,
                    "sponsor_name": "Test",
                    "products": [{"active_ingredients": [{"name": drug}]}],
                    "submissions": [
                        {
                            "submission_type": "ORIG",
                            "submission_status": "AP",
                            "submission_status_date": f"{year}0101",
                        }
                    ],
                }
            ],
        },
    }


def _earliest_join(*, years_first: bool):
    years = [
        _orig("ondansetron", 1991, "NDA020007"),
        _orig("sitagliptin", 2006, "NDA021995"),
        _orig("aprocitentan", 2024, "NDA218686"),
    ]
    labels = [
        _label("ondansetron", set_id="set-ond", boxed=True),
        _label("sitagliptin", set_id="set-sita", boxed=True),
        _label("aprocitentan", set_id="set-apro", boxed=False),
    ]
    if years_first:
        return years + labels
    interleaved = []
    for year_step, label_step in zip(years, labels):
        interleaved.extend([year_step, label_step])
    return interleaved


def test_earliest_boxed_labels_only_is_zero_hops():
    steps = [
        _label("ondansetron", set_id="set-ond"),
        _label("sitagliptin", set_id="set-sita"),
        _label("aprocitentan", set_id="set-apro", boxed=False),
    ]
    hops = hop_score(steps, EARLIEST_TASK)
    assert hops.applies
    assert hops.hits == 0
    assert hops.total == 7
    assert hops.rate == 0.0
    dense = score({"steps": steps, "final_answer": _submit("ondansetron")}, EARLIEST_TASK, dense=True)
    assert dense["hops"] == 0.0
    assert dense["correctness"] == 0.0
    assert dense["total"] == 0.1
    strict = score({"steps": steps, "final_answer": _submit("ondansetron")}, EARLIEST_TASK)
    assert strict["correctness"] == 0.0
    assert strict["total"] == 0.1


def test_earliest_boxed_does_not_credit_boxed_without_year():
    steps = [
        _orig("ondansetron", 1991, "NDA020007"),
        _label("ondansetron", set_id="set-ond"),
        _label("sitagliptin", set_id="set-sita"),
        _label("aprocitentan", set_id="set-apro", boxed=False),
    ]
    hops = hop_score(steps, EARLIEST_TASK)
    notes = " ".join(hops.notes)
    assert hops.hits == 2
    assert "hop year:ondansetron: hit" in notes
    assert "hop boxed:ondansetron: hit" in notes
    assert "hop boxed:sitagliptin: miss" in notes
    assert "hop unique: miss" in notes


def test_earliest_boxed_years_then_labels_is_full_join():
    hops = hop_score(_earliest_join(years_first=True), EARLIEST_TASK)
    assert hops.hits == hops.total == 7
    dense = score(
        {"steps": _earliest_join(years_first=True), "final_answer": _submit("sitagliptin")},
        EARLIEST_TASK,
        dense=True,
    )
    assert dense["correctness"] == HOP_CAP
    assert dense["answer_correctness"] == 0.0


def test_earliest_boxed_interleaved_is_the_same_join():
    batched = hop_score(_earliest_join(years_first=True), EARLIEST_TASK)
    interleaved = hop_score(_earliest_join(years_first=False), EARLIEST_TASK)
    assert batched.hits == interleaved.hits == 7
    assert batched.rate == interleaved.rate == 1.0
