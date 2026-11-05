"""One-way application ownership must fail the symmetric disjointness check."""

from unittest.mock import patch

import pytest

from fdagym.tasks.disjoint import ApprovalMaps, OverlapError, assert_disjoint, sheet_entities, task_entities

MAPS = ApprovalMaps(
    owners={"NDA000001": frozenset({"alpha"}), "NDA000002": frozenset({"beta"})},
    sponsors={"NDA000001": "acme pharma", "NDA000002": "beta labs"},
)


def _named(drug: str) -> dict:
    return {
        "id": f"T-{drug.upper()}",
        "question": f"placeholder question about {drug}",
        "entity": {"generic_name": drug},
        "gold": {"kind": "exact", "value": drug},
        "oracle_calls": [{"tool": "search_drug_label", "arguments": {"generic_name": drug}}],
        "gold_citations": [],
    }


def _cites(app: str, *, field: str = "sponsor_name") -> dict:
    return {
        "id": "T-CITE",
        "question": f"Application {app} is held by a sponsor that also markets one of these drugs: gamma or delta.",
        "entity": {"generic_name": "gamma"},
        "gold": {"kind": "exact", "value": "gamma"},
        "oracle_calls": [{"tool": "search_approvals", "arguments": {"application_number": app}}],
        "gold_citations": [{"endpoint": "drug/drugsfda", "ids": [app], "fields": [field]}],
    }


def _patched():
    return (
        patch("fdagym.tasks.disjoint.approval_maps", lambda snapshot: MAPS),
        patch("fdagym.tasks.disjoint.load_universe", lambda: ["alpha", "beta", "gamma", "delta"]),
    )


def test_one_way_application_ownership_is_an_overlap():
    named, cited = _named("alpha"), _cites("NDA000001")
    maps_patch, universe_patch = _patched()
    with maps_patch, universe_patch:
        named_drugs = task_entities(named, object()).drugs
        cited_drugs = task_entities(cited, object()).drugs
        assert "alpha" in named_drugs
        assert "alpha" in cited_drugs
        assert "alpha" not in {((cited.get("entity") or {}).get("generic_name") or "").casefold()}
        with pytest.raises(OverlapError, match="alpha"):
            assert_disjoint([named], [cited], object())
        with pytest.raises(OverlapError, match="alpha"):
            assert_disjoint([cited], [named], object())


def test_one_way_sponsor_from_cited_application_is_an_overlap():
    who = {
        "id": "T-WHO",
        "question": "Who is the sponsor of application NDA999999?",
        "entity": {"generic_name": "gamma"},
        "gold": {"kind": "exact", "value": "ACME PHARMA"},
        "oracle_calls": [{"tool": "search_approvals", "arguments": {"application_number": "NDA999999"}}],
        "gold_citations": [{"endpoint": "drug/drugsfda", "ids": ["NDA999999"], "fields": ["sponsor_name"]}],
    }
    cited = _cites("NDA000001")
    maps_patch, universe_patch = _patched()
    with maps_patch, universe_patch:
        with pytest.raises(OverlapError, match="acme pharma"):
            assert_disjoint([who], [cited], object())
        with pytest.raises(OverlapError, match="acme pharma"):
            assert_disjoint([cited], [who], object())


def test_shared_question_text_is_an_overlap():
    left = _named("alpha")
    right = {**_named("beta"), "question": left["question"]}
    maps_patch, universe_patch = _patched()
    with maps_patch, universe_patch, pytest.raises(OverlapError, match="questions"):
        assert_disjoint([left], [right], object())


def test_disjoint_sheets_report_empty_overlap():
    maps_patch, universe_patch = _patched()
    with maps_patch, universe_patch:
        report = assert_disjoint([_named("alpha")], [_named("beta")], object())
        assert report["overlap"] == {"drugs": [], "applications": [], "sponsors": [], "questions": []}
        left = sheet_entities([_named("alpha")], object())
        right = sheet_entities([_named("beta")], object())
        assert left.drugs.isdisjoint(right.drugs)
