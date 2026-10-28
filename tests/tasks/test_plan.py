from fdagym.queries import approvals_search, faers_report_count, faers_top_reactions, label_search
from fdagym.tasks.plan import absence_year, allow_unanswerable, load_universe, planned_queries, probe_year


def test_planned_queries_cover_labels_approvals_and_both_faers_shapes():
    drug = "semaglutide"
    year = probe_year(drug)
    gap = absence_year(drug)
    assert year != gap
    assert planned_queries(drug) == [
        label_search(drug),
        approvals_search(drug),
        faers_top_reactions(drug),
        faers_top_reactions(drug, serious=True, year=year),
        faers_top_reactions(drug, serious=True, year=gap),
        faers_report_count(drug, serious=True, year=year),
    ]


def test_probe_year_depends_on_the_drug_not_on_a_seed():
    assert probe_year("warfarin") == probe_year("Warfarin ")
    assert probe_year("warfarin") != probe_year("metformin")


def test_universe_is_drugs_txt_without_ibuprofen():
    drugs = load_universe()
    assert len(drugs) == len(set(drugs)) == 150
    assert "ibuprofen" not in {name.lower() for name in drugs}
    assert "diclofenac" in drugs
    assert "xanomeline" in drugs


def test_collision_years_are_not_clean_not_found():
    assert allow_unanswerable("xanomeline", 2020) is True
    assert allow_unanswerable("atrasentan", 2024) is True
    assert allow_unanswerable("aprocitentan", 2021) is False
    assert allow_unanswerable("resmetirom", 2020) is False
    assert allow_unanswerable("resmetirom", 2022) is False
    assert allow_unanswerable("resmetirom", 2023) is True
    assert allow_unanswerable("aprocitentan", 2020) is True
