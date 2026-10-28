from fdagym.queries import LABEL_LIMIT, all_queries, approvals_search, faers_report_count, label_search


def test_all_queries_are_unique_and_bounded():
    qs = all_queries("Warfarin")
    assert len(qs) == len(set(qs)) == 2 + 2 * 2 * 7


def test_names_are_normalized_so_cache_keys_match():
    assert faers_report_count("Metformin ") == faers_report_count("metformin")


def test_faers_filters_compose():
    q = faers_report_count("metformin", serious=True, year=2025)
    assert q.search == (
        '(patient.drug.openfda.generic_name:"metformin" OR patient.drug.activesubstance.activesubstancename:"metformin")'
        " AND serious:1 AND receivedate:[20250101 TO 20251231]"
    )


def test_approvals_search_by_active_ingredient_not_openfda_block():
    assert approvals_search("semaglutide").search.startswith("products.active_ingredients.name:")


def test_label_and_approval_searches_fetch_every_record_in_one_request():
    assert label_search("metformin").limit == LABEL_LIMIT == 1000
    assert approvals_search("metformin").limit == 1000


def test_search_uses_spaces_not_literal_plus():
    # A literal "+AND+" gets URL-encoded to %2BAND%2B and openFDA answers 500.
    assert "+" not in faers_report_count("metformin", serious=True, year=2025).search
