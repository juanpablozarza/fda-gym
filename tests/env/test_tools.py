from fdagym import queries
from fdagym.rewards import score
from fdagym.tools import PAGE_SIZE, SECTION_LIMIT, TOOLS, EpisodeState, execute, faers_modes
from tests.env.support import ALL_TASKS, WARFARIN_SET, JsonSnapshot, MapSnapshot, drive


def test_tool_names_are_the_contract():
    assert [tool["function"]["name"] for tool in TOOLS] == [
        "search_drug_label",
        "get_label_section",
        "query_adverse_events",
        "search_approvals",
        "submit_answer",
    ]


def _labels(n: int, section_len: int = 40) -> list[dict]:
    return [
        {
            "set_id": f"set-{i}",
            "effective_time": "20200101",
            "boxed_warning": ["W" * section_len],
            "openfda": {"brand_name": ["Brand"], "manufacturer_name": ["Maker"]},
        }
        for i in range(n)
    ]


def _label_snapshot(records: list[dict]) -> MapSnapshot:
    query = queries.label_search("pagedrug")
    body = {"meta": {"results": {"total": len(records)}}, "results": records}
    return MapSnapshot({(query.endpoint, query.search, query.count, query.limit): body})


def test_label_search_pages_and_only_serves_set_ids_from_this_episode():
    state = EpisodeState(_label_snapshot(_labels(12)))
    page2, error = execute(state, "search_drug_label", {"generic_name": "pagedrug", "page": 2}, step_index=4)
    assert error is None
    assert page2["id"] == "step:4"
    assert [row["set_id"] for row in page2["results"]] == ["set-10", "set-11"]
    assert page2["meta"]["results"]["total"] == 12
    assert page2["meta"]["results"]["page_size"] == PAGE_SIZE
    assert page2["results"][0]["brand"] == "Brand"
    assert "boxed_warning" in page2["results"][0]["sections"]

    section, error = execute(state, "get_label_section", {"set_id": "set-10", "section": "boxed_warning"}, step_index=5)
    assert error is None
    assert section["boxed_warning"] == "W" * 40

    missed, error = execute(state, "get_label_section", {"set_id": "set-0", "section": "boxed_warning"}, step_index=6)
    assert missed["error"] == error
    assert "set-0" in error


def test_long_sections_are_cut_at_1500_and_the_output_says_so():
    state = EpisodeState(_label_snapshot(_labels(1, section_len=2000)))
    execute(state, "search_drug_label", {"generic_name": "pagedrug"}, step_index=1)
    section, error = execute(state, "get_label_section", {"set_id": "set-0", "section": "boxed_warning"}, step_index=2)
    assert error is None
    assert len(section["boxed_warning"]) == SECTION_LIMIT == 1500
    assert section["truncated"] is True
    assert "1500" in section["note"]


def test_missing_mode_names_the_valid_values_and_the_reward_ignores_that_text():
    state = EpisodeState(JsonSnapshot())
    output, error = execute(state, "query_adverse_events", {"generic_name": "metformin"}, step_index=1)
    assert error == f"invalid arguments: mode is required; valid values are {', '.join(faers_modes())}"
    assert output["error"] == error
    _output, bad_mode = execute(
        state, "query_adverse_events", {"generic_name": "metformin", "mode": "list"}, step_index=1
    )
    assert bad_mode == f"invalid arguments: mode must be one of {', '.join(faers_modes())}"
    _output, serious = execute(
        state,
        "query_adverse_events",
        {"generic_name": "metformin", "mode": "count", "serious": False},
        step_index=1,
    )
    assert serious.startswith("invalid arguments:")
    assert "top_reactions" not in serious

    task = next(item for item in ALL_TASKS if item["id"] == "FG-0004")
    final = {"answer": "NOT_FOUND", "citations": []}

    def steps(message: str) -> list[dict]:
        return [
            {
                "index": 1,
                "thought": None,
                "tool": "query_adverse_events",
                "arguments": {"generic_name": "metformin"},
                "output": {"error": message},
                "error": message,
                "latency_ms": 0,
            }
        ]

    old = score(
        {"steps": steps("invalid arguments: 'mode' is a required property"), "final_answer": final},
        task,
    )
    new = score({"steps": steps(error), "final_answer": final}, task)
    assert old["total"] == new["total"]
    assert old["correctness"] == new["correctness"]
    assert old["citation"] == new["citation"]
    assert old["format"] == new["format"]
    assert old["efficiency"] == new["efficiency"]


def test_bad_arguments_and_snapshot_miss_are_observations():
    state = EpisodeState(JsonSnapshot())
    cases = [
        ("query_adverse_events", {"generic_name": "metformin", "mode": "count", "serious": False, "year": 2025}),
        ("query_adverse_events", {"generic_name": "metformin", "mode": "count", "year": 2019}),
        ("search_drug_label", {"generic_name": "warfarin", "page": 0}),
        ("search_drug_label", {"generic_name": "warfarin", "has_section": "Boxed"}),
        ("no_such_tool", {}),
        ("search_drug_label", {"generic_name": "aspirin"}),
    ]
    for name, arguments in cases:
        output, error = execute(state, name, arguments, step_index=1)
        assert error, name
        assert output["error"] == error


def test_count_hides_the_incidental_report_and_keeps_the_total():
    query = queries.faers_report_count("metformin", serious=True, year=2025)
    body = JsonSnapshot().get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
    state = EpisodeState(JsonSnapshot())
    output, error = execute(
        state,
        "query_adverse_events",
        {"generic_name": "metformin", "mode": "count", "serious": True, "year": 2025},
        step_index=2,
    )
    assert error is None
    assert body["meta"]["results"]["total"] == 20254
    assert output["results"] == []
    assert output["meta"]["results"]["total"] == 20254
    assert output["id"] == "step:2"


def test_approvals_expose_products_so_dosage_form_can_be_cited():
    query = queries.approvals_search("exampledrug")
    record = {
        "application_number": "NDA1",
        "sponsor_name": "Example",
        "products": [
            {
                "brand_name": "EXAMPLE",
                "dosage_form": "TABLET",
                "route": "ORAL",
                "active_ingredients": [{"name": "EXAMPLEDRUG"}],
            }
        ],
        "submissions": [{"submission_type": "ORIG", "submission_status_date": "20200102"}],
    }
    snap = MapSnapshot({(query.endpoint, query.search, query.count, query.limit): {"results": [record]}})
    output, error = execute(EpisodeState(snap), "search_approvals", {"generic_name": "exampledrug"}, step_index=1)
    assert error is None
    assert output["results"][0]["products"][0]["dosage_form"] == "TABLET"
    assert output["results"][0]["products"][0]["brand_name"] == "EXAMPLE"
    assert output["results"][0]["submissions"][0]["submission_status_date"] == "20200102"


def test_approvals_read_brand_from_products_when_openfda_is_missing():
    query = queries.approvals_search("exampledrug")
    record = {
        "application_number": "NDA1",
        "sponsor_name": "Example",
        "products": [{"brand_name": "EXAMPLE"}],
        "submissions": [{"submission_type": "ORIG", "submission_status_date": "20200102"}],
    }
    snap = MapSnapshot({(query.endpoint, query.search, query.count, query.limit): {"results": [record]}})
    output, error = execute(EpisodeState(snap), "search_approvals", {"generic_name": "exampledrug"}, step_index=1)
    assert error is None
    assert output["results"][0]["brand"] == ["EXAMPLE"]
    assert snap.calls[0][1] == query.search


def test_warfarin_page_two_is_empty_because_the_fixture_keeps_five_labels():
    observations, _reward, _env = drive(
        "FG-0001",
        [
            {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "page": 1}},
            {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "page": 2}},
            {"tool": "submit_answer", "arguments": {"answer": "unknown", "citations": []}},
        ],
    )
    assert WARFARIN_SET in {row["set_id"] for row in observations[0]["output"]["results"]}
    assert observations[0]["output"]["meta"]["results"]["total"] == 74
    assert observations[1]["output"]["results"] == []
