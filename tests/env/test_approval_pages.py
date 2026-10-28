"""Compact approval pages stay inside the 3,500-character observation clip."""

import json

from fdagym import queries
from fdagym.tools import APPROVAL_VISIBLE_CHARS, EpisodeState, execute
from fdagym.train.rollout import observation_payload
from tests.env.support import MapSnapshot


def _visible(output: dict) -> int:
    observation = {"tool": "search_approvals", "output": output, "error": None, "tool_calls_remaining": 5}
    return len(observation_payload(observation))


def _application(number: str, date: str, *, supplements: int = 12, products: int = 3) -> dict:
    return {
        "application_number": number,
        "sponsor_name": "EXAMPLE SPONSOR",
        "products": [
            {
                "brand_name": "BRAND",
                "dosage_form": "TABLET",
                "route": "ORAL",
                "active_ingredients": [
                    {"name": "EXAMPLEDRUG", "strength": f"{strength}MG"} for strength in range(1, products + 1)
                ],
            }
        ],
        "submissions": [
            {"submission_type": "SUPPL", "submission_number": str(i), "submission_status_date": "20200101"}
            for i in range(supplements)
        ]
        + [{"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": date}],
    }


def _snapshot(records: list[dict]) -> MapSnapshot:
    query = queries.approvals_search("exampledrug")
    body = {"meta": {"results": {"total": len(records)}}, "results": records}
    return MapSnapshot({(query.endpoint, query.search, query.count, query.limit): body})


def test_pages_are_nda_first_under_3500_and_cover_every_application():
    records = []
    for index in range(20):
        # Later numbers carry the earlier approval, so input order is the wrong order.
        records.append(_application(f"NDA{index:06d}", f"{2020 - index:04d}0101"))
    records.extend(_application(f"ANDA{index:06d}", "19800101") for index in range(8))
    records.append(_application("BLA000099", "19950101"))
    snap = _snapshot(records)

    page1, error = execute(EpisodeState(snap), "search_approvals", {"generic_name": "exampledrug"}, step_index=1)
    assert error is None
    assert page1["meta"]["results"]["page"] == 1
    assert page1["meta"]["results"]["total"] == 29
    assert _visible(page1) <= APPROVAL_VISIBLE_CHARS
    numbers = [row["application_number"] for row in page1["results"]]
    assert numbers
    assert all(number.startswith("NDA") for number in numbers)
    assert numbers[0] == "NDA000019"
    assert all(sub["submission_type"] == "ORIG" for row in page1["results"] for sub in row["submissions"])
    assert page1["results"][0]["n_supplements"] == 12
    assert page1["results"][0]["products"] == [
        {
            "brand_name": "BRAND",
            "dosage_form": "TABLET",
            "route": "ORAL",
            "active_ingredients": [{"name": "EXAMPLEDRUG"}],
        }
    ]

    seen = list(numbers)
    page_no = 2
    while True:
        output, error = execute(
            EpisodeState(snap),
            "search_approvals",
            {"generic_name": "exampledrug", "page": page_no},
            step_index=1,
        )
        assert error is None
        assert _visible(output) <= APPROVAL_VISIBLE_CHARS
        assert output["meta"]["results"]["page"] == page_no
        if not output["results"]:
            break
        seen.extend(row["application_number"] for row in output["results"])
        page_no += 1
        assert page_no < 12
    expected = [f"NDA{index:06d}" for index in range(19, -1, -1)]
    expected.append("BLA000099")
    expected.extend(f"ANDA{index:06d}" for index in range(8))
    assert seen == expected


def test_one_fat_application_stays_inside_the_clip_and_keeps_the_original_date():
    products = [
        {
            "brand_name": f"BRAND{i:02d}",
            "dosage_form": f"FORM{i:02d}",
            "route": "ORAL",
            "active_ingredients": [{"name": "EXAMPLEDRUG" * 20, "strength": "10MG"}],
        }
        for i in range(40)
    ]
    record = _application("NDA020977", "19981217", supplements=80)
    record["products"] = products
    snap = _snapshot([record])
    output, error = execute(EpisodeState(snap), "search_approvals", {"application_number": "NDA 020977"}, step_index=3)
    assert error is None
    assert _visible(output) <= APPROVAL_VISIBLE_CHARS
    row = output["results"][0]
    assert row["application_number"] == "NDA020977"
    assert row["submissions"][0]["submission_status_date"] == "19981217"
    assert row["n_supplements"] == 80
    assert row["products"][0]["dosage_form"] == "FORM00"
    text = json.dumps(output)
    assert "SUPPL" not in text


def test_page_zero_is_rejected_and_a_page_past_the_end_is_empty():
    snap = _snapshot([_application("NDA000001", "19990101", supplements=0, products=1)])
    bad, error = execute(
        EpisodeState(snap), "search_approvals", {"generic_name": "exampledrug", "page": 0}, step_index=1
    )
    assert bad["error"] == error
    empty, error = execute(
        EpisodeState(snap), "search_approvals", {"generic_name": "exampledrug", "page": 5}, step_index=1
    )
    assert error is None
    assert empty["results"] == []
    assert empty["meta"]["results"]["total"] == 1
    assert empty["meta"]["results"]["page"] == 5
