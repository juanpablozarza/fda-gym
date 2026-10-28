import json

from fdagym import queries
from tests.env.support import FIXTURE, WARFARIN_SET, drive, submit


def test_recorded_keys_match_the_query_contract():
    raw = json.loads(FIXTURE.read_text())
    label_key = json.dumps(list(queries.label_search("warfarin")))
    approval_key = json.dumps(list(queries.approvals_search("semaglutide")))
    assert label_key in raw
    assert queries.label_search("warfarin").limit == 1000
    assert approval_key in raw

    labels = raw[label_key]["results"]
    assert WARFARIN_SET in {row["set_id"] for row in labels}
    assert len(labels) < 10

    apps = raw[approval_key]["results"]
    assert len(apps) == 6
    ozempic = next(row for row in apps if row["application_number"] == "NDA209637")
    assert "OZEMPIC" in ozempic["brand_names"]
    assert ozempic["openfda"] == {}
    orig = next(sub for sub in ozempic["submissions"] if sub["submission_type"] == "ORIG")
    assert orig["submission_status_date"] == "20171205"


def test_semaglutide_tool_returns_ozempic_orig_approval():
    observations, reward, _env = drive(
        "FG-0005",
        [
            {"tool": "search_approvals", "arguments": {"generic_name": "semaglutide"}},
            submit(
                "2017",
                [
                    {
                        "endpoint": "drug/drugsfda",
                        "id": "NDA209637",
                        "field": "submissions",
                        "quote": "20171205",
                    }
                ],
            ),
        ],
    )
    apps = observations[0]["output"]["results"]
    ozempic = next(row for row in apps if row["application_number"] == "NDA209637")
    assert "OZEMPIC" in ozempic["brand"]
    orig = next(sub for sub in ozempic["submissions"] if sub["submission_type"] == "ORIG")
    assert orig["submission_status_date"] == "20171205"
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0
