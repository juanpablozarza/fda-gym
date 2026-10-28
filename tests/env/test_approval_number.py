"""search_approvals by application number, against an in-memory cache and the freeze."""

import json
from pathlib import Path

import pytest

from fdagym.approvals import parse_application_number
from fdagym.env import FDAEnv
from fdagym.openfda.store import DiskSnapshot
from fdagym.tools import TOOLS, EpisodeState, execute
from fdagym.train.gold import gold_submit
from tests.env.support import MapSnapshot

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15"
TASKS = ROOT / "data" / "tasks"


def _record(number: str, *, sponsor: str = "EXAMPLE", brand: str = "EXAMPLE", form: str = "TABLET") -> dict:
    return {
        "application_number": number,
        "sponsor_name": sponsor,
        "products": [
            {
                "brand_name": brand,
                "dosage_form": form,
                "route": "ORAL",
                "active_ingredients": [{"name": "EXAMPLEDRUG", "strength": "10MG"}],
            }
        ],
        "submissions": [
            {"submission_type": "SUPPL", "submission_number": "1", "submission_status_date": "20220101"},
            {"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "19981217"},
        ],
    }


def _snapshot(*records: dict) -> MapSnapshot:
    from fdagym import queries

    query = queries.approvals_search("exampledrug")
    body = {"meta": {"results": {"total": len(records)}}, "results": list(records)}
    return MapSnapshot({(query.endpoint, query.search, query.count, query.limit): body})


def test_parser_accepts_the_written_forms_and_rejects_drug_names():
    assert parse_application_number("NDA020977") == ("NDA", "020977")
    assert parse_application_number("NDA 020977") == ("NDA", "020977")
    assert parse_application_number("nda-020977") == ("NDA", "020977")
    assert parse_application_number("020977") == (None, "020977")
    assert parse_application_number("20977") == (None, "020977")
    assert parse_application_number("BLA 125057") == ("BLA", "125057")
    assert parse_application_number("anda075971") == ("ANDA", "075971")
    assert parse_application_number("methadone") is None
    assert parse_application_number("NDA") is None
    assert parse_application_number("123") is None


def test_schema_advertises_application_number_and_still_has_five_tools():
    assert [tool["function"]["name"] for tool in TOOLS] == [
        "search_drug_label",
        "get_label_section",
        "query_adverse_events",
        "search_approvals",
        "submit_answer",
    ]
    schema = next(tool for tool in TOOLS if tool["function"]["name"] == "search_approvals")
    params = schema["function"]["parameters"]
    assert set(params["properties"]) == {"generic_name", "application_number", "sponsor_name", "page"}
    assert params["required"] == []
    assert params["properties"]["page"]["minimum"] == 1
    description = schema["function"]["description"]
    assert "application_number" in description
    assert "NDA020977" in description
    assert "020977" in description
    assert "page" in description


def test_application_number_variants_return_that_record_and_do_not_query_by_name():
    snap = _snapshot(_record("NDA020977", sponsor="VIIV HLTHCARE", brand="ZIAGEN"), _record("ANDA075971"))
    for raw in ("NDA020977", "NDA 020977", "nda-020977", "020977"):
        output, error = execute(EpisodeState(snap), "search_approvals", {"application_number": raw}, step_index=1)
        assert error is None, raw
        assert [row["application_number"] for row in output["results"]] == ["NDA020977"]
        assert output["results"][0]["sponsor_name"] == "VIIV HLTHCARE"
        assert output["results"][0]["brand"] == ["ZIAGEN"]
        assert output["meta"]["results"]["total"] == 1
        kinds = [sub["submission_type"] for sub in output["results"][0]["submissions"]]
        assert kinds == ["ORIG"]
        assert output["results"][0]["n_supplements"] == 1
        assert output["meta"]["results"]["page"] == 1
    assert snap.calls == []


def test_bla_prefix_and_bare_digits_do_not_cross_match():
    snap = _snapshot(_record("BLA125057", sponsor="BIOLOGICS", brand="BIOBRAND", form="INJECTION"))
    output, error = execute(EpisodeState(snap), "search_approvals", {"application_number": "BLA 125057"}, step_index=1)
    assert error is None
    assert output["results"][0]["application_number"] == "BLA125057"
    assert output["results"][0]["products"][0]["dosage_form"] == "INJECTION"

    missed, error = execute(EpisodeState(snap), "search_approvals", {"application_number": "NDA125057"}, step_index=1)
    assert error is None
    assert missed["results"] == []
    assert missed["meta"]["results"]["total"] == 0


def test_application_number_and_generic_name_together_are_rejected():
    snap = _snapshot(_record("NDA020977", brand="ZIAGEN"))
    output, error = execute(
        EpisodeState(snap),
        "search_approvals",
        {"generic_name": "methadone", "application_number": "NDA020977"},
        step_index=1,
    )
    assert output["error"] == error
    assert "exactly one" in error
    assert snap.calls == []


def test_bad_and_missing_arguments_are_observations():
    snap = _snapshot(_record("NDA020977"))
    bad, error = execute(EpisodeState(snap), "search_approvals", {"application_number": "methadone"}, step_index=1)
    assert bad["error"] == error
    assert "application_number" in error

    neither, error = execute(EpisodeState(snap), "search_approvals", {}, step_index=1)
    assert neither["error"] == error
    assert "exactly one" in error


def test_generic_name_search_puts_the_nda_before_the_anda_and_keeps_only_the_original():
    snap = _snapshot(_record("ANDA000001"), _record("NDA000002"))
    output, error = execute(EpisodeState(snap), "search_approvals", {"generic_name": "exampledrug"}, step_index=1)
    assert error is None
    assert [row["application_number"] for row in output["results"]] == ["NDA000002", "ANDA000001"]
    assert [sub["submission_type"] for sub in output["results"][0]["submissions"]] == ["ORIG"]
    assert output["meta"]["results"]["total"] == 2
    assert snap.calls


def _approval_tasks() -> list[dict]:
    rows = []
    for name in ("train.jsonl", "test.jsonl"):
        path = TASKS / name
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            if task.get("type") == "approval":
                rows.append(task)
    return rows


@pytest.fixture(scope="module")
def pilot():
    if not (SNAPSHOT / "manifest.jsonl").exists():
        pytest.skip("pilot snapshot absent")
    return DiskSnapshot("2026-10-15"), _approval_tasks()


def test_fg0002_application_search_scores_one(pilot):
    snapshot, tasks = pilot
    task = next(row for row in tasks if row["id"] == "FG-0002")
    env = FDAEnv(snapshot, tasks, run_id="fg-0002")
    env.reset(task["id"])
    obs, _reward, done, _info = env.step(
        {"tool": "search_approvals", "arguments": {"application_number": "NDA 020977"}}
    )
    assert done is False
    assert obs["error"] is None
    assert obs["output"]["results"][0]["application_number"] == "NDA020977"
    assert "ZIAGEN" in obs["output"]["results"][0]["brand"]
    _obs, _reward, done, info = env.step({"tool": "submit_answer", "arguments": gold_submit(task, env.steps)})
    assert done is True
    assert info["reward"]["total"] == 1.0
    assert info["reward"]["correctness"] == 1.0
    assert info["reward"]["citation"] == 1.0
