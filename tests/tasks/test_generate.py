import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from fdagym.queries import approvals_search, faers_report_count, faers_top_reactions, label_search
from fdagym.tasks.config import GenConfig
from fdagym.tasks.difficulty import tier
from fdagym.tasks.generators import _candidates, generate
from fdagym.tasks.meddra import aliases
from fdagym.tasks.plan import absence_year, probe_year
from tests.tasks.fake_snapshot import MemorySnapshot

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "task.schema.json").read_text())
REGISTRY = Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))
VALIDATOR = Draft202012Validator(SCHEMA, registry=REGISTRY)


def test_every_generated_task_matches_the_schema(tasks):
    from fdagym.tasks.validate import require_oracle_calls

    assert tasks
    for task in tasks:
        VALIDATOR.validate(task)
        require_oracle_calls(task)


def test_tier_is_computed_and_pass_rate_is_unset(tasks):
    for task in tasks:
        assert task["difficulty"] == tier(task["difficulty_features"])
        assert task["difficulty_features"]["sources"] == len(task["sources"])
        assert task["base_pass_rate"] is None
        assert task["difficulty_features"]["min_tool_calls"] >= 1


def test_no_entity_appears_in_both_splits(tasks):
    train = {task["entity"]["generic_name"] for task in tasks if task["split"] == "train"}
    test = {task["entity"]["generic_name"] for task in tasks if task["split"] == "test"}
    assert train and test
    assert train.isdisjoint(test)
    assert abs(len(test) / (len(train) + len(test)) - 0.2) < 0.01


def test_unanswerable_share_is_a_cap_not_a_quota(tasks):
    for split in ("train", "test"):
        group = [task for task in tasks if task["split"] == split]
        share = sum(not task["answerable"] for task in group) / len(group)
        assert share <= 0.07
        assert all(task["gold"]["value"] == "NOT_FOUND" for task in group if not task["answerable"])


def test_unanswerable_tasks_match_an_empty_union_result(tasks, snapshot):
    from fdagym.queries import faers_top_reactions

    for task in tasks:
        if task["answerable"]:
            continue
        arguments = task["oracle_calls"][0]["arguments"]
        serious = True if arguments.get("serious") is True else None
        query = faers_top_reactions(arguments["generic_name"], serious=serious, year=arguments.get("year"))
        body = snapshot.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
        assert body.get("results") == []
        assert body["meta"]["results"]["total"] == 0


def test_brand_name_is_omitted_when_it_is_the_generic(tasks):
    for task in tasks:
        brand = task["entity"].get("brand_name")
        if brand is not None:
            assert brand.strip().lower() != task["entity"]["generic_name"].strip().lower()


def test_snapshot_version_comes_from_the_snapshot(tasks, snapshot):
    assert snapshot.version == "2026-10-15"
    assert {task["snapshot_version"] for task in tasks} == {"2026-10-15"}


def test_faers_citations_use_the_aggregate_fields(tasks):
    for task in tasks:
        for citation in task["gold_citations"]:
            if citation["endpoint"] != "drug/event":
                continue
            assert citation["ids"] == []
            if task["type"] == "faers_top_reaction" and task["answerable"]:
                assert citation["fields"] == ["results"]
            else:
                assert citation["fields"] == ["meta.results.total"]


def test_gold_aliases_come_from_the_meddra_map(tasks):
    tops = [task for task in tasks if task["type"] == "faers_top_reaction" and task["answerable"]]
    assert tops
    for task in tops:
        assert task["gold"].get("aliases", []) == aliases(task["gold"]["value"])


def test_questions_never_say_suspect_drug(tasks):
    for task in tasks:
        assert "suspect" not in task["question"].lower()


def test_label_open_stays_at_most_about_a_tenth(tasks):
    """Easy templates grow the pool. Judge tasks stay capped near 10%, not padded back up."""
    for split in ("train", "test"):
        group = [task for task in tasks if task["split"] == split]
        share = sum(task["type"] == "label_open" for task in group) / len(group)
        assert share <= 0.12


def test_warfarin_boxed_warning_cites_every_agreeing_label(tasks):
    boxed = [
        task
        for task in tasks
        if task["type"] == "boxed_warning"
        and task["entity"]["generic_name"] == "warfarin"
        and task["gold"]["value"] is True
        and "keywords" in task["gold"]
    ]
    assert boxed
    assert len(boxed[0]["gold_citations"][0]["ids"]) > 1
    assert "bleeding" in boxed[0]["gold"]["keywords"]


def test_unanimous_boxed_yes_no_is_easy_and_includes_a_no(tasks):
    plain = [
        task
        for task in tasks
        if task["type"] == "boxed_warning" and task["question"].endswith("label carry a boxed warning?")
    ]
    assert plain
    for task in plain:
        assert task["difficulty"] == "easy"
        assert task["gold"]["kind"] == "boolean"
        assert "keywords" not in task["gold"]
        assert task["gold_citations"][0]["fields"] == ["sections"]
        assert task["gold_citations"][0]["ids"]
        assert task["difficulty_features"]["needs_disambiguation"] is False
        assert task["difficulty_features"]["needs_aggregation"] is False
    assert any(task["gold"]["value"] is False for task in plain)
    assert any(task["gold"]["value"] is True for task in plain)


def test_named_application_facts_are_easy(tasks):
    facts = [task for task in tasks if task["question"].startswith("Who is the sponsor of application ")]
    brands = [task for task in tasks if task["question"].startswith("What brand name is marketed under application ")]
    forms = [
        task
        for task in tasks
        if task["question"].startswith("What is the dosage form") or task["question"].startswith("What is the route")
    ]
    assert facts and brands and forms
    for task in facts + brands + forms:
        assert task["difficulty"] == "easy"
        assert task["type"] == "approval"
        assert task["gold"]["kind"] == "exact"
        assert task["difficulty_features"]["min_tool_calls"] == 1
        assert task["difficulty_features"]["needs_disambiguation"] is False
    assert facts[0]["gold_citations"][0]["fields"] == ["sponsor_name"]
    assert brands[0]["gold_citations"][0]["fields"] == ["products"]


def test_pilot_tasks_resolve_without_snapshot_miss():
    import pytest

    from fdagym.snapshot import SnapshotMiss
    from fdagym.tasks.plan import queries_for_oracle

    root = Path("data/snapshots/2026-10-15")
    if not (root / "manifest.jsonl").exists():
        pytest.skip("pilot snapshot absent")
    from fdagym.openfda.store import DiskSnapshot

    snapshot = DiskSnapshot("2026-10-15")
    rows = []
    for name in ("train.jsonl", "test.jsonl"):
        path = Path("data/tasks") / name
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    assert rows
    for task in rows:
        for query in queries_for_oracle(task["oracle_calls"]):
            try:
                snapshot.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
            except SnapshotMiss as exc:
                raise AssertionError(f"{task['id']} missed {query}") from exc


def test_queries_follow_oracle_calls_not_question_text():
    from fdagym.tasks.plan import queries_for_oracle

    calls = [{"tool": "search_approvals", "arguments": {"generic_name": "warfarin"}}]
    assert queries_for_oracle(calls) == [approvals_search("warfarin")]


def test_boxed_yes_no_is_balanced_within_each_split(tasks):
    for split in ("train", "test"):
        group = [
            task
            for task in tasks
            if task["split"] == split and task["type"] == "boxed_warning" and "keywords" not in task["gold"]
        ]
        yes = sum(task["gold"]["value"] is True for task in group)
        no = sum(task["gold"]["value"] is False for task in group)
        if yes and no:
            assert yes == no


def test_balance_drops_surplus_yes_and_keeps_a_split_with_no_nos():
    from fdagym.tasks.split import balance_boxed_yes_no

    def boxed(split, drug, value):
        return {
            "type": "boxed_warning",
            "split": split,
            "question": f"Does {drug}'s label carry a boxed warning?",
            "entity": {"generic_name": drug},
            "gold": {"kind": "boolean", "value": value},
        }

    tasks = [
        boxed("train", "zeta", True),
        boxed("train", "alpha", True),
        boxed("train", "atorvastatin", False),
        {"type": "approval", "split": "train", "question": "keep me", "entity": {"generic_name": "zeta"}, "gold": {}},
        boxed("test", "semaglutide", True),
    ]
    kept = balance_boxed_yes_no(tasks)
    train = [task for task in kept if task["split"] == "train" and task["type"] == "boxed_warning"]
    assert [task["entity"]["generic_name"] for task in train] == ["alpha", "atorvastatin"]
    assert any(task["question"] == "keep me" for task in kept)
    assert any(task["entity"]["generic_name"] == "semaglutide" for task in kept)


def test_absent_boxed_warning_is_cited_from_the_sections_list():
    drug = "exampledrug"
    year = probe_year(drug)
    gap = absence_year(drug)
    mapping = {}
    _put(
        mapping,
        label_search(drug),
        {
            "results": [
                {
                    "set_id": "set-1",
                    "contraindications": ["Do not use during pregnancy."],
                    "openfda": {
                        "generic_name": ["exampledrug"],
                        "brand_name": ["Example"],
                        "product_type": ["HUMAN PRESCRIPTION DRUG"],
                    },
                }
            ]
        },
    )
    _put(
        mapping,
        approvals_search(drug),
        {
            "results": [
                {
                    "application_number": "NDA000002",
                    "sponsor_name": "EXAMPLE",
                    "products": [
                        {
                            "brand_name": "EXAMPLE",
                            "dosage_form": "TABLET",
                            "route": "ORAL",
                            "active_ingredients": [{"name": "EXAMPLEDRUG"}],
                        }
                    ],
                    "submissions": [
                        {"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "19950301"}
                    ],
                }
            ]
        },
    )
    _put(mapping, faers_top_reactions(drug), {"results": [{"term": "NAUSEA", "count": 4}, {"term": "HEADACHE", "count": 1}]})
    _put(
        mapping,
        faers_top_reactions(drug, serious=True, year=year),
        {"results": [{"term": "NAUSEA", "count": 3}, {"term": "HEADACHE", "count": 1}]},
    )
    _put(
        mapping,
        faers_top_reactions(drug, serious=True, year=gap),
        {"results": [{"term": "NAUSEA", "count": 3}, {"term": "HEADACHE", "count": 1}]},
    )
    _put(mapping, faers_report_count(drug), {"meta": {"results": {"total": 80}}, "results": []})
    _put(mapping, faers_report_count(drug, serious=True, year=year), {"meta": {"results": {"total": 12}}, "results": []})
    tasks = generate(MemorySnapshot(mapping, "toy"), GenConfig(), drugs=[drug])
    easy = next(task for task in tasks if task["question"] == f"Does {drug}'s label carry a boxed warning?")
    assert easy["gold"]["value"] is False
    assert easy["gold_citations"][0]["fields"] == ["sections"]
    assert easy["oracle_calls"] == [{"tool": "search_drug_label", "arguments": {"generic_name": drug}}]
    composite = next(task for task in tasks if task["type"] == "composite")
    assert composite["gold"]["value"].endswith("no")
    assert composite["gold_citations"][1]["fields"] == ["sections"]
    assert [call["tool"] for call in composite["oracle_calls"]] == ["query_adverse_events", "search_drug_label"]


def _put(mapping, query, response):
    mapping[(query.endpoint, query.search, query.count, query.limit)] = response


def test_combination_application_cannot_set_the_approval_year():
    """Deleting the single-ingredient filter makes this gold 2001, the combo NDA."""
    drug = "exampledrug"
    year = probe_year(drug)
    gap = absence_year(drug)
    mapping = {}
    _put(
        mapping,
        label_search(drug),
        {
            "results": [
                {
                    "set_id": "set-1",
                    "boxed_warning": ["WARNING: BLEEDING RISK"],
                    "contraindications": ["Do not use during pregnancy."],
                    "openfda": {"generic_name": ["exampledrug"], "brand_name": ["Example"], "product_type": ["HUMAN PRESCRIPTION DRUG"]},
                }
            ]
        },
    )
    _put(
        mapping,
        approvals_search(drug),
        {
            "results": [
                {
                    "application_number": "NDA000001",
                    "products": [{"brand_name": "COMBO", "active_ingredients": [{"name": "EXAMPLEDRUG"}, {"name": "OTHER"}]}] ,
                    "submissions": [{"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "20010101"}],
                },
                {
                    "application_number": "NDA000002",
                    "products": [{"brand_name": "EXAMPLE", "active_ingredients": [{"name": "EXAMPLEDRUG"}]}],
                    "submissions": [{"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "19950301"}],
                },
            ]
        },
    )
    _put(mapping, faers_top_reactions(drug), {"results": [{"term": "DIARRHOEA", "count": 9}, {"term": "NAUSEA", "count": 2}]})
    _put(mapping, faers_top_reactions(drug, serious=True, year=year), {"meta": {"results": {"total": 0}}, "results": []})
    _put(
        mapping,
        faers_top_reactions(drug, serious=True, year=gap),
        {"results": [{"term": "NAUSEA", "count": 4}, {"term": "HEADACHE", "count": 1}]},
    )
    _put(mapping, faers_report_count(drug), {"meta": {"results": {"total": 80}}, "results": []})
    _put(mapping, faers_report_count(drug, serious=True, year=year), {"meta": {"results": {"total": 12}}, "results": [{"safetyreportid": "1"}]})
    snapshot = MemorySnapshot(mapping, "toy")
    tasks = generate(snapshot, GenConfig(), drugs=[drug])
    approval = next(task for task in tasks if task["type"] == "approval")
    assert approval["gold"]["value"] == 1995
    assert approval["gold_citations"][0]["ids"] == ["NDA000002"]
    assert approval["difficulty"] == "easy"
    top = next(task for task in tasks if task["type"] == "faers_top_reaction" and task["answerable"] and "serious" not in task["question"])
    assert top["gold"]["value"] == "DIARRHOEA"
    assert top["gold"]["aliases"] == ["DIARRHEA"]
    candidates = list(_candidates(snapshot, drug, GenConfig()))
    missing = next(task for task in candidates if task["answerable"] is False)
    assert missing["gold"]["value"] == "NOT_FOUND"
    assert str(year) in missing["question"]


def test_tied_top_reaction_is_not_emitted():
    drug = "exampledrug"
    year = probe_year(drug)
    gap = absence_year(drug)
    mapping = {}
    label = {
        "results": [
            {
                "set_id": "set-1",
                "boxed_warning": ["WARNING: BLEEDING RISK"],
                "contraindications": ["Do not use during pregnancy."],
                "openfda": {"generic_name": ["exampledrug"], "brand_name": ["Example"], "product_type": ["HUMAN PRESCRIPTION DRUG"]},
            }
        ]
    }
    _put(mapping, label_search(drug), label)
    _put(
        mapping,
        approvals_search(drug),
        {
            "results": [
                {
                    "application_number": "NDA000002",
                    "products": [{"brand_name": "EXAMPLE", "active_ingredients": [{"name": "EXAMPLEDRUG"}]}],
                    "submissions": [{"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "19950301"}],
                }
            ]
        },
    )
    _put(mapping, faers_top_reactions(drug), {"results": [{"term": "NAUSEA", "count": 5}, {"term": "FATIGUE", "count": 5}]})
    _put(mapping, faers_top_reactions(drug, serious=True, year=year), {"results": [{"term": "NAUSEA", "count": 3}, {"term": "FATIGUE", "count": 1}]})
    _put(mapping, faers_top_reactions(drug, serious=True, year=gap), {"results": [{"term": "NAUSEA", "count": 3}, {"term": "FATIGUE", "count": 1}]})
    _put(mapping, faers_report_count(drug), {"meta": {"results": {"total": 80}}, "results": []})
    _put(mapping, faers_report_count(drug, serious=True, year=year), {"meta": {"results": {"total": 12}}, "results": []})
    tasks = generate(MemorySnapshot(mapping, "toy"), GenConfig(), drugs=[drug])
    questions = [task["question"] for task in tasks if task["type"] == "faers_top_reaction"]
    assert not any("serious" not in question and "received" not in question for question in questions)


def test_generate_defaults_to_drugs_txt(monkeypatch):
    from fdagym.tasks.plan import load_universe

    seen: list[str] = []

    def fake_candidates(snapshot, drug, cfg):
        seen.append(drug)
        return iter(())

    monkeypatch.setattr("fdagym.tasks.generators._candidates", fake_candidates)
    generate(MemorySnapshot({}, "2026-10-15"), GenConfig())
    assert seen == list(load_universe())
    assert "ibuprofen" not in seen


def test_assign_splits_puts_unans_and_easy_no_on_test():
    from fdagym.tasks.split import assign_splits

    def rows(*, n_unans: int = 0, easy_no: bool = False, difficulty: str = "easy"):
        tasks = [
            {
                "type": "faers_top_reaction",
                "answerable": False,
                "difficulty": "hard",
                "gold": {"value": "NOT_FOUND"},
            }
            for _ in range(n_unans)
        ]
        tasks.append({"type": "approval", "answerable": True, "difficulty": difficulty, "gold": {"value": 1995}})
        if easy_no:
            tasks.append(
                {
                    "type": "boxed_warning",
                    "answerable": True,
                    "difficulty": "easy",
                    "gold": {"kind": "boolean", "value": False},
                }
            )
        return tasks

    by_drug = {f"d{i:02d}": rows() for i in range(10)}
    by_drug["u-heavy"] = rows(n_unans=8)
    by_drug["u-mid"] = rows(n_unans=6)
    by_drug["u-light"] = rows(n_unans=2)
    by_drug["no-a"] = rows(easy_no=True)
    by_drug["no-b"] = rows(easy_no=True)
    splits = assign_splits(by_drug, GenConfig())
    test = {name for name, split in splits.items() if split == "test"}
    train = {name for name, split in splits.items() if split == "train"}
    unans = {"u-heavy", "u-mid", "u-light"}
    assert test & unans
    assert train & unans
    assert test & {"no-a", "no-b"}
    assert abs(len(test) / (len(train) + len(test)) - 0.2) < 0.05


def test_collision_windows_are_not_emitted_from_the_snapshot():
    import pytest

    root = Path("data/snapshots/2026-10-15")
    if not (root / "manifest.jsonl").exists():
        pytest.skip("pilot snapshot absent")
    from fdagym.openfda.store import DiskSnapshot

    snapshot = DiskSnapshot("2026-10-15")
    cfg = GenConfig()
    blocked = {
        ("aprocitentan", 2021),
        ("resmetirom", 2020),
        ("resmetirom", 2021),
        ("resmetirom", 2022),
    }
    got = set()
    for drug in ("aprocitentan", "resmetirom", "xanomeline"):
        for task in _candidates(snapshot, drug, cfg):
            if task["answerable"]:
                continue
            year = task["oracle_calls"][0]["arguments"].get("year")
            got.add((drug, year))
    assert got.isdisjoint(blocked)
    assert any(drug == "xanomeline" for drug, _year in got)


def test_shipped_sheet_follows_the_150_drug_contract():
    from fdagym.tasks.plan import load_universe

    rows = []
    for name in ("train.jsonl", "test.jsonl"):
        path = Path("data/tasks") / name
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    universe = set(load_universe())
    names = {task["entity"]["generic_name"] for task in rows}
    assert rows
    assert "ibuprofen" not in names
    assert names <= universe
    assert {task["snapshot_version"] for task in rows} == {"2026-10-15"}
    train_names = {task["entity"]["generic_name"] for task in rows if task["split"] == "train"}
    test_names = {task["entity"]["generic_name"] for task in rows if task["split"] == "test"}
    assert train_names and test_names and train_names.isdisjoint(test_names)
    for split in ("train", "test"):
        group = [task for task in rows if task["split"] == split]
        share = sum(not task["answerable"] for task in group) / len(group)
        assert 0.03 <= share <= 0.07, (split, share, len(group))
    test_no = [
        task
        for task in rows
        if task["split"] == "test"
        and task["type"] == "boxed_warning"
        and "keywords" not in task["gold"]
        and task["gold"]["value"] is False
    ]
    assert test_no
    assert all(task.get("oracle_calls") for task in rows)
