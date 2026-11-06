"""Hard training split: disjoint from hard eval and TEST, reachable, opt-in 60/40 mix."""

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from fdagym.agents.baseline import load_split
from fdagym.schemas import load as load_schema
from fdagym.tasks.disjoint import OverlapError, assert_disjoint
from fdagym.tasks.hard_solver import (
    Unsolvable,
    parse_question,
    question_earliest_reaction,
    question_none_yes_no,
    question_year_reaction,
    solve,
)
from fdagym.tasks.hard_train import ANSWER_CAP, NAMED_CAP
from fdagym.tasks.reachability import check_reachability
from fdagym.tasks.validate import require_oracle_calls
from fdagym.train.curriculum import in_training_pool, type_mix_pool
from fdagym.train.hard_mix import HARD_TRAIN_MIX, mix_hard_train
from fdagym.train.tool_config import ToolTrainConfig

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = load_schema("task.schema.json")
VALIDATOR = Draft202012Validator(
    SCHEMA, registry=Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))
)
HARD_TRAIN_SHA256 = "e43533ef78f8c68b449579832b4dc59e7b8b1f6e07c657bf19a47ba573a585e1"
HARD_TRAIN_HOLDOUT_SHA256 = "2e7f9f538e561d3c0def58d4288f9953ac11a13c697fa61eedf95a70341565c9"
TRAIN_SHA256 = "940c3a333b19570c2097a5d6b39dd17a91e2615474fa2e7e6b831a850d01e0c0"
TEST_SHA256 = "8731d64d4c7de81914ec0c37d96cc49afa3698e116c01d3f2a4c02c99efa7f1e"
N_TRAIN = 334
N_HOLDOUT = 36
N_TASKS = N_TRAIN + N_HOLDOUT
ID_START = 8101
BY_VARIANT = {
    "boxed_none": 3,
    "earliest_boxed": 40,
    "earliest_reaction": 40,
    "faers_reaction": 40,
    "faers_year": 38,
    "sponsor_count": 65,
    "sponsor_drug": 34,
    "sponsor_faers": 36,
    "year_reaction": 38,
}
HOLDOUT_BY_VARIANT = {
    "boxed_none": 1,
    "earliest_boxed": 4,
    "earliest_reaction": 4,
    "faers_reaction": 4,
    "faers_year": 4,
    "sponsor_count": 7,
    "sponsor_drug": 4,
    "sponsor_faers": 4,
    "year_reaction": 4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def disk():
    from fdagym.openfda import DiskSnapshot

    return DiskSnapshot("2026-10-15")


@pytest.fixture(scope="module")
def hard_train():
    return load_split("hard_train")


def test_train_and_test_bytes_did_not_move():
    assert _sha256(ROOT / "data" / "tasks" / "train.jsonl") == TRAIN_SHA256
    assert _sha256(ROOT / "data" / "tasks" / "test.jsonl") == TEST_SHA256
    assert _sha256(ROOT / "data" / "tasks" / "hard_train.jsonl") == HARD_TRAIN_SHA256
    assert _sha256(ROOT / "data" / "tasks" / "hard_train_holdout.jsonl") == HARD_TRAIN_HOLDOUT_SHA256


def test_file_matches_schema_and_stays_off_the_eval_ids(hard_train):
    holdout = load_split("hard_train_holdout")
    assert len(hard_train) == N_TRAIN
    assert len(holdout) == N_HOLDOUT
    ids = sorted(task["id"] for task in [*hard_train, *holdout])
    assert ids == [f"FG-{ID_START + offset:04d}" for offset in range(N_TASKS)]
    assert set(ids).isdisjoint(task["id"] for task in load_split("public"))
    assert set(ids).isdisjoint(task["id"] for task in load_split("train"))
    assert set(ids).isdisjoint(task["id"] for task in load_split("test"))
    variants = []
    for task in hard_train:
        VALIDATOR.validate(task)
        require_oracle_calls(task)
        spec = parse_question(task["question"])
        variants.append(spec["variant"])
        assert task["split"] == "hard_train"
        assert task["type"] in {"compare", "sponsor_hop"}
        assert task["answerable"] is True
        assert task["gold"]["value"] != "NOT_FOUND"
        assert task["snapshot_version"] == "2026-10-15"
        assert task["entity"]["generic_name"] in spec["drugs"]
        calls = task["difficulty_features"]["min_tool_calls"]
        assert calls == len(task["oracle_calls"])
        assert 3 <= calls <= 8
        if spec["variant"] == "boxed_none":
            assert task["gold"] == {"kind": "boolean", "value": False}
        if spec["variant"] == "sponsor_count":
            assert task["gold"]["kind"] == "number"
            assert isinstance(task["gold"]["value"], int)
        if spec["variant"] in {"sponsor_faers", "sponsor_drug"}:
            tools = {call["tool"] for call in task["oracle_calls"]}
            assert tools.isdisjoint({"search_drug_label", "get_label_section"})
    assert {variant: variants.count(variant) for variant in BY_VARIANT} == BY_VARIANT
    assert sum(task["type"] == "compare" for task in hard_train) == 199
    assert sum(task["type"] == "sponsor_hop" for task in hard_train) == 135
    holdout_variants = []
    for task in holdout:
        VALIDATOR.validate(task)
        require_oracle_calls(task)
        spec = parse_question(task["question"])
        holdout_variants.append(spec["variant"])
        assert task["split"] == "hard_train_holdout"
        if spec["variant"] in {"sponsor_faers", "sponsor_drug"}:
            tools = {call["tool"] for call in task["oracle_calls"]}
            assert tools.isdisjoint({"search_drug_label", "get_label_section"})
    assert {variant: holdout_variants.count(variant) for variant in HOLDOUT_BY_VARIANT} == HOLDOUT_BY_VARIANT


def test_rewritten_compare_templates_match_committed_questions(hard_train):
    rebuilt = 0
    for task in [*hard_train, *load_split("hard_train_holdout")]:
        spec = parse_question(task["question"])
        if spec["variant"] == "year_reaction":
            assert task["question"] == question_year_reaction(spec["drugs"], spec["year"])
            rebuilt += 1
        elif spec["variant"] == "earliest_reaction":
            assert task["question"] == question_earliest_reaction(spec["drugs"])
            rebuilt += 1
        elif spec["variant"] == "boxed_none":
            assert task["question"] == question_none_yes_no(spec["drugs"])
            rebuilt += 1
    expected = sum(
        BY_VARIANT[variant] + HOLDOUT_BY_VARIANT[variant]
        for variant in ("year_reaction", "earliest_reaction", "boxed_none")
    )
    assert rebuilt == expected


def test_sponsor_names_stay_under_the_cap(hard_train):
    from collections import Counter

    named = Counter()
    answers = Counter()
    for task in [*hard_train, *load_split("hard_train_holdout")]:
        spec = parse_question(task["question"])
        if spec["variant"] not in {"sponsor_faers", "sponsor_drug", "sponsor_count"}:
            continue
        for drug in spec["drugs"]:
            named[drug] += 1
        answers[task["entity"]["generic_name"]] += 1
    assert named
    assert named.most_common(1)[0][1] <= NAMED_CAP
    assert answers.most_common(1)[0][1] <= ANSWER_CAP


def test_compare_unfiltered_faers_rank_is_not_identifying(disk, hard_train):
    from collections import Counter, defaultdict

    from fdagym.queries import faers_report_count
    from fdagym.tasks.hard import _Cache, _faers_join_winner, _unique_max_value

    cache = _Cache(disk)
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    totals_of: dict[str, int] = {}

    def all_years(drug: str) -> int:
        if drug not in totals_of:
            body = cache.query(faers_report_count(drug))
            totals_of[drug] = int(((body.get("meta") or {}).get("results") or {}).get("total") or 0)
        return totals_of[drug]

    for task in [*hard_train, *load_split("hard_train_holdout")]:
        spec = parse_question(task["question"])
        variant = spec["variant"]
        if variant not in {"faers_year", "faers_reaction", "year_reaction"}:
            continue
        winner = _faers_join_winner(cache, spec, variant, task["gold"]["value"])
        values = [all_years(drug) for drug in spec["drugs"]]
        answer_n = all_years(winner)
        if _unique_max_value(answer_n, values):
            rank = "max"
        elif answer_n == min(values) and values.count(min(values)) == 1:
            rank = "min"
        elif values.count(answer_n) == 1:
            rank = "middle"
        else:
            rank = "tie"
        by_variant[variant][rank] += 1
    assert by_variant
    for variant, ranks in by_variant.items():
        n = sum(ranks.values())
        if n < 5:
            continue
        top = ranks.most_common(1)[0]
        assert top[1] / n < 0.70, (variant, dict(ranks))


def test_sponsor_keyword_hits_a_named_distractor(disk, hard_train):
    from fdagym.tasks.hard import _Cache, _profile
    from fdagym.tools import SECTION_LIMIT

    cache = _Cache(disk)
    seen: dict[str, str] = {}
    unique = []
    for task in [*hard_train, *load_split("hard_train_holdout")]:
        spec = parse_question(task["question"])
        if spec["variant"] not in {"sponsor_faers", "sponsor_drug"}:
            continue
        keyword = spec["keyword"]
        hits = []
        for drug in spec["drugs"]:
            if drug not in seen:
                profile = _profile(disk, cache, drug)
                seen[drug] = (profile.boxed_text or "")[:SECTION_LIMIT].casefold()
            if keyword in seen[drug]:
                hits.append(drug)
        if len(hits) < 2:
            unique.append((task["id"], keyword, spec["drugs"], hits))
    assert unique == []


def test_real_file_is_disjoint_and_a_planted_overlap_is_loud(disk, hard_train):
    from fdagym.tasks.hard_train import load_exclusion_splits

    public = load_split("public")
    holdout = load_split("hard_train_holdout")
    excluded = load_exclusion_splits()
    report = assert_disjoint(hard_train, public, disk)
    assert report["overlap"] == {"drugs": [], "applications": [], "sponsors": [], "questions": []}
    assert report["train"]["drugs"] == 68
    assert report["holdout"]["drugs"] == 62
    assert report["train_tasks"] == N_TRAIN
    assert_disjoint(holdout, public, disk)
    assert_disjoint(hard_train, excluded, disk)
    assert_disjoint(holdout, excluded, disk)

    leaked = json.loads(json.dumps(public[0]))
    leaked["split"] = "hard_train"
    with pytest.raises(OverlapError, match="drugs"):
        assert_disjoint([leaked], public, disk)

    tainted = json.loads(json.dumps(hard_train[0]))
    tainted["question"] = tainted["question"] + " Cited application NDA016012."
    with pytest.raises(OverlapError, match="NDA016012"):
        assert_disjoint([tainted], public, disk)

    sponsor_task = next(task for task in public if "sponsor_name" in str(task.get("gold_citations")))
    forged = json.loads(json.dumps(hard_train[0]))
    forged["question"] = "Who is the sponsor of application NDA999999?"
    forged["entity"] = {"generic_name": "olezarsen"}
    forged["gold"] = {"kind": "exact", "value": "ENDO OPERATIONS"}
    forged["gold_citations"] = [{"endpoint": "drug/drugsfda", "ids": ["NDA007337"], "fields": ["sponsor_name"]}]
    forged["oracle_calls"] = [{"tool": "search_approvals", "arguments": {"application_number": "NDA999999"}}]
    with pytest.raises(OverlapError, match="endo operations"):
        assert_disjoint([forged], [sponsor_task], disk)


def test_default_training_pool_still_refuses_the_file(hard_train):
    assert all(not in_training_pool(task) for task in hard_train)
    leaked = json.loads(json.dumps(hard_train[0]))
    leaked["split"] = "train"
    assert leaked["type"] in {"compare", "sponsor_hop"}
    assert not in_training_pool(leaked)
    rows, _mix = type_mix_pool([*load_split("train"), *hard_train], n_rows=40)
    assert {task["id"] for task in rows}.isdisjoint(task["id"] for task in hard_train)


def test_sixty_forty_mix_is_opt_in(hard_train):
    assert HARD_TRAIN_MIX == {"hard": 0.60, "current": 0.40}
    assert ToolTrainConfig().hard_train_mix is None
    jobs = (ROOT / "fdagym" / "train" / "jobs.py").read_text()
    cli = (ROOT / "scripts" / "run_gpu.py").read_text()
    # Unflagged train-tools stays at 0 (no mix).
    assert "hard_train_mix: float = 0.0" in jobs
    assert 'train.add_argument("--hard-train-mix", type=float, default=0.0)' in cli
    rows, report = mix_hard_train(load_split("train"), hard_train, n_rows=100, seed=0)
    assert report["rows"] == {"hard": 60, "current": 40}
    assert sum(task["split"] == "hard_train" for task in rows) == 60
    assert all(task["split"] != "hard" for task in rows)
    with pytest.raises(RuntimeError, match="eval split"):
        mix_hard_train(load_split("train"), load_split("public"), n_rows=10)


def test_question_only_solver_reaches_every_task_at_6_and_8(disk, hard_train):
    both = [*hard_train, *load_split("hard_train_holdout")]
    report = check_reachability(disk, both, [6, 8])
    assert report["min_calls"] == {"3": 90, "4": 164, "5": 72, "6": 44}
    # Named-sponsor oracles are 4; original_approval on the summary keeps year hops at 6.
    assert [row["solved"] for row in report["by_budget"]] == [N_TASKS, N_TASKS]
    assert report["budgets"] == [6, 8]
    need = max(task["difficulty_features"]["min_tool_calls"] for task in hard_train)
    longest = next(task for task in hard_train if task["difficulty_features"]["min_tool_calls"] == need)
    solved = solve(disk, longest["question"], budget=8)
    assert solved.n_calls == need
    with pytest.raises(Unsolvable, match="budget"):
        solve(disk, longest["question"], budget=need - 1)
