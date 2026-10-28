"""Hard eval split: schema, unique gold, question-only reachability, and no training leak."""

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.tasks.hard import (
    ANSWER_CAP,
    NAMED_CAP,
    Profile,
    _needs_year_compare,
    _one,
    _unfiltered_faers_max,
    _unfiltered_faers_rank,
    _unique_max_value,
    generate_hard,
    snapshot_answer,
)
from fdagym.tasks.hard_solver import (
    Unsolvable,
    parse_question,
    question_earliest,
    question_earliest_reaction,
    question_faers_year,
    question_none,
    question_none_yes_no,
    question_reaction,
    question_sponsor_drug,
    question_sponsor_faers,
    question_year_reaction,
    same_answer,
    solve,
)
from fdagym.tasks.hard_sponsor import (
    _anchor_options,
    _labels_within,
    _next_slot,
    _set_is_symmetric,
    _shared_keyword,
)
from fdagym.tasks.validate import require_oracle_calls

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "task.schema.json").read_text())
VALIDATOR = Draft202012Validator(
    SCHEMA, registry=Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))
)
TRAIN_SHA256 = "a6c895f14c3399cfd47de5b0d2ac311bc38e2f853f7de13f24343d588d739a31"
TEST_SHA256 = "3b52b30c45e19956bf183799f51a7e811a2f981a2777413f6fced837a82ce3ea"
QUOTAS = {
    "earliest_boxed": 16,
    "faers_year": 12,
    "faers_reaction": 18,
    "boxed_none": 4,
    "sponsor_faers": 25,
    "sponsor_drug": 25,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _play(snapshot, task, solution) -> dict:
    env = FDAEnv(snapshot, [task], run_id="hard-test", seed=0)
    env.reset(task["id"])
    for call in solution.calls:
        _obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            return env.trajectory["reward"]
    env.step({"tool": "submit_answer", "arguments": {"answer": solution.answer, "citations": solution.citations}})
    return env.trajectory["reward"]


@pytest.fixture(scope="module")
def disk():
    from fdagym.openfda import DiskSnapshot

    return DiskSnapshot("2026-10-15")


@pytest.fixture(scope="module")
def hard_tasks():
    path = ROOT / "data" / "tasks" / "hard.jsonl"
    if not path.exists():
        pytest.skip("data/tasks/hard.jsonl stays in the private repo")
    return load_split("hard")


def test_train_and_test_files_are_byte_identical():
    assert _sha256(ROOT / "data" / "tasks" / "train.jsonl") == TRAIN_SHA256
    assert _sha256(ROOT / "data" / "tasks" / "test.jsonl") == TEST_SHA256


def test_hard_file_matches_the_schema_and_stays_out_of_the_old_ids(hard_tasks):
    assert len(hard_tasks) == 100
    ids = [task["id"] for task in hard_tasks]
    assert ids == [f"FG-{8001 + offset:04d}" for offset in range(100)]
    train_ids = {task["id"] for task in load_split("train")}
    test_ids = {task["id"] for task in load_split("test")}
    assert set(ids).isdisjoint(train_ids)
    assert set(ids).isdisjoint(test_ids)
    variants = []
    questions = []
    named: set[str] = set()
    for task in hard_tasks:
        VALIDATOR.validate(task)
        require_oracle_calls(task)
        spec = parse_question(task["question"])
        variants.append(spec["variant"])
        questions.append(task["question"])
        named.update(spec["drugs"])
        assert task["split"] == "hard"
        assert task["type"] in {"compare", "sponsor_hop"}
        assert task["answerable"] is True
        assert task["gold"]["value"] != "NOT_FOUND"
        assert task["snapshot_version"] == "2026-10-15"
        assert task["base_pass_rate"] is None
        assert 1 <= task["difficulty_features"]["min_tool_calls"] <= 6
        assert task["entity"]["generic_name"] in spec["drugs"]
        assert all("," not in drug for drug in spec["drugs"])
        if spec["variant"] == "boxed_none":
            assert task["difficulty"] == "medium"
            assert task["gold"] == {"kind": "boolean", "value": False}
        else:
            assert task["difficulty"] == "hard"
        if spec["variant"] == "faers_year":
            event = [row for row in task["gold_citations"] if row["endpoint"] == "drug/event"]
            assert event == [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}]
        if spec["variant"] in {"faers_reaction", "sponsor_faers"}:
            event = [row for row in task["gold_citations"] if row["endpoint"] == "drug/event"]
            assert event and event[0]["ids"] == [] and event[0]["fields"] == ["results"]
    assert len(set(questions)) == 100
    assert {variant: variants.count(variant) for variant in QUOTAS} == QUOTAS
    assert sum(task["type"] == "compare" for task in hard_tasks) == 50
    assert sum(task["type"] == "sponsor_hop" for task in hard_tasks) == 50
    assert len(named) >= 40


def test_sponsor_keyword_hits_a_named_distractor(disk, hard_tasks):
    from fdagym.tasks.hard import _Cache, _profile
    from fdagym.tools import SECTION_LIMIT

    cache = _Cache(disk)
    seen: dict[str, str] = {}
    unique = []
    for task in hard_tasks:
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


def test_question_parser_reads_only_the_printed_strings():
    assert parse_question(question_earliest(["a", "b", "c"]))["variant"] == "earliest_boxed"
    assert parse_question(question_faers_year(["a", "b", "c"], 2023))["year"] == 2023
    assert parse_question(question_reaction(["a", "b", "c"], "NAUSEA"))["term"] == "NAUSEA"
    assert parse_question(question_none(["a", "b", "c"]))["variant"] == "boxed_none"
    assert parse_question(question_none_yes_no(["a", "b", "c"]))["variant"] == "boxed_none"
    assert parse_question(question_year_reaction(["a", "b", "c"], 2023)) == {
        "variant": "year_reaction",
        "drugs": ["a", "b", "c"],
        "year": 2023,
    }
    assert parse_question(question_earliest_reaction(["a", "b", "c"]))["variant"] == "earliest_reaction"
    faers = parse_question(question_sponsor_faers("NDA000001", ["a", "b"], "thyroid"))
    assert faers["drugs"] == ["a", "b"] and faers["keyword"] == "thyroid"
    drug = parse_question(question_sponsor_drug("BLA000001", ["a", "b", "c"], "thyroid"))
    assert drug["variant"] == "sponsor_drug" and drug["drugs"] == ["a", "b", "c"]
    with pytest.raises(Unsolvable):
        parse_question("Of a, b, and c, which drug is nicest?")


def test_sponsor_names_stay_under_the_cap(hard_tasks):
    from collections import Counter

    named = Counter()
    answers = Counter()
    for task in hard_tasks:
        spec = parse_question(task["question"])
        if spec["variant"] not in {"sponsor_faers", "sponsor_drug"}:
            continue
        for drug in spec["drugs"]:
            named[drug] += 1
        answers[task["entity"]["generic_name"]] += 1
    assert named
    assert named.most_common(1)[0][1] <= NAMED_CAP
    assert answers.most_common(1)[0][1] <= ANSWER_CAP


def test_compare_unfiltered_faers_rank_is_not_identifying(disk, hard_tasks):
    from collections import Counter, defaultdict

    from fdagym.queries import faers_report_count
    from fdagym.tasks.hard import _Cache

    cache = _Cache(disk)
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    totals_of: dict[str, int] = {}

    def all_years(drug: str) -> int:
        if drug not in totals_of:
            body = cache.query(faers_report_count(drug))
            totals_of[drug] = int(((body.get("meta") or {}).get("results") or {}).get("total") or 0)
        return totals_of[drug]

    for task in hard_tasks:
        spec = parse_question(task["question"])
        if spec["variant"] not in {"faers_year", "faers_reaction"}:
            continue
        gold = task["entity"]["generic_name"]
        values = [all_years(drug) for drug in spec["drugs"]]
        answer_n = all_years(gold)
        if _unique_max_value(answer_n, values):
            rank = "max"
        elif answer_n == min(values) and values.count(min(values)) == 1:
            rank = "min"
        elif values.count(answer_n) == 1:
            rank = "middle"
        else:
            rank = "tie"
        by_variant[spec["variant"]][rank] += 1
    assert by_variant
    for variant, ranks in by_variant.items():
        n = sum(ranks.values())
        top = ranks.most_common(1)[0]
        assert n >= 5
        assert top[1] / n < 0.70, (variant, dict(ranks))


def test_unique_max_value_is_the_all_years_faers_leak():
    famous = Profile("losartan", faers_all=231187)
    mid = Profile("celecoxib", faers_all=136146)
    rare = Profile("azithromycin", faers_all=98386)
    peer = Profile("olmesartan", faers_all=231187)
    assert _unique_max_value(231187, [231187, 136146, 98386])
    assert not _unique_max_value(136146, [231187, 136146, 98386])
    assert _unfiltered_faers_max(famous, [mid, rare])
    assert _unfiltered_faers_rank(mid, [famous, rare]) == "middle"
    assert _unfiltered_faers_rank(rare, [famous, mid]) == "min"
    assert not _unfiltered_faers_max(mid, [famous, rare])
    assert not _unfiltered_faers_max(famous, [peer, rare])


def test_comparable_band_rejects_a_fame_gap():
    rare = Profile("revumenib", label_n=1)
    near = Profile("aprocitentan", label_n=3)
    famous = Profile("metformin", label_n=365)
    assert _labels_within(rare, near)
    assert not _labels_within(rare, famous)


def test_a_shared_sponsor_cannot_be_rotated():
    from collections import Counter

    acme = Profile("a", sponsors={"acme": ["NDA000010"]}, all_sponsors={"acme": ["NDA000010"]})
    also = Profile("b", sponsors={"acme": ["NDA000020"]}, all_sponsors={"acme": ["NDA000020"]})
    index = {"acme": [("anchor", "NDA000099"), ("a", "NDA000010"), ("b", "NDA000020")]}
    assert _anchor_options(acme, [also], index, Counter(), 2) == []
    left = Profile("left", sponsors={"acme": ["NDA000010"]}, all_sponsors={"acme": ["NDA000010"]})
    right = Profile("right", sponsors={"beta": ["NDA000020"]}, all_sponsors={"beta": ["NDA000020"]})
    private = {
        "acme": [("anchor", "NDA000099"), ("left", "NDA000010")],
        "beta": [("other", "NDA000088"), ("right", "NDA000020")],
    }
    assert _anchor_options(left, [right], private, Counter(), 2) == ["NDA000099"]
    assert _set_is_symmetric([left, right], private, Counter(), 2, need_keyword=False)


def test_answer_slot_stays_even():
    import random

    counts = [0, 0, 0]
    rng = random.Random(0)
    for _ in range(30):
        slot = _next_slot(counts, rng)
        counts[slot] += 1
    assert counts == [10, 10, 10]


def test_sponsor_keyword_must_also_hit_a_distractor():
    unique = _shared_keyword(
        "Risk of thyroid C-cell tumors",
        ["Risk of serious infections"],
        ["tirzepatide", "natalizumab"],
    )
    shared = _shared_keyword(
        "QT prolong and toxicity",
        ["May prolong the QT interval"],
        ["thioridazine", "revumenib"],
    )
    assert unique is None
    assert shared == "prolong"


def test_earliest_boxed_needs_two_boxed_drugs():
    boxed = Profile("ondansetron", year=1991, boxed=True)
    boxed_later = Profile("sitagliptin", year=2006, boxed=True)
    plain = Profile("aprocitentan", year=2024, boxed=False)
    other = Profile("seladelpar", year=2024, boxed=False)
    assert _needs_year_compare([boxed, boxed_later, plain])
    assert not _needs_year_compare([boxed, plain, other])


def test_a_tie_is_not_a_gold_answer():
    with pytest.raises(Unsolvable, match="tied"):
        _one([("a", 3), ("b", 3)], least=False, what="count")
    assert _one([("a", 1), ("b", 4)], least=True, what="year") == "a"
    assert same_answer("no", False)
    assert same_answer("NAUSEA", "nausea")


def test_training_pools_drop_hard_tasks(hard_tasks):
    from copy import deepcopy

    from fdagym.train.curriculum import filter_pool, in_training_pool, type_mix_pool
    from tests.env.support import ALL_TASKS

    assert all(not in_training_pool(task) for task in hard_tasks)
    mixed = [*ALL_TASKS, *hard_tasks]
    assert {task["id"] for task in filter_pool(mixed, 1)}.isdisjoint(task["id"] for task in hard_tasks)
    rows, _mix = type_mix_pool(mixed, n_rows=30)
    assert {task["id"] for task in rows}.isdisjoint(task["id"] for task in hard_tasks)
    leaked = deepcopy(hard_tasks[0])
    leaked["id"] = "FG-8999"
    leaked["split"] = "train"
    assert leaked["type"] == "compare"
    assert not in_training_pool(leaked)
    held = deepcopy(ALL_TASKS[0])
    held["split"] = "hard"
    assert not in_training_pool(held)


def test_eval_entry_points_accept_a_hard_split_without_loading_it_for_training():
    modal = (ROOT / "modal_train.py").read_text()
    assert "def eval_tools(" in modal
    assert 'split: str = "test"' in modal
    assert 'hf_token: str = ""' in modal
    assert "split=split" in modal
    assert 'load_split("hard")' not in modal
    assert 'load_split("train")' in modal
    script = (ROOT / "scripts" / "eval_tools.py").read_text()
    assert "--split public" in script
    assert "import modal" not in script
    assert 'path = TASKS_DIR / f"{split}.jsonl"' in (ROOT / "fdagym" / "agents" / "baseline.py").read_text()


def test_solver_budget_stops_before_a_six_call_question_finishes(disk, hard_tasks):
    task = next(
        item
        for item in hard_tasks
        if item["difficulty_features"]["min_tool_calls"] == 6
    )
    with pytest.raises(Unsolvable, match="tool budget exhausted"):
        solve(disk, task["question"], budget=5)
    solved = solve(disk, task["question"], budget=8)
    assert solved.n_calls == 6


def test_snapshot_gold_is_unique_and_the_question_only_solver_scores(disk, hard_tasks):
    for task in hard_tasks:
        gold = snapshot_answer(disk, task["question"])
        assert gold == task["gold"]["value"], task["id"]
        solution = solve(disk, task["question"])
        assert same_answer(solution.answer, gold), task["id"]
        assert solution.calls == task["oracle_calls"], task["id"]
        assert solution.n_calls <= 6
        assert all(length <= 3500 for length in solution.observation_chars)
        reward = _play(disk, task, solution)
        assert reward["total"] == 1.0, (task["id"], reward["notes"][:6])


def test_generator_reproduces_the_committed_split(disk, hard_tasks):
    tasks, report = generate_hard(disk, seed=0)
    assert report["by_variant"] == QUOTAS
    assert report["by_type"] == {"compare": 50, "sponsor_hop": 50}
    assert [(task["id"], task["question"], task["gold"]) for task in tasks] == [
        (task["id"], task["question"], task["gold"]) for task in hard_tasks
    ]
