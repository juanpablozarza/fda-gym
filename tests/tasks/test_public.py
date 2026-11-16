"""The published sheet stays disjoint from hard_train and the question holdout."""

from pathlib import Path

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.tasks.disjoint import assert_disjoint
from fdagym.tasks.hard import snapshot_answer
from fdagym.tasks.hard_solver import parse_question, same_answer, solve
from tests.integration.test_oracle import gold_answer, play

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
TRAIN = ROOT / "data" / "tasks" / "hard_train.jsonl"
HOLDOUT = ROOT / "data" / "tasks" / "hard_train_holdout.jsonl"
EMPTY_OVERLAP = {"drugs": [], "applications": [], "sponsors": [], "questions": []}


def test_public_sheet_is_its_own_split():
    assert PUBLIC.exists()
    tasks = load_split("public")
    assert len(tasks) == 93
    assert [task["id"] for task in tasks] == [f"FG-{9101 + offset:04d}" for offset in range(93)]
    assert {task["split"] for task in tasks} == {"public"}
    assert {task["type"] for task in tasks} <= {"compare", "sponsor_hop"}
    named_sponsor = 0
    for task in tasks:
        spec = parse_question(task["question"])
        if spec["variant"] not in {"sponsor_faers", "sponsor_drug"}:
            continue
        tools = {call["tool"] for call in task["oracle_calls"]}
        assert tools.isdisjoint({"search_drug_label", "get_label_section"})
        assert task["difficulty_features"]["min_tool_calls"] == len(task["oracle_calls"]) == 4
        named_sponsor += 1
    assert named_sponsor > 0


def test_published_splits_share_no_entities_or_questions():
    """Always-on gate. hard_train_holdout may share drugs with hard_train; neither may share with public."""
    assert PUBLIC.exists() and TRAIN.exists() and HOLDOUT.exists()
    public = load_split("public")
    train = load_split("hard_train")
    holdout = load_split("hard_train_holdout")
    snapshot = DiskSnapshot("2026-10-15")

    train_vs_public = assert_disjoint(train, public, snapshot)
    public_vs_train = assert_disjoint(public, train, snapshot)
    holdout_vs_public = assert_disjoint(holdout, public, snapshot)
    public_vs_holdout = assert_disjoint(public, holdout, snapshot)
    assert train_vs_public["overlap"] == EMPTY_OVERLAP
    assert public_vs_train["overlap"] == EMPTY_OVERLAP
    assert holdout_vs_public["overlap"] == EMPTY_OVERLAP
    assert public_vs_holdout["overlap"] == EMPTY_OVERLAP

    public_questions = {task["question"] for task in public}
    train_questions = {task["question"] for task in train}
    shared_with_public = [task["id"] for task in [*train, *holdout] if task["question"] in public_questions]
    shared_train_holdout = [task["id"] for task in holdout if task["question"] in train_questions]
    assert shared_with_public == []
    assert shared_train_holdout == []

    ids = [task["id"] for task in [*public, *train, *holdout]]
    assert len(ids) == len(set(ids))

    public_9171 = next(task["question"] for task in public if task["id"] == "FG-9171")
    assert all(task["question"] != public_9171 for task in [*train, *holdout])


def test_fg9118_and_fg9140_are_golded_on_the_real_boxed_warning():
    """Amoxicillin's snapshot 'boxed warning' is an unfilled template; gold is the other boxed drug."""
    public = {task["id"]: task for task in load_split("public")}
    expected = {"FG-9118": "buprenorphine", "FG-9140": "rivaroxaban"}
    snapshot = DiskSnapshot("2026-10-15")
    env = FDAEnv(snapshot, list(public.values()), run_id="public-regold", tool_budget=8)
    for task_id, winner in expected.items():
        task = public[task_id]
        assert parse_question(task["question"])["variant"] == "earliest_boxed"
        assert task["gold"] == {"kind": "exact", "value": winner}
        assert task["entity"]["generic_name"] == winner
        assert "482511ed-4531-45ec-a4eb-c600aeb03325" not in {
            cid for row in task["gold_citations"] for cid in row.get("ids") or []
        }
        gold = snapshot_answer(snapshot, task["question"])
        solution = solve(snapshot, task["question"])
        assert gold == winner
        assert same_answer(solution.answer, gold)
        assert solution.calls == task["oracle_calls"]
        reward = play(env, task, gold_answer(task))
        assert reward["total"] == 1.0, (task_id, reward["notes"][:8])
        assert reward["correctness"] == 1.0
