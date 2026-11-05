"""The published sheet is a hard-style generation that does not reuse the private hard set."""

from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.openfda import DiskSnapshot
from fdagym.tasks.disjoint import assert_disjoint
from fdagym.tasks.hard_solver import parse_question

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
HARD = ROOT / "data" / "tasks" / "hard.jsonl"
TRAIN = ROOT / "data" / "tasks" / "hard_train.jsonl"
HOLDOUT = ROOT / "data" / "tasks" / "hard_train_holdout.jsonl"


@pytest.mark.skipif(not PUBLIC.exists(), reason="public sheet has not been generated")
def test_public_sheet_is_its_own_split():
    tasks = load_split("public")
    assert len(tasks) == 93
    assert [task["id"] for task in tasks] == [f"FG-{9101 + offset:04d}" for offset in range(93)]
    assert {task["split"] for task in tasks} == {"public"}
    assert {task["type"] for task in tasks} <= {"compare", "sponsor_hop"}
    for task in tasks:
        parse_question(task["question"])


def test_published_splits_share_no_entities_or_questions():
    """Always-on C2 gate. hard_train_holdout may share drugs with hard_train; neither may share with public."""
    assert PUBLIC.exists() and TRAIN.exists() and HOLDOUT.exists()
    public = load_split("public")
    train = load_split("hard_train")
    holdout = load_split("hard_train_holdout")
    snapshot = DiskSnapshot("2026-10-15")

    train_vs_public = assert_disjoint(train, public, snapshot)
    holdout_vs_public = assert_disjoint(holdout, public, snapshot)
    assert train_vs_public["overlap"] == {"drugs": [], "applications": [], "sponsors": []}
    assert holdout_vs_public["overlap"] == {"drugs": [], "applications": [], "sponsors": []}
    assert train_vs_public["train_tasks"] == 337
    assert holdout_vs_public["train_tasks"] == 36

    public_questions = {task["question"] for task in public}
    train_questions = {task["question"] for task in train}
    shared_with_public = [task["id"] for task in [*train, *holdout] if task["question"] in public_questions]
    shared_train_holdout = [task["id"] for task in holdout if task["question"] in train_questions]
    assert shared_with_public == []
    assert shared_train_holdout == []

    ids = [task["id"] for task in [*public, *train, *holdout]]
    assert len(ids) == len(set(ids))

    # The previous hard_train FG-8393 question was a copy of public FG-9171.
    public_9171 = next(task["question"] for task in public if task["id"] == "FG-9171")
    assert all(task["question"] != public_9171 for task in [*train, *holdout])


@pytest.mark.skipif(not PUBLIC.exists() or not HARD.exists(), reason="overlap check needs both sheets")
def test_public_sheet_does_not_overlap_the_private_hard_set():
    report = assert_disjoint(load_split("public"), load_split("hard"), DiskSnapshot("2026-10-15"))
    assert report["overlap"] == {"drugs": [], "applications": [], "sponsors": []}
    assert report["train_tasks"] == 93
    assert report["holdout_tasks"] == 100
