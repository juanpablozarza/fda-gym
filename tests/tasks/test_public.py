"""The published sheet is a hard-style generation that does not reuse the private hard set."""

from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.tasks.hard_solver import parse_question

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
HARD = ROOT / "data" / "tasks" / "hard.jsonl"


@pytest.mark.skipif(not PUBLIC.exists(), reason="public sheet has not been generated")
def test_public_sheet_is_its_own_split():
    tasks = load_split("public")
    assert len(tasks) == 93
    assert [task["id"] for task in tasks] == [f"FG-{9101 + offset:04d}" for offset in range(93)]
    assert {task["split"] for task in tasks} == {"public"}
    assert {task["type"] for task in tasks} <= {"compare", "sponsor_hop"}
    for task in tasks:
        parse_question(task["question"])


@pytest.mark.skipif(not PUBLIC.exists() or not HARD.exists(), reason="overlap check needs both sheets")
def test_public_sheet_does_not_overlap_the_private_hard_set():
    from fdagym.openfda import DiskSnapshot
    from fdagym.tasks.disjoint import assert_disjoint

    report = assert_disjoint(load_split("public"), load_split("hard"), DiskSnapshot("2026-10-15"))
    assert report["overlap"] == {"drugs": [], "applications": [], "sponsors": []}
    assert report["train_tasks"] == 93
    assert report["holdout_tasks"] == 100
