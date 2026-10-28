"""Sponsor questions must not be guessable from a surface feature of the named drugs."""

from collections import Counter

import pytest

from fdagym.tasks.hard_solver import question_sponsor_count, question_sponsor_drug, question_sponsor_faers
from fdagym.tasks.leaks import clear_leaks, format_leak_report, leak_report

SPLITS = ("hard", "hard_train", "hard_train_holdout")


def _task(drugs: list[str], answer: str, variant: str = "sponsor_drug", keyword: str = "thyroid") -> dict:
    if variant == "sponsor_faers":
        question = question_sponsor_faers("NDA100001", drugs, keyword)
        gold = "NAUSEA"
    elif variant == "sponsor_count":
        question = question_sponsor_count("NDA100001", drugs, 2023)
        gold = 4
    else:
        question = question_sponsor_drug("NDA100001", drugs, keyword)
        gold = answer
    return {
        "question": question,
        "gold": {"kind": "exact", "value": gold},
        "entity": {"generic_name": answer},
        "type": "sponsor_hop",
    }


def _report(tasks, labels, texts=None, train=None):
    return leak_report(
        tasks,
        label_counts={name.casefold(): count for name, count in labels.items()},
        boxed_texts={name.casefold(): text for name, text in (texts or {}).items()},
        train_tasks=train if train is not None else tasks,
    )


def test_alphabetical_and_tie_shortcuts_fail_the_detector():
    alpha = []
    labels = {}
    for i in range(20):
        drugs = [f"a{i:02d}", f"m{i:02d}", f"z{i:02d}"]
        alpha.append(_task(drugs, drugs[0]))
        labels.update({drug: 10 for drug in drugs})
    report = _report(alpha, labels, train=alpha)
    assert any(item.startswith("sponsor_drug alpha_rank1") for item in clear_leaks(report))

    tied = []
    odd_labels = {}
    for i in range(24):
        left, odd, right = f"left{i:02d}", f"odd{i:02d}", f"right{i:02d}"
        tied.append(_task([left, odd, right], left))
        odd_labels.update({left: 5, odd: 9, right: 5})
    report = _report(tied, odd_labels, train=tied)
    assert any("tied_group" in item for item in clear_leaks(report))


def test_a_rotated_set_is_not_a_leak():
    drugs = ["alpha", "middle", "zulu"]
    labels = {"alpha": 10, "middle": 40, "zulu": 90}
    texts = {name: "thyroid risk and another thyroid" for name in drugs}
    tasks = []
    for i in range(18):
        answer = drugs[i % 3]
        slot = (i // 3) % 3
        order = [drug for drug in drugs if drug != answer]
        order.insert(slot, answer)
        tasks.append(_task(order, answer))
    report = _report(tasks, labels, texts, train=tasks)
    assert clear_leaks(report) == [], format_leak_report("rotated", report)


def test_a_unique_keyword_is_a_leak_even_on_a_short_sheet():
    drugs = ["alpha", "middle"]
    tasks = [_task(drugs, "alpha", variant="sponsor_faers", keyword="thyroid") for _ in range(3)]
    texts = {"alpha": "thyroid storm", "middle": "no boxed word here"}
    labels = {"alpha": 10, "middle": 10}
    report = _report(tasks, labels, texts, train=tasks)
    assert any("unique_keyword" in item for item in clear_leaks(report))


def _caches(snapshot, tasks):
    from fdagym.tasks.hard import sponsor_surface_caches

    return sponsor_surface_caches(snapshot, tasks)


@pytest.fixture(scope="module")
def surface():
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot

    snapshot = DiskSnapshot("2026-10-15")
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data" / "tasks"
    names = [name for name in SPLITS if (root / f"{name}.jsonl").exists()]
    splits = {name: load_split(name) for name in names}
    labels, texts = _caches(snapshot, [task for rows in splits.values() for task in rows])
    return splits, labels, texts


def test_generated_sponsor_questions_have_no_surface_leak(surface):
    splits, labels, texts = surface
    train = splits["hard_train"]
    failures = []
    blocks = []
    for name in splits:
        report = leak_report(
            splits[name],
            label_counts=labels,
            boxed_texts=texts,
            train_tasks=train,
        )
        blocks.append(format_leak_report(name, report))
        for item in clear_leaks(report):
            failures.append(f"{name}: {item}")
    assert failures == [], "\n".join([*failures, *blocks])


def test_no_sponsor_variant_dominates_training():
    from fdagym.agents.baseline import load_split
    from fdagym.tasks.hard_solver import parse_question

    train = load_split("hard_train")
    assert len(train) >= 300
    counts = Counter(parse_question(task["question"])["variant"] for task in train)
    assert counts["sponsor_faers"] >= 15
    assert max(counts.values()) <= 0.30 * len(train)
