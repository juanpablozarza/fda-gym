"""Stored oracles are the shortest path the current grader already accepts.

Generic-name summaries expose original_approval, so hops._visible_year does not
need a follow-up search_approvals(application_number=…) hop. Composite gold is
an exact ``"<count>; yes|no"`` string; one FAERS citation already unlocks it, so
label hops after that call are prefix padding.
"""

from pathlib import Path

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.tasks.hard_solver import parse_question
from tests.integration.test_oracle import citations_from_outputs, gold_answer, play
from tests.rewards.test_public_attacks import _citations_preferring_entity, _submit

ROOT = Path(__file__).resolve().parents[2]
EXAM = ROOT / "data" / "snapshots" / "2026-10-15"
_SKIP_TYPES = frozenset({"label_open"})
_SPLITS = ("public", "hard_train", "hard_train_holdout", "train", "test")
_HIGH = 0.95


def _detail(call: dict) -> bool:
    return call.get("tool") == "search_approvals" and "application_number" in (call.get("arguments") or {})


def _play_prefix(env: FDAEnv, task: dict, n: int) -> dict:
    env.reset(task["id"])
    for call in task["oracle_calls"][:n]:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None, f"{task['id']} {obs.get('error')}"
        assert done is False
    visible = _visible_gold(env.steps, task)
    citations = citations_from_outputs(env.steps, {**task, "gold_citations": visible}) if visible else []
    _obs, _reward, done, info = env.step(
        {"tool": "submit_answer", "arguments": {"answer": gold_answer(task), "citations": citations}}
    )
    assert done is True
    return info["reward"]


def _visible_gold(steps: list[dict], task: dict) -> list[dict]:
    """Gold citation rows whose field actually appeared, so a prefix can still submit."""
    kept = []
    for gold in task["gold_citations"]:
        try:
            citations_from_outputs(steps, {**task, "gold_citations": [gold]})
        except AssertionError:
            continue
        kept.append(gold)
    return kept


def _play_full(env: FDAEnv, task: dict) -> dict:
    """Honest oracle. Compare/sponsor_hop citations prefer the deciding drug."""
    if task["type"] not in {"compare", "sponsor_hop"}:
        return play(env, task, gold_answer(task))
    env.reset(task["id"])
    for call in task["oracle_calls"]:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None, f"{task['id']} {call['tool']} {obs.get('error')}"
        assert done is False
    citations = _citations_preferring_entity(env.steps, task)
    return _submit(env, gold_answer(task), citations)


def test_public_earliest_boxed_oracle_is_the_shortest_proving_path():
    assert (EXAM / "manifest.jsonl").exists()
    tasks = [task for task in load_split("public") if parse_question(task["question"])["variant"] == "earliest_boxed"]
    assert len(tasks) == 16
    env = FDAEnv(DiskSnapshot("2026-10-15"), tasks, run_id="n2-public-oracle", tool_budget=8)
    for task in tasks:
        assert task["difficulty_features"]["min_tool_calls"] == 6
        assert len(task["oracle_calls"]) == 6
        assert not any(_detail(call) for call in task["oracle_calls"])
        full = play(env, task, gold_answer(task))
        assert full["total"] == 1.0, (task["id"], full["total"], full["notes"][:6])
        for n in range(len(task["oracle_calls"])):
            prefix = _play_prefix(env, task, n)
            assert prefix["total"] < 0.95, (task["id"], n, prefix["total"])


def test_fg9102_documents_that_the_old_detail_hop_was_unnecessary():
    """FG-9102 previously stored search_approvals(application_number=NDA016012) as an extra hop."""
    assert (EXAM / "manifest.jsonl").exists()
    task = next(item for item in load_split("public") if item["id"] == "FG-9102")
    assert parse_question(task["question"])["variant"] == "earliest_boxed"
    assert not any(_detail(call) for call in task["oracle_calls"])
    year_cite = next(row for row in task["gold_citations"] if row["endpoint"] == "drug/drugsfda")
    assert year_cite["fields"] == ["original_approval"]
    assert "NDA016012" in year_cite["ids"]
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="n2-fg9102", tool_budget=8)
    reward = play(env, task, gold_answer(task))
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0
    assert reward["efficiency"] == 1.0


def test_fg0031_keeps_one_submissions_hop_and_stays_labeled_hard():
    """Year questions still cite submissions, so one detail hop stays. Extra winner apps do not."""
    assert (EXAM / "manifest.jsonl").exists()
    task = next(item for item in load_split("test") if item["id"] == "FG-0031")
    assert task["type"] == "approval"
    # Label is left as stored (medium on this sheet). Only oracle/min change.
    assert task["difficulty"] == "medium"
    assert task["difficulty_features"]["min_tool_calls"] == 2
    assert len(task["oracle_calls"]) == 2
    assert not _detail(task["oracle_calls"][0])
    assert _detail(task["oracle_calls"][1])
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="n2-fg0031", tool_budget=8)
    full = play(env, task, gold_answer(task))
    assert full["total"] == 1.0
    prefix = _play_prefix(env, task, 1)
    assert prefix["total"] < 0.95


def test_composite_oracles_are_the_faers_count_call():
    """17 test + 69 train composites. Public and hard sheets have none."""
    train = [task for task in load_split("train") if task["type"] == "composite"]
    test = [task for task in load_split("test") if task["type"] == "composite"]
    assert len(train) == 69
    assert len(test) == 17
    assert [task["id"] for task in test] == [f"FG-{177 + offset:04d}" for offset in range(17)]
    assert not any(task["type"] == "composite" for task in load_split("public"))
    for task in (*train, *test):
        assert task["difficulty"] == "hard"
        assert task["difficulty_features"]["sources"] == 2
        assert task["difficulty_features"]["min_tool_calls"] == 1
        assert len(task["oracle_calls"]) == 1
        call = task["oracle_calls"][0]
        assert call["tool"] == "query_adverse_events"
        assert call["arguments"]["mode"] == "count"
        assert call["arguments"]["serious"] is True
        assert "year" in call["arguments"]


def test_every_gradable_oracle_is_the_shortest_proving_path():
    """A strict prefix of the recorded oracle must not already score high."""
    assert (EXAM / "manifest.jsonl").exists()
    seen: set[str] = set()
    composite_counts = {"train": 0, "test": 0}
    for split in _SPLITS:
        tasks = [task for task in load_split(split) if task["type"] not in _SKIP_TYPES]
        assert tasks, split
        env = FDAEnv(DiskSnapshot("2026-10-15"), tasks, run_id=f"n9-{split}", tool_budget=8)
        for task in tasks:
            seen.add(task["type"])
            if task["type"] == "composite":
                composite_counts[split] = composite_counts.get(split, 0) + 1
            full = _play_full(env, task)
            assert full["total"] == 1.0, (task["id"], split, full["total"], full["notes"][:6])
            for n in range(len(task["oracle_calls"])):
                prefix = _play_prefix(env, task, n)
                assert prefix["total"] < _HIGH, (task["id"], task["type"], split, n, prefix["total"])
    assert "composite" in seen
    assert composite_counts == {"train": 69, "test": 17}
