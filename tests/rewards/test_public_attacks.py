"""Adversarial probes on the published sheet. These must run without private files."""

from collections import Counter
from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.rewards.citation import _TOOL_ENDPOINT, _collapse, _dicts, _render, _resolve
from fdagym.tasks.hard_solver import parse_question
from tests.integration.test_oracle import gold_answer

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15"
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"


def _quote(text: str) -> str:
    collapsed = _collapse(text)
    if len(collapsed) <= 12 or len(collapsed) < 20:
        return collapsed
    return collapsed[:20]


def _accepted_id(node: dict, gold: dict) -> str | None:
    for key in ("set_id", "application_number", "safetyreportid", "id"):
        value = node.get(key)
        if not isinstance(value, str) or not value:
            continue
        if gold["ids"]:
            if value in gold["ids"]:
                return value
        elif value.startswith("step:"):
            return value
    return None


def _first_gold_cite(steps: list[dict], task: dict) -> dict | None:
    for gold in task["gold_citations"]:
        for step in steps:
            if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != gold["endpoint"]:
                continue
            for node in _dicts(step.get("output")):
                if not isinstance(node, dict):
                    continue
                cid = _accepted_id(node, gold)
                if cid is None:
                    continue
                for field in gold["fields"]:
                    text = _render(_resolve(node, field))
                    if text:
                        return {
                            "endpoint": gold["endpoint"],
                            "id": cid,
                            "field": field,
                            "quote": _quote(text),
                        }
    return None


def _citations_preferring_entity(steps: list[dict], task: dict) -> list[dict]:
    prefer = ((task.get("entity") or {}).get("generic_name") or "").strip().casefold()

    def _rank(step: dict) -> int:
        name = (step.get("arguments") or {}).get("generic_name")
        if isinstance(name, str) and prefer and name.strip().casefold() == prefer:
            return 0
        return 1

    ordered = sorted(steps, key=_rank)
    citations = []
    for gold in task["gold_citations"]:
        found = None
        for step in ordered:
            if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != gold["endpoint"]:
                continue
            for node in _dicts(step.get("output")):
                if not isinstance(node, dict):
                    continue
                cid = _accepted_id(node, gold)
                if cid is None:
                    continue
                for field in gold["fields"]:
                    text = _render(_resolve(node, field))
                    if text:
                        found = (gold["endpoint"], cid, field, text)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            raise AssertionError(f"{task['id']} gold field {gold['fields']} never appeared")
        endpoint, cid, field, text = found
        citations.append({"endpoint": endpoint, "id": cid, "field": field, "quote": _quote(text)})
    return citations


def _submit(env: FDAEnv, answer, citations: list[dict]) -> dict:
    _obs, _reward, done, info = env.step(
        {"tool": "submit_answer", "arguments": {"answer": answer, "citations": citations}}
    )
    assert done is True
    return info["reward"]


@pytest.fixture(scope="module")
def public_bundle():
    if not (SNAPSHOT / "manifest.jsonl").exists():
        pytest.fail("committed snapshot data/snapshots/2026-10-15 is required")
    if not PUBLIC.exists():
        pytest.fail("data/tasks/public.jsonl is required")
    tasks = load_split("public")
    env = FDAEnv(DiskSnapshot("2026-10-15"), tasks, run_id="public-attacks")
    return env, tasks


def test_one_call_generic_quote_oracle_answer_passes_zero_public_items(public_bundle):
    env, tasks = public_bundle
    passed = Counter()
    totals = Counter()
    for task in tasks:
        kind = task["type"]
        totals[kind] += 1
        env.reset(task["id"])
        call = task["oracle_calls"][0]
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None and done is False, obs.get("error")
        cite = _first_gold_cite(env.steps, task)
        reward = _submit(env, gold_answer(task), [cite] if cite else [])
        if reward["correctness"] > 0:
            passed[kind] += 1
    assert totals["compare"] == 50
    assert totals["sponsor_hop"] == 43
    assert passed["compare"] == 0, dict(passed)
    assert passed["sponsor_hop"] == 0, dict(passed)


def test_quitter_and_first_named_guess_pass_zero_public_items(public_bundle):
    env, tasks = public_bundle
    quitter = Counter()
    first_named = Counter()
    totals = Counter()
    for task in tasks:
        kind = task["type"]
        totals[kind] += 1
        env.reset(task["id"])
        reward = _submit(env, "unknown", [])
        if reward["correctness"] > 0:
            quitter[kind] += 1

        env.reset(task["id"])
        call = task["oracle_calls"][0]
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None and done is False, obs.get("error")
        cite = _first_gold_cite(env.steps, task)
        drugs = parse_question(task["question"]).get("drugs") or []
        guess = drugs[0] if drugs else gold_answer(task)
        reward = _submit(env, guess, [cite] if cite else [])
        if reward["correctness"] > 0:
            first_named[kind] += 1
    assert quitter["compare"] == 0 and quitter["sponsor_hop"] == 0
    assert first_named["compare"] == 0 and first_named["sponsor_hop"] == 0
    assert totals["compare"] + totals["sponsor_hop"] == 93


def test_fg9101_one_call_pain_quote_does_not_pass(public_bundle):
    env, tasks = public_bundle
    task = next(item for item in tasks if item["id"] == "FG-9101")
    env.reset(task["id"])
    obs, _reward, done, _info = env.step(
        {"tool": "query_adverse_events", "arguments": {"generic_name": "abacavir", "mode": "top_reactions"}}
    )
    assert obs["error"] is None and done is False
    reward = _submit(
        env,
        "pravastatin",
        [{"endpoint": "drug/event", "id": "step:1", "field": "results", "quote": "PAIN"}],
    )
    assert reward["citation"] == 1.0
    assert reward["correctness"] == 0.0
    assert reward["efficiency"] == 0.0
    assert any("compare proof" in note for note in reward["notes"])


def test_honest_oracle_passes_every_public_item(public_bundle):
    env, tasks = public_bundle
    passed = Counter()
    totals = Counter()
    failures = []
    for task in tasks:
        kind = task["type"]
        totals[kind] += 1
        env.reset(task["id"])
        for call in task["oracle_calls"]:
            obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
            assert obs["error"] is None and done is False, f"{task['id']} {obs.get('error')}"
        citations = _citations_preferring_entity(env.steps, task)
        reward = _submit(env, gold_answer(task), citations)
        if reward["correctness"] > 0:
            passed[kind] += 1
        else:
            failures.append((task["id"], kind, reward["total"], reward["notes"][:8]))
    assert not failures, failures
    assert passed["compare"] == totals["compare"] == 50
    assert passed["sponsor_hop"] == totals["sponsor_hop"] == 43
