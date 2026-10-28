"""A perfect agent plays every generated task through FDAEnv and the pilot snapshot.

The oracle is a contract test: the task generator and the reward must agree on the same evidence.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from fdagym.env import FDAEnv
from fdagym.rewards.citation import _TOOL_ENDPOINT, _collapse, _dicts, _render, _resolve

SNAPSHOT = Path("data/snapshots/2026-10-15")
TASKS = Path("data/tasks")
_SKIP = "label_open"


def _rows() -> list[dict]:
    rows = []
    for name in ("train.jsonl", "test.jsonl"):
        path = TASKS / name
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


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


def citations_from_outputs(steps: list[dict], task: dict) -> list[dict]:
    """One citation per gold entry, taken from a field the tool output actually contained."""
    citations = []
    for gold in task["gold_citations"]:
        found = None
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
                        found = (cid, field, text)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            raise AssertionError(f"{task['id']} gold field {gold['fields']} never appeared")
        cid, field, text = found
        citations.append({"endpoint": gold["endpoint"], "id": cid, "field": field, "quote": _quote(text)})
    return citations


def gold_answer(task: dict):
    if not task["answerable"]:
        return "NOT_FOUND"
    gold = task["gold"]
    if gold["kind"] == "boolean":
        if gold["value"]:
            keywords = gold.get("keywords") or []
            if keywords:
                return f"Yes. {keywords[0]}."
            return "Yes."
        return "No."
    return gold["value"]


def wrong_answer(task: dict):
    gold = task["gold"]
    if gold["kind"] == "boolean":
        return "No." if gold["value"] else "Yes."
    if gold["kind"] == "number":
        return float(gold["value"]) + 1
    return "WRONG ANSWER"


def play(env: FDAEnv, task: dict, answer) -> dict:
    env.reset(task["id"])
    for call in task["oracle_calls"]:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None, f"{task['id']} {call['tool']} {obs['error']}"
        assert done is False
    citations = citations_from_outputs(env.steps, task)
    _obs, _reward, done, info = env.step(
        {"tool": "submit_answer", "arguments": {"answer": answer, "citations": citations}}
    )
    assert done is True
    return info["reward"]


@pytest.fixture(scope="module")
def pilot_tasks():
    if not (SNAPSHOT / "manifest.jsonl").exists():
        pytest.skip("pilot snapshot absent")
    from fdagym.openfda import DiskSnapshot

    tasks = _rows()
    assert tasks
    return DiskSnapshot("2026-10-15"), tasks


def test_oracle_scores_one_on_every_answerable_task(pilot_tasks, capsys):
    snapshot, tasks = pilot_tasks
    env = FDAEnv(snapshot, tasks, run_id="oracle")
    scored: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    failures = []
    skipped = 0
    for task in tasks:
        totals[task["type"]] += 1
        if task["type"] == _SKIP:
            skipped += 1
            continue
        reward = play(env, task, gold_answer(task))
        if reward["total"] == 1.0:
            scored[task["type"]] += 1
        else:
            failures.append((task["id"], task["type"], task["question"], reward))
    print(f"skipped {_SKIP}: {skipped}")
    for kind in sorted(totals):
        if kind == _SKIP:
            print(f"{kind}: skipped {totals[kind]}")
        else:
            print(f"{kind}: {scored[kind]}/{totals[kind]}")
    if failures:
        lines = []
        for task_id, kind, question, reward in failures:
            lines.append(f"{task_id} {kind} total={reward['total']} {question}")
            lines.extend(f"  {note}" for note in reward["notes"])
        raise AssertionError("\n".join(lines))
    assert skipped == totals[_SKIP]
    assert sum(scored.values()) == sum(totals.values()) - skipped


def test_wrong_answer_with_oracle_citations_scores_citation_and_format_only(pilot_tasks):
    snapshot, tasks = pilot_tasks
    env = FDAEnv(snapshot, tasks, run_id="oracle-wrong")
    seen = set()
    for task in tasks:
        if task["type"] in seen or task["type"] == _SKIP:
            continue
        seen.add(task["type"])
        reward = play(env, task, wrong_answer(task))
        assert reward["total"] == 0.35, (task["id"], reward)
        assert reward["correctness"] == 0.0
        assert reward["citation"] == 1.0
        assert reward["format"] == 1.0
        assert reward["efficiency"] == 0.0
    assert seen == {"approval", "boxed_warning", "faers_top_reaction", "faers_count", "composite"}
