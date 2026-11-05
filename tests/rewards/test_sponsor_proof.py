"""A public sponsor_faers item keeps a pass when the label keyword was not read."""

import json
from pathlib import Path

import pytest

from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from fdagym.rewards import score
from fdagym.rewards.hops import hop_score

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15"
TASK_ID = "FG-9153"


def _task() -> dict:
    for line in PUBLIC.read_text().splitlines():
        task = json.loads(line)
        if task["id"] == TASK_ID:
            return task
    raise AssertionError(f"{TASK_ID} missing from public.jsonl")


def _play(calls: list[dict]) -> FDAEnv:
    task = _task()
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="fg-9153", seed=0)
    env.reset(task["id"])
    for call in calls:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None and done is False, obs.get("error")
    return env


def _submit(env: FDAEnv, citation: dict) -> dict:
    task = _task()
    env.step({"tool": "submit_answer", "arguments": {"answer": task["gold"]["value"], "citations": [citation]}})
    return env.trajectory["reward"]


@pytest.mark.skipif(not (SNAPSHOT / "manifest.jsonl").exists(), reason="snapshot absent")
def test_public_sponsor_faers_passes_without_reading_the_boxed_warning():
    task = _task()
    calls = [call for call in task["oracle_calls"] if call["tool"] not in {"search_drug_label", "get_label_section"}]
    env = _play(calls)
    faers = next(step for step in env.steps if step["tool"] == "query_adverse_events")
    quote = str(task["gold"]["value"])
    reward = _submit(
        env,
        {"endpoint": "drug/event", "id": faers["output"]["id"], "field": "results", "quote": quote},
    )
    hop = hop_score(env.steps, task)
    assert hop.variant == "sponsor_faers"
    assert ("label", False) in hop.flags
    assert hop.proves()
    assert reward["correctness"] == 1.0
    assert reward["citation"] == 1.0
    dense = score({"steps": env.steps, "final_answer": env.final_answer}, task, dense=True)
    assert dense["correctness"] == 1.0

    approvals = [call for call in calls if call["tool"] == "search_approvals"]
    short = _play(approvals)
    matched = next(
        step for step in short.steps if step["arguments"].get("generic_name") == task["entity"]["generic_name"]
    )
    row = next(
        item for item in matched["output"]["results"] if item["application_number"] in task["gold_citations"][0]["ids"]
    )
    withheld = _submit(
        short,
        {
            "endpoint": "drug/drugsfda",
            "id": row["application_number"],
            "field": "sponsor_name",
            "quote": row["sponsor_name"],
        },
    )
    assert withheld["citation"] == 1.0
    assert withheld["correctness"] == 0.0
    assert any("hop faers: miss" in note for note in withheld["notes"])
