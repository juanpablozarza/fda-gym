"""Named sponsor oracles are the shortest path ``proves()`` already accepts.

Label and section calls used to pad ``sponsor_faers`` / ``sponsor_drug`` to 6
calls. ``HopScore.proves`` does not require them, so ``min_tool_calls`` 6 taxed
efficiency on a 4-call proof. Public IDs only.
"""

from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot
from tests.integration.test_oracle import gold_answer
from tests.rewards.test_public_attacks import _citations_preferring_entity, _submit

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15"
LABEL_TOOLS = frozenset({"search_drug_label", "get_label_section"})

# Frozen copies of the pre-cut 6-call public oracles (tool names and arguments only).
OLD_FG_9151 = [
    {"tool": "search_approvals", "arguments": {"application_number": "NDA007337"}},
    {"tool": "search_approvals", "arguments": {"generic_name": "risperidone"}},
    {"tool": "search_approvals", "arguments": {"generic_name": "sumatriptan"}},
    {"tool": "search_approvals", "arguments": {"generic_name": "olanzapine"}},
    {"tool": "search_drug_label", "arguments": {"generic_name": "sumatriptan", "has_section": "boxed_warning"}},
    {
        "tool": "get_label_section",
        "arguments": {"set_id": "3f2ec27e-0e96-4028-a1fa-114e6d501464", "section": "boxed_warning"},
    },
]
OLD_FG_9153 = [
    {"tool": "search_approvals", "arguments": {"application_number": "NDA017862"}},
    {"tool": "search_approvals", "arguments": {"generic_name": "febuxostat"}},
    {"tool": "search_approvals", "arguments": {"generic_name": "ambrisentan"}},
    {"tool": "search_drug_label", "arguments": {"generic_name": "febuxostat", "has_section": "boxed_warning"}},
    {
        "tool": "get_label_section",
        "arguments": {"set_id": "04f80eb2-8171-4e18-93e4-514440e84b75", "section": "boxed_warning"},
    },
    {"tool": "query_adverse_events", "arguments": {"generic_name": "febuxostat", "mode": "top_reactions"}},
]


def _task(task_id: str) -> dict:
    return next(task for task in load_split("public") if task["id"] == task_id)


def _play(task: dict, calls: list[dict]) -> dict:
    env = FDAEnv(DiskSnapshot("2026-10-15"), [task], run_id="sponsor-oracle-min", seed=0)
    env.reset(task["id"])
    for call in calls:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None and done is False, obs.get("error")
    citations = _citations_preferring_entity(env.steps, task)
    return _submit(env, gold_answer(task), citations)


@pytest.mark.skipif(not (SNAPSHOT / "manifest.jsonl").exists(), reason="snapshot absent")
@pytest.mark.parametrize(
    "task_id,old_oracle",
    [("FG-9151", OLD_FG_9151), ("FG-9153", OLD_FG_9153)],
)
def test_new_oracle_scores_high_and_every_prefix_does_not(task_id, old_oracle):
    task = _task(task_id)
    calls = task["oracle_calls"]
    assert task["difficulty_features"]["min_tool_calls"] == len(calls) == 4
    assert {call["tool"] for call in calls}.isdisjoint(LABEL_TOOLS)

    reward = _play(task, calls)
    assert reward["correctness"] == 1.0
    assert reward["efficiency"] == 1.0
    assert reward["total"] == 1.0

    for length in range(1, len(calls)):
        prefix = _play(task, calls[:length])
        assert prefix["total"] < 0.95, (task_id, length, prefix["total"], prefix["notes"][:6])
        assert prefix["correctness"] == 0.0

    assert len(old_oracle) == 6
    assert {call["tool"] for call in old_oracle} >= LABEL_TOOLS
    stripped = [call for call in old_oracle if call["tool"] not in LABEL_TOOLS]
    assert stripped == calls
    dropped = _play(task, stripped)
    assert dropped["correctness"] == 1.0
    assert dropped["total"] >= 0.95
