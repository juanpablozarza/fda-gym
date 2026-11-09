"""Prime stores the four grader terms as weighted rewards."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from fdagym.demo import gold_submit
from fdagym.prime.episode import (
    ADAPTER_SYSTEM_PROMPT,
    DEFAULT_TASK_ID,
    grade_steps,
    make_public_env,
    play_public_adapter_episode,
    step_tool,
)
from fdagym.rewards.combine import WEIGHTS
from fdagym.rewards.version import GRADER_VERSION

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
SNAPSHOT = ROOT / "data" / "snapshots"
TERMS = ("correctness", "citation", "format", "efficiency")
_ID_KEYS = ("id", "set_id", "application_number", "safetyreportid")


def _prime():
    pytest.importorskip("verifiers.v1")
    import verifiers.v1 as vf

    from fdagym.prime.servers.tools import FDAGymState
    from fdagym.prime.taskset import FDAGymTask, FDAGymTaskConfig, FDAGymTaskData

    return vf, FDAGymState, FDAGymTask, FDAGymTaskConfig, FDAGymTaskData


def _replay(task_id: str = DEFAULT_TASK_ID):
    env, task = make_public_env(task_id, public_path=PUBLIC, snapshot_root=SNAPSHOT)
    for call in task["oracle_calls"]:
        _obs, _reward, done, _info = step_tool(env, call["tool"], dict(call["arguments"]))
        assert done is False
    _obs, _reward, done, _info = step_tool(env, "submit_answer", gold_submit(task, env.steps))
    assert done is True
    return env, task


def _gym_task(task: dict):
    _vf, _state, FDAGymTask, FDAGymTaskConfig, FDAGymTaskData = _prime()
    return FDAGymTask(
        FDAGymTaskData(
            id=task["id"],
            name=task["id"],
            prompt=task["question"],
            system_prompt=ADAPTER_SYSTEM_PROMPT,
            sheet_row=task,
        ),
        FDAGymTaskConfig(),
    )


def _trace(gym_task, steps: list[dict], final_answer: dict):
    vf, FDAGymState, *_rest = _prime()
    return vf.Trace(
        task=vf.TraceTask(type="FDAGymTask", data=gym_task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        state=FDAGymState(done=True, steps=steps, final_answer=final_answer),
    )


def _score(gym_task, steps: list[dict], final_answer: dict):
    trace = _trace(gym_task, copy.deepcopy(steps), copy.deepcopy(final_answer))
    asyncio.run(gym_task.score(trace))
    return trace


def _without_cited_ids(steps: list[dict], cited: set[str]) -> list[dict]:
    cloned = copy.deepcopy(steps)

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key in _ID_KEYS:
                if node.get(key) in cited:
                    node[key] = "gone"
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    for step in cloned:
        walk(step.get("output"))
    return cloned


@pytest.fixture(scope="module")
def oracle():
    graded = play_public_adapter_episode(DEFAULT_TASK_ID, public_path=PUBLIC, snapshot_root=SNAPSHOT)
    env, task = _replay()
    gym_task = _gym_task(task)
    trace = _score(gym_task, env.steps, env.final_answer)
    return {
        "graded": graded,
        "task": task,
        "gym_task": gym_task,
        "steps": env.steps,
        "final_answer": env.final_answer,
        "trace": trace,
    }


def test_oracle_trace_publishes_four_weighted_terms(oracle):
    graded = oracle["graded"]
    trace = oracle["trace"]
    assert graded["grader_version"] == GRADER_VERSION == "5"
    assert graded["total"] == 1.0
    assert set(trace.rewards) == set(TERMS)
    assert "score_total" not in trace.rewards
    assert set(trace.metrics).isdisjoint({*TERMS, "fdagym"})
    for name in TERMS:
        reward = trace.rewards[name]
        assert reward.weight == WEIGHTS[name]
        assert reward.score == graded[name] == 1.0
        assert reward.value == reward.score * reward.weight
    assert round(trace.reward, 10) == graded["total"]
    for name in ("citations_valid", "citations_total", "hops", "answer_correctness"):
        assert name not in trace.rewards

    asyncio.run(oracle["gym_task"].finalize(trace, None))
    info = trace.info["fdagym"]
    assert info["grader_version"] == "5"
    assert info["total"] == graded["total"]
    for name in TERMS:
        assert info[name] == graded[name]


def test_wrong_answer_keeps_citation_and_drops_correctness_and_efficiency(oracle):
    honest = oracle["trace"]
    final_answer = copy.deepcopy(oracle["final_answer"])
    final_answer["answer"] = "not-a-drug"
    wrong = _score(oracle["gym_task"], oracle["steps"], final_answer)

    assert wrong.rewards["citation"].score == honest.rewards["citation"].score == 1.0
    assert wrong.rewards["format"].score == honest.rewards["format"].score == 1.0
    assert wrong.rewards["correctness"].score == 0.0
    assert honest.rewards["correctness"].score == 1.0
    assert wrong.rewards["efficiency"].score == 0.0
    assert honest.rewards["efficiency"].score == 1.0
    drop = honest.reward - wrong.reward
    expected = honest.rewards["correctness"].value + honest.rewards["efficiency"].value
    assert wrong.reward < honest.reward
    assert abs(drop - expected) <= 1e-12


def test_missing_cited_id_drops_citation_and_gates_correctness(oracle):
    honest = oracle["trace"]
    final_answer = oracle["final_answer"]
    cited = {citation["id"] for citation in final_answer["citations"]}
    assert cited
    stripped = _without_cited_ids(oracle["steps"], cited)
    dropped = _score(oracle["gym_task"], stripped, final_answer)
    graded = grade_steps(oracle["task"], stripped, final_answer)

    assert dropped.rewards["citation"].score == 0.0
    assert honest.rewards["citation"].score == 1.0
    assert dropped.rewards["correctness"].score == 0.0
    assert honest.rewards["correctness"].score == 1.0
    assert dropped.rewards["format"].score == 1.0
    assert final_answer["answer"] == oracle["final_answer"]["answer"]
    assert any("id did not appear" in note for note in graded["notes"])
    assert any(note == "correctness gated: no valid citation" for note in graded["notes"])
    assert dropped.reward < honest.reward
