from pathlib import Path

import pytest

from fdagym.agents import litellm_agent as agent_mod
from fdagym.agents.litellm_agent import FORCED_SUBMIT, LiteLLMAgent
from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.openfda import DiskSnapshot
from tests.agents.fake import ScriptedCompletion
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot
from tests.integration.test_oracle import citations_from_outputs, gold_answer
from tests.test_schemas import validator

SNAPSHOT = Path("data/snapshots/2026-10-15")
TEST_TASKS = Path("data/tasks/test.jsonl")


def _oracle_calls(snapshot, tasks, task) -> list[dict]:
    env = FDAEnv(snapshot, tasks)
    env.reset(task["id"])
    for call in task["oracle_calls"]:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        assert obs["error"] is None and done is False
    citations = citations_from_outputs(env.steps, task)
    return list(task["oracle_calls"]) + [
        {"tool": "submit_answer", "arguments": {"answer": gold_answer(task), "citations": citations}}
    ]


def _run(snapshot, tasks, task, completion, run_id="agent-test", seed=1):
    env = FDAEnv(snapshot, tasks, run_id=run_id, seed=seed)
    agent = LiteLLMAgent("scripted-oracle", completion=completion)
    return agent.run(env, task["id"]), env


def test_scripted_oracle_on_fixture_validates_and_scores_one():
    snapshot = JsonSnapshot()
    calls = _oracle_calls(snapshot, ALL_TASKS, EXAMPLE_TASK)
    completion = ScriptedCompletion(calls)
    trajectory, env = _run(snapshot, ALL_TASKS, EXAMPLE_TASK, completion)
    validator("trajectory.schema.json").validate(trajectory)
    assert trajectory["reward"]["total"] == 1.0
    assert trajectory["terminated_by"] == "submit"
    assert trajectory["model"] == {"name": "scripted-oracle", "kind": "baseline", "checkpoint_step": None}
    assert trajectory["usage"]["prompt_tokens"] == 12 * len(calls)
    assert trajectory["usage"]["completion_tokens"] == 8 * len(calls)
    assert trajectory["usage"]["cost_usd"] == pytest.approx(0.0001 * len(calls))
    assert completion.n == len(calls)
    assert env.tool_calls == len(EXAMPLE_TASK["oracle_calls"])


def test_budget_forces_submit_answer():
    snapshot = JsonSnapshot()
    searches = [{"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}] * TOOL_BUDGET
    extra = [{"tool": "search_drug_label", "arguments": {"generic_name": "aspirin"}}]
    completion = ScriptedCompletion(searches + extra)
    trajectory, env = _run(snapshot, ALL_TASKS, EXAMPLE_TASK, completion, run_id="force-submit")
    assert trajectory["terminated_by"] == "submit"
    assert trajectory["final_answer"] == FORCED_SUBMIT
    assert trajectory["reward"]["format"] == 1.0
    assert trajectory["reward"]["correctness"] == 0.0
    assert env.tool_calls == TOOL_BUDGET
    assert completion.kwargs[-1]["tools"][0]["function"]["name"] == "submit_answer"


def test_custom_budget_offers_submit_without_tool_choice():
    snapshot = JsonSnapshot()
    completion = ScriptedCompletion(
        [{"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}] * 2
        + [{"tool": "submit_answer", "arguments": {"answer": "NOT_FOUND", "citations": []}}]
    )
    env = FDAEnv(snapshot, ALL_TASKS, run_id="custom-budget", seed=1, tool_budget=2)
    trajectory = LiteLLMAgent("scripted", completion=completion).run(env, EXAMPLE_TASK["id"])
    assert trajectory["terminated_by"] == "submit"
    assert len(completion.kwargs) == 3
    assert "at most 2 tool calls" in completion.kwargs[0]["messages"][0]["content"]
    assert [tool["function"]["name"] for tool in completion.kwargs[-1]["tools"]] == ["submit_answer"]
    assert "tool_choice" not in completion.kwargs[-1]


def test_provider_error_text_is_recorded_in_trajectory():
    def fail(**_kwargs):
        raise TimeoutError("provider rejected forced tool_choice")

    trajectory = agent_mod.run_episode(
        FDAEnv(JsonSnapshot(), ALL_TASKS, run_id="provider-error", seed=1),
        EXAMPLE_TASK["id"],
        model="scripted",
        completion=fail,
    )
    assert trajectory["terminated_by"] == "error"
    assert "provider rejected forced tool_choice" in trajectory["steps"][-1]["error"]
    assert trajectory["final_answer"] is None


def test_correct_answer_without_citations_scores_format_only():
    snapshot = JsonSnapshot()
    completion = ScriptedCompletion(
        [
            {
                "tool": "submit_answer",
                "arguments": {
                    "answer": "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
                    "citations": [],
                },
            }
        ]
    )
    trajectory, _env = _run(snapshot, ALL_TASKS, EXAMPLE_TASK, completion, run_id="no-cite")
    assert trajectory["reward"]["total"] == pytest.approx(0.10)
    assert trajectory["reward"]["correctness"] == 0.0
    assert trajectory["reward"]["format"] == 1.0
    assert trajectory["reward"]["citation"] == 0.0


def test_scripted_oracle_on_pilot_test_task(tmp_path):
    if not (SNAPSHOT / "manifest.jsonl").exists():
        pytest.skip("pilot snapshot absent")
    import json

    tasks = [json.loads(line) for line in TEST_TASKS.read_text().splitlines() if line.strip()]
    task = next(row for row in tasks if row["id"] == "FG-0016")
    snapshot = DiskSnapshot("2026-10-15")
    calls = _oracle_calls(snapshot, tasks, task)
    completion = ScriptedCompletion(calls)
    env = FDAEnv(snapshot, tasks, run_id="pilot-oracle", seed=1)
    trajectory = LiteLLMAgent("scripted-oracle", completion=completion).run(env, task["id"])
    validator("trajectory.schema.json").validate(trajectory)
    assert trajectory["reward"]["total"] == 1.0
    out = tmp_path / "pilot.jsonl"
    out.write_text(json.dumps(trajectory) + "\n")
    assert json.loads(out.read_text())["task_id"] == "FG-0016"


def test_text_submit_without_tool_call_is_accepted():
    import json
    from types import SimpleNamespace

    snapshot = JsonSnapshot()
    answer = {
        "answer": "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
        "citations": [],
    }

    def complete(**_kwargs):
        message = SimpleNamespace(content=json.dumps(answer), tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            _hidden_params={"response_cost": 0.0},
        )

    trajectory, _env = _run(snapshot, ALL_TASKS, EXAMPLE_TASK, complete, run_id="text-submit")
    assert trajectory["terminated_by"] == "submit"
    assert trajectory["final_answer"]["answer"] == answer["answer"]
