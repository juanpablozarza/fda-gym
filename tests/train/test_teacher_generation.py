import json

from fdagym.agents.litellm_agent import SYSTEM_PROMPT
from fdagym.train.distill import parse_teacher_trajectory
from fdagym.train.distill_sft import build_distill_rows
from fdagym.train.rollout_dry import CharTokenizer
from fdagym.train.teacher_generation import (
    generate_teacher_trajectories,
    prefixed_model,
    teacher_row,
    trajectory_messages,
)
from tests.agents.fake import ScriptedCompletion
from tests.env.support import EXAMPLE_TASK, JsonSnapshot

TASK = {
    "id": "FG-0001",
    "question": "Does warfarin have a boxed warning?",
    "type": "boxed_warning",
    "variant": "",
    "snapshot_version": "2026-10-15",
}


def _trajectory(*, cost=0.01, correctness=1.0):
    return {
        "steps": [
            {
                "index": 1,
                "tool": "search_drug_label",
                "arguments": {"generic_name": "warfarin"},
                "output": {"results": [{"set_id": "set-1"}]},
                "error": None,
            },
            {
                "index": 2,
                "tool": "submit_answer",
                "arguments": {"answer": True, "citations": []},
                "output": None,
                "error": None,
            },
        ],
        "reward": {
            "total": correctness,
            "correctness": correctness,
            "citation": 1.0,
            "format": 1.0,
            "efficiency": 1.0,
            "citations_valid": 1,
            "citations_total": 1,
            "notes": [],
        },
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost_usd": cost, "latency_ms": 12},
    }


def test_messages_are_replayable_action_observation_pairs():
    messages = trajectory_messages(TASK, _trajectory(), tool_budget=8)
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == '{"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}'
    observation = json.loads(messages[3]["content"])
    assert observation["tool_calls_remaining"] == 7
    assert observation["output"]["results"][0]["set_id"] == "set-1"
    assert json.loads(messages[4]["content"])["tool"] == "submit_answer"


def test_teacher_rows_validate_and_derive_pass_from_reward():
    row = teacher_row(TASK, _trajectory(), teacher_model="openrouter/anthropic/claude-sonnet-4.5", attempt_index=0, tool_budget=8)
    assert row["passed"] is True
    assert row["template"] == "boxed_warning"
    failed = teacher_row(
        TASK,
        _trajectory(correctness=0.0),
        teacher_model="openrouter/anthropic/claude-sonnet-4.5",
        attempt_index=1,
        tool_budget=8,
    )
    assert failed["passed"] is False


def test_generation_resumes_by_teacher_question_attempt_and_enforces_cost_cap(tmp_path):
    out = tmp_path / "teachers.jsonl"
    calls = []

    def episode_runner(agent, env, task_id):
        calls.append((agent.model, task_id))
        return _trajectory()

    kwargs = {
        "tasks": [TASK],
        "snapshot": JsonSnapshot(),
        "models": ["anthropic/claude-sonnet-4.5", "openrouter/anthropic/claude-haiku-4.5"],
        "output": out,
        "episode_runner": episode_runner,
        "tool_budget": 8,
        "cost_cap_usd": 1.0,
    }
    first = generate_teacher_trajectories(**kwargs)
    assert first["added"] == 2
    assert first["passed"] == 2
    assert calls == [
        ("openrouter/anthropic/claude-sonnet-4.5", "FG-0001"),
        ("openrouter/anthropic/claude-haiku-4.5", "FG-0001"),
    ]
    second = generate_teacher_trajectories(**kwargs)
    assert second["added"] == 0
    assert second["skipped"] == 2
    assert len(out.read_text().splitlines()) == 2
    assert prefixed_model("openrouter/qwen/model") == "openrouter/qwen/model"


def test_cost_cap_stops_before_the_next_teacher_episode(tmp_path):
    calls = []

    def episode_runner(agent, env, task_id):
        calls.append(task_id)
        return _trajectory(cost=0.6)

    result = generate_teacher_trajectories(
        tasks=[TASK, {**TASK, "id": "FG-0002"}, {**TASK, "id": "FG-0003"}],
        snapshot=JsonSnapshot(),
        models=["anthropic/claude-sonnet-4.5"],
        output=tmp_path / "teachers.jsonl",
        episode_runner=episode_runner,
        cost_cap_usd=1.0,
    )
    assert calls == ["FG-0001", "FG-0002"]
    assert result["added"] == 2
    assert result["spent_usd"] == 1.2
    assert "cost cap" in result["aborted"]


def test_generator_runs_the_real_agent_loop_with_v4_budget_prompt(tmp_path):
    scripted = [
        {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}},
        {
            "tool": "get_label_section",
            "arguments": {"set_id": "0cbce382-9c88-4f58-ae0f-532a841e8f95", "section": "boxed_warning"},
        },
        {"tool": "submit_answer", "arguments": {"answer": "NOT_FOUND", "citations": []}},
    ]
    completion = ScriptedCompletion(scripted)
    seen_prompt = []

    def episode_runner(agent, env, task_id):
        agent.completion = completion
        seen_prompt.append(agent.system_prompt)
        return agent.run(env, task_id)

    result = generate_teacher_trajectories(
        tasks=[EXAMPLE_TASK],
        snapshot=JsonSnapshot(),
        models=["anthropic/claude-sonnet-5.5"],
        output=tmp_path / "real-loop.jsonl",
        episode_runner=episode_runner,
    )
    assert result["added"] == 1
    assert seen_prompt[0] != SYSTEM_PROMPT
    assert "at most 8 tool calls" in seen_prompt[0]
    assert completion.n == 3
    row = json.loads((tmp_path / "real-loop.jsonl").read_text())
    assert row["question_id"] == EXAMPLE_TASK["id"]
    assert [message["role"] for message in row["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "assistant"
    ]
    parsed = parse_teacher_trajectory(row)
    assert build_distill_rows([parsed], CharTokenizer())["n_kept"] == 1
