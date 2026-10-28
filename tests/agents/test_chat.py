"""Offline tests for the Chat Completions episode loop. No snapshot disk, no network."""

import json
from types import SimpleNamespace

from fdagym.agents.chat import PUBLIC_PROMPT, run_chat_episode
from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.tools import TOOLS
from tests.agents.fake import ScriptedCompletion
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot
from tests.integration.test_oracle import citations_from_outputs, gold_answer
from tests.test_schemas import validator


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


def _run(snapshot, tasks, task, completion, run_id="chat-test", seed=0, model="scripted-oracle"):
    env = FDAEnv(snapshot, tasks, run_id=run_id, seed=seed)
    trajectory = run_chat_episode(env, task["id"], completion, model=model)
    return trajectory, env


def test_scripted_oracle_on_fixture_scores_one_and_validates():
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
    assert trajectory["usage"]["cost_usd"] == 0.0
    assert env.tool_calls == len(EXAMPLE_TASK["oracle_calls"])


def test_tool_message_matches_previous_call_and_first_round_uses_tools():
    snapshot = JsonSnapshot()
    calls = _oracle_calls(snapshot, ALL_TASKS, EXAMPLE_TASK)
    completion = ScriptedCompletion(calls)
    _run(snapshot, ALL_TASKS, EXAMPLE_TASK, completion, run_id="chat-protocol")
    first = completion.kwargs[0]
    assert first["tools"] == TOOLS
    assert first["messages"][0] == {"role": "system", "content": PUBLIC_PROMPT}
    assert first["messages"][1] == {"role": "user", "content": EXAMPLE_TASK["question"]}
    second = completion.kwargs[1]
    assistant = next(row for row in second["messages"] if row["role"] == "assistant")
    tool_messages = [row for row in second["messages"] if row["role"] == "tool"]
    assert tool_messages
    assert tool_messages[0]["tool_call_id"] == assistant["tool_calls"][0]["id"]
    body = json.loads(tool_messages[0]["content"])
    assert "tool_calls_remaining" not in body
    assert "results" in body


def test_public_prompt_is_short_and_does_not_teach_citation_law():
    lines = [line for line in PUBLIC_PROMPT.splitlines() if line.strip()]
    assert len(lines) <= 6
    assert "step:<" not in PUBLIC_PROMPT
    assert "Quotes shorter" not in PUBLIC_PROMPT
    assert "worked citation" not in PUBLIC_PROMPT


def test_searches_until_budget_exhausted_finish_with_submit():
    snapshot = JsonSnapshot()
    searches = [{"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}] * TOOL_BUDGET
    extra = [{"tool": "search_drug_label", "arguments": {"generic_name": "aspirin"}}]
    completion = ScriptedCompletion(searches + extra)
    trajectory, env = _run(snapshot, ALL_TASKS, EXAMPLE_TASK, completion, run_id="chat-budget")
    assert trajectory["terminated_by"] == "submit"
    assert trajectory["steps"][-1]["tool"] == "submit_answer"
    assert trajectory["final_answer"] == {"answer": "NOT_FOUND", "citations": []}
    assert env.tool_calls == TOOL_BUDGET
    assert not any(step["arguments"].get("generic_name") == "aspirin" for step in trajectory["steps"])
    forced = completion.kwargs[-1]
    assert [tool["function"]["name"] for tool in forced["tools"]] == ["submit_answer"]
    assert forced["tool_choice"] == {"type": "function", "function": {"name": "submit_answer"}}


def test_parallel_calls_past_the_budget_are_not_stepped():
    snapshot = JsonSnapshot()
    seen = []

    def complete(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            tool_calls = [
                SimpleNamespace(
                    id=f"call_{index}",
                    type="function",
                    function=SimpleNamespace(
                        name="search_drug_label",
                        arguments=json.dumps({"generic_name": "warfarin"}),
                    ),
                )
                for index in range(TOOL_BUDGET + 1)
            ]
            message = SimpleNamespace(content="batch", tool_calls=tool_calls)
        else:
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_submit",
                        type="function",
                        function=SimpleNamespace(
                            name="submit_answer",
                            arguments=json.dumps({"answer": "NOT_FOUND", "citations": []}),
                        ),
                    )
                ],
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    trajectory, env = _run(snapshot, ALL_TASKS, EXAMPLE_TASK, complete, run_id="chat-parallel", model="scripted")
    assert trajectory["terminated_by"] == "submit"
    assert env.tool_calls == TOOL_BUDGET
    assert len(trajectory["steps"]) == TOOL_BUDGET + 1
    errors = [
        json.loads(row["content"])
        for row in seen[1]["messages"]
        if row["role"] == "tool" and "tool budget exhausted" in row["content"]
    ]
    assert errors == [{"error": "tool budget exhausted; call submit_answer"}]
    assistant = next(row for row in seen[1]["messages"] if row["role"] == "assistant")
    error_message = next(row for row in seen[1]["messages"] if row["content"].startswith('{"error"'))
    assert error_message["tool_call_id"] == assistant["tool_calls"][-1]["id"]


def test_json_text_without_a_tool_call_is_submitted():
    answer = {
        "answer": "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
        "citations": [],
    }

    def complete(**_kwargs):
        message = SimpleNamespace(content=json.dumps(answer), tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    trajectory, _env = _run(JsonSnapshot(), ALL_TASKS, EXAMPLE_TASK, complete, run_id="chat-text")
    assert trajectory["terminated_by"] == "submit"
    assert trajectory["final_answer"]["answer"] == answer["answer"]


def test_plain_text_without_a_tool_call_still_submits():
    def complete(**_kwargs):
        message = SimpleNamespace(content="The label mentions bleeding.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    trajectory, _env = _run(JsonSnapshot(), ALL_TASKS, EXAMPLE_TASK, complete, run_id="chat-prose")
    assert trajectory["terminated_by"] == "submit"
    assert trajectory["final_answer"] == {"answer": "NOT_FOUND", "citations": []}
    assert trajectory["usage"]["prompt_tokens"] == 0
    assert trajectory["usage"]["completion_tokens"] == 0
    assert trajectory["usage"]["cost_usd"] == 0.0
