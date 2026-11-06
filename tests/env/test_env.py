import json

import pytest

from fdagym.env import TOOL_BUDGET, FDAEnv, tool_hard_cap
from tests.env.support import ALL_TASKS, WARFARIN_SET, JsonSnapshot, cite, drive, submit
from tests.test_schemas import validator


def test_reset_shows_the_question_and_the_five_tools():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    obs = env.reset("FG-0001")
    assert "warfarin" in obs["question"].lower()
    assert [tool["function"]["name"] for tool in obs["tools"]] == [
        "search_drug_label",
        "get_label_section",
        "query_adverse_events",
        "search_approvals",
        "submit_answer",
    ]


def test_boxed_warning_episode_scores_one_and_matches_the_trajectory_schema():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS, run_id="run-test", seed=0)
    env.reset("FG-0001")
    obs, reward, done, _info = env.step(
        {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}}
    )
    assert done is False and reward == 0.0
    assert obs["tool_calls_remaining"] == TOOL_BUDGET - 1
    _obs, _reward, done, _info = env.step(
        {"tool": "get_label_section", "arguments": {"set_id": WARFARIN_SET, "section": "boxed_warning"}}
    )
    text = _obs["output"]["boxed_warning"]
    quote = "Warfarin sodium can cause major or fatal bleeding"
    assert quote in text
    _obs, reward, done, info = env.step(
        submit(
            "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
            [cite("drug/label", WARFARIN_SET, "boxed_warning", quote)],
        )
    )
    assert done is True
    assert reward == 1.0
    assert info["reward"]["efficiency"] == 1.0
    validator("trajectory.schema.json").validate(env.trajectory)
    validator("answer.schema.json").validate(env.trajectory["final_answer"])


def test_local_tasks_match_the_task_schema():
    check = validator("task.schema.json")
    for task in ALL_TASKS:
        check.validate(task)


def test_hard_cap_is_twice_the_advertised_budget():
    assert tool_hard_cap(TOOL_BUDGET) == 16
    assert tool_hard_cap(2) == 4
    default = FDAEnv(JsonSnapshot(), ALL_TASKS)
    assert default.tool_budget == TOOL_BUDGET == 8
    assert default.tool_hard_cap == 16
    custom = FDAEnv(JsonSnapshot(), ALL_TASKS, tool_budget=2)
    custom.reset("FG-0001")
    assert custom.tool_hard_cap == 4
    action = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    for _ in range(4):
        _obs, _reward, done, _info = custom.step(action)
        assert done is False
    blocked, _reward, done, _info = custom.step(action)
    assert done is True
    assert blocked["error"] == "tool budget exhausted"


def test_submit_still_works_after_the_tool_budget():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    action = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    for _ in range(env.tool_hard_cap):
        _obs, _reward, done, _info = env.step(action)
        assert done is False
    _obs, reward, done, _info = env.step(submit("unknown", []))
    assert done is True
    assert env.trajectory["terminated_by"] == "submit"
    assert reward == pytest.approx(0.1)


def test_configured_budget_of_eight_still_accepts_submit_and_rejects_the_ninth_call():
    action = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    env = FDAEnv(JsonSnapshot(), ALL_TASKS, tool_budget=8)
    env.reset("FG-0001")
    obs, _reward, done, _info = env.step(action)
    assert done is False
    assert obs["tool_calls_remaining"] == 7
    for _ in range(7):
        _obs, _reward, done, _info = env.step(action)
        assert done is False
    ninth, _reward, done, _info = env.step(action)
    assert done is False
    assert ninth["error"] is None
    for _ in range(7):
        _obs, _reward, done, _info = env.step(action)
        assert done is False
    assert env.tool_calls == env.tool_hard_cap == 16
    blocked, _reward, done, _info = env.step(action)
    assert done is True
    assert blocked["error"] == "tool budget exhausted"
    assert blocked["tool_calls_remaining"] == 0

    fresh = FDAEnv(JsonSnapshot(), ALL_TASKS, tool_budget=8)
    fresh.reset("FG-0001")
    for _ in range(fresh.tool_hard_cap):
        _obs, _reward, done, _info = fresh.step(action)
        assert done is False
    _obs, _reward, done, _info = fresh.step(submit("unknown", []))
    assert done is True
    assert fresh.trajectory["terminated_by"] == "submit"
    with pytest.raises(ValueError, match="tool_budget"):
        FDAEnv(JsonSnapshot(), ALL_TASKS, tool_budget=0)


def test_call_past_the_budget_stops_without_running_the_tool():
    class CountingSnapshot(JsonSnapshot):
        def __init__(self):
            super().__init__()
            self.n_get = 0

        def get(self, *args, **kwargs):
            self.n_get += 1
            return super().get(*args, **kwargs)

    snapshot = CountingSnapshot()
    env = FDAEnv(snapshot, ALL_TASKS)
    env.reset("FG-0001")
    action = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    for _ in range(TOOL_BUDGET + 1):
        obs, _reward, done, _info = env.step(action)
        assert done is False
        assert obs["error"] is None
    for _ in range(env.tool_hard_cap - (TOOL_BUDGET + 1)):
        env.step(action)
    assert env.tool_calls == 16
    hits = snapshot.n_get
    obs, reward, done, _info = env.step({"tool": "search_drug_label", "arguments": {"generic_name": "aspirin"}})
    assert snapshot.n_get == hits
    assert done is True
    assert obs["error"] == "tool budget exhausted"
    assert reward == 0.0
    assert env.trajectory["terminated_by"] == "max_steps"
    assert env.trajectory["final_answer"] is None
    validator("trajectory.schema.json").validate(env.trajectory)


def test_unanswerable_not_found_scores_one_only_after_an_empty_search():
    _obs, reward, _env = drive(
        "FG-0004",
        [
            {
                "tool": "query_adverse_events",
                "arguments": {"generic_name": "lecanemab", "mode": "count", "year": 2020},
            },
            submit("NOT_FOUND", [cite("drug/event", "step:1", "meta.results.total", "0")]),
        ],
    )
    assert _obs[0]["output"]["meta"]["results"]["total"] == 0
    assert reward["total"] == 1.0
    assert reward["correctness"] == 1.0


def test_invalid_json_is_logged_as_invalid_action():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    obs, _reward, done, _info = env.step("{not json")
    assert done is True
    assert obs["tool"] == "invalid_action"
    assert obs["error"] is not None and "invalid JSON" in obs["error"]
    assert env.trajectory["steps"][-1]["tool"] == "invalid_action"
    assert env.trajectory["terminated_by"] == "error"
    assert env.trajectory["final_answer"] is None
    validator("trajectory.schema.json").validate(env.trajectory)


def test_label_open_is_rejected_on_reset():
    task = json.loads(json.dumps(ALL_TASKS[0]))
    task["id"] = "FG-0099"
    task["type"] = "label_open"
    task["gold"] = {"kind": "judge", "value": "anything"}
    env = FDAEnv(JsonSnapshot(), [task])
    with pytest.raises(ValueError, match="label_open"):
        env.reset("FG-0099")


def test_step_records_latency_ms():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    env.step({"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}, latency_ms=12.4)
    assert env.steps[-1]["latency_ms"] == 12.4
