"""Env default, hard solver, and task schema share one tool-call cap."""

from fdagym.env import TOOL_BUDGET, FDAEnv, tool_hard_cap
from fdagym.schemas import load as load_schema
from fdagym.tasks.hard_solver import BUDGET
from tests.env.support import ALL_TASKS, JsonSnapshot


def test_env_solver_and_schema_share_one_budget():
    schema = load_schema("task.schema.json")
    schema_max = schema["properties"]["oracle_calls"]["maxItems"]
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    assert env.tool_budget == TOOL_BUDGET
    assert TOOL_BUDGET == BUDGET == schema_max
    assert TOOL_BUDGET >= 8


def test_eight_calls_are_not_truncated_at_six():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    env.reset("FG-0001")
    action = {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
    for index in range(8):
        obs, _reward, done, _info = env.step(action)
        assert done is False, f"call {index + 1} ended the episode"
        assert obs["error"] is None
        assert env.tool_calls == index + 1
    assert env.tool_calls == 8
    ninth, _reward, done, _info = env.step(action)
    assert done is False
    assert ninth["error"] is None
    for _ in range(tool_hard_cap(TOOL_BUDGET) - 9):
        obs, _reward, done, _info = env.step(action)
        assert done is False
        assert obs["error"] is None
    blocked, _reward, done, _info = env.step(action)
    assert done is True
    assert blocked["error"] == "tool budget exhausted"
