"""JP's 2026-10-19 tool-run decisions: type mix, composite half credit, T=0.8, tied-group logging, parse stop."""

import collections
from pathlib import Path
from types import SimpleNamespace

import pytest

from fdagym.agents.baseline import load_split
from fdagym.rewards import score
from fdagym.rewards.dense import COMPOSITE_HALF, PUNCT, composite_halves
from fdagym.train.curriculum import TOOL_TYPE_MIX, filter_pool, stage_pool, type_mix_pool
from fdagym.train.grpo_tools import build_tool_rows
from fdagym.train.rollout import ASSISTANT_TURN, QwenToolCodec, RolloutLimits
from fdagym.train.tool_config import ToolTrainConfig, tool_train_config
from fdagym.train.tool_guard import ParseRateGuard, ParseRateStop
from fdagym.train.tool_score import episode_metrics, frac_zero_std
from tests.env.support import ALL_TASKS
from tests.train.test_rollout import _run

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def train_tasks():
    return load_split("train")


# ---------- type mix ----------


def test_type_mix_hits_the_approved_shares(train_tasks):
    rows, mix = type_mix_pool(train_tasks)
    assert mix["n_rows"] == len(rows) == 600
    assert mix["rows"] == {
        "boxed_warning": 257,
        "faers_top_reaction": 171,
        "approval": 129,
        "faers_count": 43,
    }
    kinds = collections.Counter(task["type"] for task in rows)
    assert kinds == {
        "boxed_warning": 257,
        "faers_top_reaction": 171,
        "approval": 129,
        "faers_count": 43,
    }
    assert "composite" not in kinds
    assert all(task.get("split") != "test" and task["type"] != "label_open" for task in rows)


def test_type_mix_uses_every_task_before_repeating(train_tasks):
    rows, mix = type_mix_pool(train_tasks)
    boxed = {task["id"] for task in rows if task["type"] == "boxed_warning"}
    assert len(boxed) == mix["unique"]["boxed_warning"] == 106  # 257 rows cover all 106
    approval = [task["id"] for task in rows if task["type"] == "approval"]
    assert len(set(approval)) == len(approval) == 129  # 129 of 476, no repeats


def test_type_mix_is_seeded(train_tasks):
    first, _ = type_mix_pool(train_tasks, seed=0)
    again, _ = type_mix_pool(train_tasks, seed=0)
    other, _ = type_mix_pool(train_tasks, seed=1)
    assert [task["id"] for task in first] == [task["id"] for task in again]
    assert [task["id"] for task in first] != [task["id"] for task in other]


def test_type_mix_redistributes_missing_types(train_tasks):
    subset = [task for task in train_tasks if task["type"] in {"approval", "boxed_warning"}]
    _rows, mix = type_mix_pool(subset, n_rows=90)
    # approval 0.15 and boxed 0.30 renormalize to 1/3 and 2/3.
    assert mix["rows"] == {"approval": 30, "boxed_warning": 60}
    with pytest.raises(RuntimeError, match="empty"):
        type_mix_pool([task for task in train_tasks if task["type"] == "label_open"])
    with pytest.raises(ValueError):
        type_mix_pool(train_tasks, n_rows=0)


def test_stage_three_is_not_the_lever(train_tasks):
    stage1, _ = stage_pool(train_tasks, 1)
    stage3, _ = stage_pool(train_tasks, 3)
    boxed = lambda pool: sum(1 for task in pool if task["type"] == "boxed_warning")
    assert boxed(stage1) == 106
    assert boxed(stage3) == 26
    assert len(filter_pool(train_tasks, 1)) == 1160


def test_tool_rows_follow_the_mix_and_stay_question_only(train_tasks):
    rows, mix = build_tool_rows(train_tasks, tool_train_config(), QwenToolCodec(), type_mix=TOOL_TYPE_MIX)
    assert len(rows) == 600
    assert mix["rows"]["boxed_warning"] == 257
    for row in rows[:50]:
        assert row["prompt"].endswith(ASSISTANT_TURN)
        assert "<tool_response>" not in row["prompt"]
    plan_a_rows, _ = build_tool_rows(ALL_TASKS, tool_train_config(), QwenToolCodec())
    assert [row["task_id"] for row in plan_a_rows] == ["FG-0001", "FG-0002", "FG-0003"]


# ---------- config ----------


def test_tool_config_has_the_approved_knobs():
    tool = ToolTrainConfig()
    assert tool.train.temperature == 0.65
    assert tool.limits.temperature == 0.65
    assert tool.train.grpo_kwargs("/tmp/x")["temperature"] == 0.65
    assert tool.train.num_generations == 8
    assert tool.type_mix == TOOL_TYPE_MIX
    assert "composite" not in tool.type_mix
    assert sum(tool.type_mix.values()) == pytest.approx(0.7)
    assert tool.pool_types == ("boxed_warning", "faers_top_reaction", "approval", "faers_count")
    assert tool.parse_stop_steps == 20
    assert tool.parse_stop_min == 0.9
    with pytest.raises(ValueError, match="temperature"):
        ToolTrainConfig(limits=RolloutLimits(temperature=0.7))


def test_modal_and_script_pass_the_mix_and_the_guard():
    jobs = (ROOT / "fdagym" / "train" / "jobs.py").read_text()
    assert "type_mix=tool.type_mix" in jobs
    assert "ParseRateGuard(window=tool.parse_stop_steps, min_rate=tool.parse_stop_min)" in jobs
    for relative in ("modal_train.py", "scripts/run_gpu.py"):
        text = (ROOT / relative).read_text()
        assert "run_train_tools(" in text
    dry = (ROOT / "scripts" / "grpo_tools.py").read_text()
    assert "--train" not in dry
    assert "run_gpu.py" in dry
    source = (ROOT / "fdagym" / "train" / "grpo_tools.py").read_text()
    assert '"frac_reward_zero_std"' in source
    assert "episode_metrics(rollouts)" in source
    assert "adapter-stopped" in source
    assert "raise ParseRateStop" in source


# ---------- composite half credit ----------


def _composite(value="8714; yes"):
    return {"type": "composite", "gold": {"kind": "exact", "value": value}}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("8714; no", COMPOSITE_HALF),
        ("8000; yes", COMPOSITE_HALF),
        ("8000; no", 0.0),
        ("8,714; Yes", PUNCT),
        ("8714 yes", 0.0),
        (8714, 0.0),
    ],
)
def test_composite_halves(answer, expected):
    got, notes = composite_halves(answer, _composite())
    assert got == pytest.approx(expected)
    assert notes and notes[0].startswith("dense composite")


def test_composite_halves_ignores_other_tasks():
    assert composite_halves("8714; no", {"type": "approval", "gold": {"kind": "exact", "value": "8714; yes"}}) is None
    assert composite_halves("8714; no", {"type": "composite", "gold": {"kind": "number", "value": 8714}}) is None


def test_half_credit_is_dense_only_on_a_real_trajectory():
    from fdagym.env import FDAEnv
    from fdagym.openfda import DiskSnapshot
    from fdagym.train.context import play_oracle
    from fdagym.train.gold import gold_submit

    snapshot = DiskSnapshot("2026-10-15")
    tasks = load_split("test")
    task = next(task for task in tasks if task["id"] == "FG-0182")
    env = FDAEnv(snapshot, tasks, run_id="t")
    play_oracle(env, task)
    final = gold_submit(task, env.steps)
    assert final["answer"] == "8714; yes"
    final = {**final, "answer": "8714; no"}
    trajectory = {"steps": env.steps, "final_answer": final}
    strict = score(trajectory, task)
    dense = score(trajectory, task, dense=True, snapshot=snapshot)
    assert strict["correctness"] == 0.0
    assert strict["total"] == pytest.approx(0.35)
    assert dense["correctness"] == pytest.approx(0.5)
    # 0.60 * 0.5 + 0.25 + 0.10 + 0.05 * 1.0 (efficiency now counts: correctness > 0)
    assert dense["total"] == pytest.approx(0.70)


# ---------- metrics ----------


def test_frac_zero_std_counts_tied_groups():
    assert frac_zero_std([1.05] * 8, 8) == 1.0
    assert frac_zero_std([1.05] * 8 + [1.05, 0.4] * 4, 8) == 0.5
    assert frac_zero_std([0.1, 0.1, 0.2, 0.2], 2) == 1.0
    with pytest.raises(ValueError):
        frac_zero_std([1.0, 0.0, 1.0], 2)


def test_episode_metrics_split_terminations():
    def ep(terminated_by, closed_by=None, calls=1, turns=2, parsed=2):
        return SimpleNamespace(
            terminated_by=terminated_by,
            closed_by=closed_by,
            parsed_tool_calls=calls,
            n_turns=turns,
            parsed_turns=parsed,
        )

    batch = [
        ep("submit"),
        ep("submit", calls=3, turns=4, parsed=4),
        ep("error", calls=0, turns=1, parsed=0),  # malformed first turn
        ep("error", closed_by="turn_limit", calls=6, turns=7, parsed=7),
    ]
    out = episode_metrics(batch)
    assert out["episodes/terminated_submit"] == 0.5
    assert out["episodes/terminated_error"] == 0.25
    assert out["episodes/closed_turn_limit"] == 0.25
    assert out["episodes/closed_context"] == 0.0
    assert out["episodes/terminated_max_steps"] == 0.0
    assert out["rewards/parse_rate"] == 0.75
    assert out["rewards/turn_parse_rate"] == pytest.approx(13 / 14)
    assert out["episodes/tool_calls_mean"] == pytest.approx(10 / 4)
    assert episode_metrics([]) == {}


def test_rollout_records_parsed_turns_and_who_closed_it():
    rollouts, *_ = _run(["FG-0001", "FG-0002"], codec=QwenToolCodec())
    for rollout in rollouts:
        assert rollout.closed_by is None
        assert rollout.parsed_turns == rollout.n_turns
        assert rollout.terminated_by == "submit"
    short, *_ = _run(["FG-0001"], codec=QwenToolCodec(), limits=RolloutLimits(max_turns=1))
    assert short[0].closed_by == "turn_limit"
    assert short[0].terminated_by == "error"
    tight, *_ = _run(["FG-0001"], codec=QwenToolCodec(), limits=RolloutLimits(max_context_tokens=900))
    assert tight[0].closed_by == "context"
    metrics = episode_metrics(short + tight + rollouts)
    assert metrics["episodes/closed_turn_limit"] == 0.25
    assert metrics["episodes/closed_context"] == 0.25
    assert metrics["episodes/terminated_submit"] == 0.5


def test_tool_metrics_stay_on_the_trainer_log():
    """TRL already logs these. W&B records that log. There is no second curve file."""
    assert not (ROOT / "fdagym" / "train" / "curve.py").exists()
    grpo = (ROOT / "fdagym" / "train" / "grpo_tools.py").read_text()
    assert "training-curve" not in grpo
    assert "append_point" not in grpo
    assert 'self._metrics[mode]["rewards/env/mean"]' in grpo
    assert 'self._metrics[mode]["rewards/dense/mean"]' in grpo
    assert 'self._metrics[mode]["rewards/hops/mean"]' in grpo
    assert 'self._metrics[mode]["frac_reward_zero_std"]' in grpo
    assert "episode_metrics(rollouts)" in grpo
    assert "report_to" in grpo


# ---------- stop rule ----------


def _logs(rate, turn=1.0):
    return {"rewards/parse_rate": rate, "rewards/turn_parse_rate": turn}


def test_guard_passes_a_healthy_start():
    guard = ParseRateGuard()
    for step in range(1, 30):
        assert guard.observe(step, _logs(0.875 if step % 4 == 0 else 1.0)) is None
    assert guard.checked and guard.reason is None


def test_guard_stops_at_step_twenty_on_either_rate():
    guard = ParseRateGuard()
    for step in range(1, 20):
        assert guard.observe(step, _logs(0.8)) is None
    reason = guard.observe(20, _logs(0.8))
    assert reason and "rewards/parse_rate" in reason and "0.800" in reason
    assert guard.observe(21, _logs(1.0)) == reason
    turn_guard = ParseRateGuard()
    for step in range(1, 21):
        result = turn_guard.observe(step, _logs(1.0, turn=0.85))
    assert result and "turn_parse_rate" in result


def test_guard_is_not_rechecked_after_a_late_resume():
    guard = ParseRateGuard()
    assert guard.observe(51, _logs(0.1)) is None
    assert guard.checked
    with pytest.raises(ValueError):
        ParseRateGuard(window=0)
    with pytest.raises(ValueError):
        ParseRateGuard(min_rate=1.5)
    assert issubclass(ParseRateStop, RuntimeError)
