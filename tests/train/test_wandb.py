"""W&B is optional. No key, no crash. The key is never an argument and never logged."""

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

from fdagym.train.tool_guard import ParseRateGuard
from fdagym.train.wandb_chart import chart_series, render_chart
from fdagym.train.wandb_log import (
    attach_grpo_train_logs,
    build_run_config,
    eval_history_metrics,
    flatten_memorization,
    merge_memorization,
    resolve_wandb_target,
    resume_floor,
    scrub_config,
    should_skip_log,
    start_wandb,
    step_from_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]


class _Config:
    def __init__(self):
        self.updates = []

    def update(self, payload=None, *args, **kwargs):
        self.updates.append(payload)


class _Run:
    def __init__(self, *, resumed=False, summary_step=None):
        self.resumed = resumed
        self.summary = {} if summary_step is None else {"train/global_step": summary_step}
        self.config = _Config()
        self.id = None
        self.name = None
        self.url = "https://wandb.ai/lab/bio-rl/runs/grpo-qwen3b-tools"


class _Wandb:
    def __init__(self, *, resumed=False, summary_step=None, explode=False):
        self.run = None
        self.calls = []
        self.init_kwargs = None
        self.finished = 0
        self._resumed = resumed
        self._summary_step = summary_step
        self._explode = explode

    def init(self, **kwargs):
        if self._explode:
            raise RuntimeError("wandb down")
        self.init_kwargs = kwargs
        self.run = _Run(resumed=self._resumed, summary_step=self._summary_step)
        self.run.id = kwargs.get("id")
        self.run.name = kwargs.get("name")
        return self.run

    def log(self, data=None, step=None, commit=None, **kwargs):
        self.calls.append((data, step))

    def finish(self, *args, **kwargs):
        self.finished += 1


def _install(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_disabled_without_a_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    fake = _install(monkeypatch, _Wandb())
    session = start_wandb("grpo-qwen3b-tools", {"run_id": "grpo-qwen3b-tools"})
    assert session.enabled is False
    assert session.report_to == []
    assert session.project == "bio-rl"
    assert fake.init_kwargs is None
    session.log({"reward": 1})
    session.finish()
    assert fake.calls == []
    assert fake.finished == 0


def test_blank_key_is_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "   ")
    fake = _install(monkeypatch, _Wandb())
    session = start_wandb("local", {})
    assert session.enabled is False
    assert fake.init_kwargs is None


def test_missing_install_does_not_crash(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    session = start_wandb("local", {"run_id": "local"})
    assert session.enabled is False
    assert session.report_to == []


def test_init_outage_disables_the_run(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    fake = _install(monkeypatch, _Wandb(explode=True))
    session = start_wandb("grpo-qwen3b-tools", {"run_id": "grpo-qwen3b-tools"})
    assert session.enabled is False
    assert session.report_to == []
    assert fake.init_kwargs is None


def test_config_resume_and_project_overrides(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    monkeypatch.setenv("WANDB_ENTITY", "from-env-lab")
    fake = _install(monkeypatch, _Wandb())
    config = build_run_config(
        run_id="grpo-qwen3b-tools",
        job="grpo",
        prompt_version="v4",
        tool_budget=8,
        hard_train_mix={"hard": 0.6, "current": 0.4},
        seeds=[0],
        adapter="/vol/runs/sft-qwen3b-tools-v4/adapter",
        model="Qwen/Qwen2.5-3B-Instruct",
    )
    session = start_wandb(
        "grpo-qwen3b-tools",
        config,
        project="from-flag",
        entity="from-flag-lab",
        resume_step=50,
    )
    assert session.enabled is True
    assert session.report_to == ["wandb"]
    assert session.floor is None  # new run, even with a checkpoint
    kwargs = fake.init_kwargs
    assert kwargs["project"] == "from-flag"
    assert kwargs["entity"] == "from-flag-lab"
    assert kwargs["id"] == "grpo-qwen3b-tools"
    assert kwargs["name"] == "grpo-qwen3b-tools"
    assert kwargs["resume"] == "allow"
    logged = json.dumps(kwargs)
    assert "test-key" not in logged
    assert "WANDB_API_KEY" not in logged
    assert kwargs["config"]["prompt_version"] == "v4"
    assert kwargs["config"]["tool_budget"] == 8
    assert kwargs["config"]["hard_train_mix"]["hard"] == 0.6
    assert kwargs["config"]["seeds"] == [0]
    assert kwargs["config"]["adapter"].endswith("/adapter")
    assert kwargs["config"]["git_sha"]
    assert kwargs["config"]["model"] == "Qwen/Qwen2.5-3B-Instruct"
    session.finish()
    assert fake.finished == 1


def test_env_project_and_entity_when_flags_are_blank(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    monkeypatch.setenv("WANDB_ENTITY", "lab")
    assert resolve_wandb_target("", "") == ("from-env", "lab")
    assert resolve_wandb_target(None, None) == ("from-env", "lab")
    monkeypatch.delenv("WANDB_PROJECT")
    monkeypatch.delenv("WANDB_ENTITY")
    assert resolve_wandb_target(None, None) == ("bio-rl", None)


def test_resume_skips_steps_already_logged(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    fake = _install(monkeypatch, _Wandb(resumed=True, summary_step=73))
    session = start_wandb("grpo-qwen3b-tools", {"run_id": "grpo-qwen3b-tools"}, resume_step=50)
    assert session.floor == 73
    import wandb

    wandb.log({"train/global_step": 73, "train/reward": 0.9})
    wandb.log({"train/global_step": 74, "train/reward": 0.91})
    assert fake.calls == [({"train/global_step": 74, "train/reward": 0.91}, None)]


def test_log_outage_does_not_raise(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    fake = _install(monkeypatch, _Wandb())

    def boom(data=None, step=None, commit=None, **kwargs):
        raise RuntimeError("connection reset")

    fake.log = boom
    session = start_wandb("grpo-qwen3b-tools", {"run_id": "grpo-qwen3b-tools"})
    session.log({"train/reward": 0.2}, step=1)
    import wandb

    wandb.log({"train/global_step": 2, "train/reward": 0.3})
    session.finish()


def test_config_update_scrubs_tokens(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    fake = _install(monkeypatch, _Wandb())
    start_wandb("run", {"run_id": "run", "hub_token": "should-drop", "note": "keep"})
    fake.run.config.update({"hub_token": "abc", "learning_rate": 1e-5, "api_key": "nope"})
    assert fake.run.config.updates == [{"learning_rate": 1e-5}]
    scrubbed = scrub_config({"adapter": "/vol/a", "WANDB_API_KEY": "test-key", "ok": 1})
    assert "WANDB_API_KEY" not in scrubbed
    assert scrubbed["ok"] == 1


def test_resume_floor_and_checkpoint_step():
    assert resume_floor(resumed=False, summary_step=10, resume_step=50) is None
    assert resume_floor(resumed=True, summary_step=73, resume_step=50) == 73
    assert resume_floor(resumed=True, summary_step=None, resume_step=50) == 50
    assert resume_floor(resumed=True, summary_step=None, resume_step=None) is None
    assert step_from_checkpoint("/vol/runs/grpo/checkpoint-50") == 50
    assert step_from_checkpoint("/vol/runs/grpo/adapter") is None
    assert should_skip_log({"train/global_step": 50, "train/reward": 0.1}, None, 50)
    assert not should_skip_log({"train/reward": 0.1}, None, 50)
    assert not should_skip_log({"eval/strict": 0.4}, None, 50)


def test_memorization_fields_and_eval_summary():
    report = {
        "gap": {"strict": 0.2, "pass": 0.1},
        "answered_without_looking": {"rate": 0.0},
        "known": {"strict": 0.5, "pass": 0.5},
        "unseen": {"strict": 0.3, "pass": 0.4},
        "flags": {
            "gap": "warn",
            "answered_without_looking": "ok",
            "swapped_facts": "ok",
            "level": "warn",
        },
        "auto_stop": False,
    }
    flat = flatten_memorization(report)
    assert flat["memorization/gap/strict"] == 0.2
    assert flat["memorization/flags/level"] == 1
    assert flat["memorization/flags/level_name"] == "warn"
    assert flat["memorization/auto_stop"] is False

    def measure(step):
        return report

    logs = {"reward": 0.4}
    merge_memorization(logs, 10, measure, every=10)
    assert logs["memorization/gap/strict"] == 0.2
    quiet = {"loss": 1.2}
    merge_memorization(quiet, 10, measure, every=10)
    assert "memorization/gap/strict" not in quiet
    early = {"reward": 0.4}
    merge_memorization(early, 9, measure, every=10)
    assert "memorization/gap/strict" not in early

    summary = {
        "models": {"Qwen/Qwen2.5-3B-Instruct+tools": {"mean_reward": 0.5, "pass_rate": 0.4}},
        "leaderboard": [{"name": "Qwen/Qwen2.5-3B-Instruct+tools"}],
        "seeds": [0],
        "memorization": report,
    }
    tasks = [
        {"id": "A", "type": "approval"},
        {"id": "B", "type": "boxed_warning"},
    ]
    trajectories = [
        {"task_id": "A", "reward": {"total": 1.0, "correctness": 1.0}},
        {"task_id": "B", "reward": {"total": 0.0, "correctness": 0.0}},
    ]
    metrics = eval_history_metrics(summary, tasks, trajectories)
    assert metrics["eval/strict"] == 0.5
    assert metrics["eval/pass"] == 0.4
    assert metrics["eval/type/approval/strict"] == 1.0
    assert metrics["eval/type/approval/pass"] == 1.0
    assert metrics["eval/type/boxed_warning/strict"] == 0.0
    assert metrics["eval/type/boxed_warning/pass"] == 0.0
    assert metrics["memorization/flags/level_name"] == "warn"


def test_every_memorization_number_is_logged_and_the_report_is_saved(tmp_path):
    report = {
        "known": {"n": 4, "strict": 0.5, "pass": 0.5, "mean_tool_calls": 3.5},
        "unseen": {"n": 5, "strict": 0.3, "pass": 0.4, "mean_tool_calls": 4.25},
        "gap": {"strict": 0.2, "pass": 0.1},
        "answered_without_looking": {
            "rate": 0.0,
            "correct": 2,
            "without_looking": 0,
            "known": {"correct": 1, "without_looking": 0, "rate": 0.0},
            "unseen": {"correct": 1, "without_looking": 0, "rate": 0.0},
            "rule": "gold value or gold_citation id in a tool output",
        },
        "mean_tool_calls": {"known": 3.5, "unseen": 4.25},
        "swapped_facts": {
            "n": 30,
            "follow": 12,
            "follow_rate": 0.4,
            "original": 10,
            "original_rate": 1 / 3,
            "other": 8,
        },
        "flags": {
            "gap": "ok",
            "answered_without_looking": "ok",
            "swapped_facts": "warn",
            "level": "warn",
        },
        "auto_stop": False,
        "thresholds": {"every_n_steps": 10, "gap_warn": 0.15, "no_look_warn": 0.2, "follow_warn": 0.7},
    }
    flat = flatten_memorization(report)
    assert flat["memorization/known/mean_tool_calls"] == 3.5
    assert flat["memorization/unseen/mean_tool_calls"] == 4.25
    assert flat["memorization/known/n"] == 4
    assert flat["memorization/swapped/follow_rate"] == 0.4
    assert flat["memorization/swapped/follow"] == 12
    assert flat["memorization/swapped/n"] == 30
    assert flat["memorization/swapped/original"] == 10
    assert flat["memorization/swapped/original_rate"] == 1 / 3
    assert flat["memorization/swapped/other"] == 8
    assert flat["memorization/mean_tool_calls/known"] == 3.5
    assert flat["memorization/answered_without_looking/correct"] == 2
    assert flat["memorization/answered_without_looking/known/rate"] == 0.0
    assert flat["memorization/thresholds/follow_warn"] == 0.7
    assert flat["memorization/no_look"] == 0.0
    assert flat["memorization/flags/swapped_facts"] == 1
    assert flat["memorization/flags/swapped_facts_name"] == "warn"
    assert flat["memorization/auto_stop"] is False
    assert "memorization/answered_without_looking/rule" not in flat
    assert "gold_citation" not in json.dumps(flat)
    absent = flatten_memorization(
        {"known": {"strict": None, "mean_tool_calls": None}, "swapped_facts": {"n": 0, "follow_rate": None}}
    )
    assert "memorization/known/strict" not in absent
    assert "memorization/known/mean_tool_calls" not in absent
    assert "memorization/swapped/follow_rate" not in absent
    assert absent["memorization/swapped/n"] == 0.0

    def measure(step):
        return report

    logs = {"reward": 0.4}
    merge_memorization(logs, 10, measure, every=10, report_dir=tmp_path / "runs" / "grpo-qwen3b-tools")
    saved_path = tmp_path / "runs" / "grpo-qwen3b-tools" / "memorization-10.json"
    assert json.loads(saved_path.read_text()) == report
    assert logs["memorization/swapped/follow_rate"] == 0.4
    assert logs["memorization/known/mean_tool_calls"] == 3.5
    early = {"reward": 0.4}
    merge_memorization(early, 9, measure, every=10, report_dir=tmp_path / "runs" / "grpo-qwen3b-tools")
    assert not (tmp_path / "runs" / "grpo-qwen3b-tools" / "memorization-9.json").exists()
    assert "memorization/swapped/follow" not in early
    quiet = {"loss": 1.0}
    merge_memorization(quiet, 10, measure, every=10, report_dir=tmp_path / "runs" / "grpo-qwen3b-tools")
    assert list((tmp_path / "runs" / "grpo-qwen3b-tools").glob("memorization-*.json")) == [saved_path]


def test_chart_series_and_png(tmp_path):
    rows = [
        {"train/global_step": 1, "train/reward": 0.2, "train/rewards/env/mean": 0.1},
        {"train/global_step": 1, "train/reward": 0.4, "train/rewards/env/mean": 0.3},
        {
            "train/global_step": 2,
            "train/reward": 0.5,
            "train/memorization/gap/strict": 0.2,
            "train/memorization/flags/level": 2,
            "train/memorization/flags/gap": 1,
            "train/memorization/flags/answered_without_looking": 0,
            "train/memorization/flags/swapped_facts": 0,
        },
    ]
    series = chart_series(rows)
    assert series["reward"] == [(1, 0.4), (2, 0.5)]
    assert series["strict"] == [(1, 0.3)]
    assert series["gap"] == [(2, 0.2)]
    assert series["flags"]["level"] == [(2, 2.0)]
    assert series["flags"]["gap"] == [(2, 1.0)]
    destination = tmp_path / "chart.png"
    render_chart(series, destination, title="grpo-qwen3b-tools")
    data = destination.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 1000
    empty = chart_series([{"loss": 1.0}])
    render_chart(empty, tmp_path / "empty.png", title="empty")
    assert (tmp_path / "empty.png").read_bytes().startswith(b"\x89PNG")


def test_key_is_not_a_parameter_or_a_literal():
    signature = inspect.signature(start_wandb)
    assert "api_key" not in signature.parameters
    assert "wandb_api_key" not in signature.parameters
    for relative in (
        "modal_train.py",
        "fdagym/train/wandb_log.py",
        "fdagym/train/grpo_tools.py",
        "fdagym/train/sft_launch.py",
        "fdagym/train/sft.py",
        "fdagym/train/eval_tools.py",
        "fdagym/train/launch.py",
        "scripts/wandb_chart.py",
        "fdagym/train/jobs.py",
        "scripts/run_gpu.py",
    ):
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {arg.arg for arg in node.args.args}
                assert "wandb_api_key" not in names
                assert "api_key" not in names
    modal = (ROOT / "modal_train.py").read_text()
    assert 'Secret.from_name("wandb")' in modal
    assert "WANDB_API_KEY=" not in modal
    from fdagym.train.gpu_stack import modal_pip

    assert any(item.startswith("wandb>=") for item in modal_pip())
    assert "_modal_pip()" in modal
    assert "FDAGYM_GIT_SHA" in modal
    assert modal.index('image.env({"FDAGYM_GIT_SHA"') < modal.index(".add_local_dir(")
    assert ".env(" not in modal.split(".add_local_dir(", 1)[1]
    for name in ("def sft(", "def train_tools(", "def eval_tools("):
        assert name in modal
    assert modal.count("secrets=[WANDB_SECRET]") >= 4
    grpo = (ROOT / "fdagym" / "train" / "grpo_tools.py").read_text()
    assert "report_to" in grpo
    assert "attach_grpo_train_logs" in grpo
    chart = (ROOT / "scripts" / "wandb_chart.py").read_text()
    assert "fetch_history" in chart
    assert "WANDB_API_KEY" not in chart or "WANDB_API_KEY is not set" in chart


def test_build_run_config_drops_a_value_equal_to_the_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    config = build_run_config(
        run_id="eval-tools",
        job="eval",
        prompt_version="v4",
        tool_budget=8,
        hard_train_mix=None,
        seeds=[0],
        adapter=None,
        extra={"leaked": "test-key"},
    )
    assert config["leaked"] is None
    assert config["prompt_version"] == "v4"
    assert "test-key" not in json.dumps({k: v for k, v in config.items() if k != "leaked"})


def test_grpo_log_fields_reach_wandb_before_the_callback():
    """Hugging Face logs W&B in default callbacks; mutating in on_log is too late."""
    report = {
        "gap": {"strict": 0.2, "pass": 0.1},
        "answered_without_looking": {"rate": 0.0},
        "flags": {
            "gap": "warn",
            "answered_without_looking": "ok",
            "swapped_facts": "ok",
            "level": "warn",
        },
        "auto_stop": False,
    }

    def measure(step):
        return report

    wandb_seen = {}
    logs = {"reward": 0.4}
    attach_grpo_train_logs(logs, step=10, measure=measure, every=10)
    wandb_seen.update(logs)
    assert wandb_seen["memorization/gap/strict"] == 0.2
    assert wandb_seen["memorization/flags/level_name"] == "warn"

    control = type("Control", (), {"should_training_stop": False})()
    guard = ParseRateGuard(window=2, min_rate=0.9)
    first = {"reward": 0.1, "rewards/parse_rate": 0.1}
    attach_grpo_train_logs(first, step=1, measure=None, every=10, guard=guard, control=control)
    assert "parse_stop" not in first
    assert control.should_training_stop is False
    second = {"reward": 0.1, "rewards/parse_rate": 0.1}
    attach_grpo_train_logs(second, step=2, measure=None, every=10, guard=guard, control=control)
    assert "parse_stop" in second
    assert control.should_training_stop is True

    from fdagym.train.grpo_tools import ToolGRPOTrainer, run_tool_grpo

    source = inspect.getsource(ToolGRPOTrainer.log)
    assert "attach_grpo_train_logs(" in source
    assert source.index("self._metrics") < source.index("attach_grpo_train_logs(")
    assert source.index("attach_grpo_train_logs(") < source.index("super().log")
    assert "def on_log" not in inspect.getsource(run_tool_grpo)
    sft_log = (ROOT / "fdagym" / "train" / "sft_launch.py").read_text()
    assert "return super().log(logs, *args, **kwargs)" in sft_log


@pytest.mark.parametrize(
    "source",
    [
        "fdagym/train/grpo_tools.py",
        "fdagym/train/launch.py",
        "fdagym/train/sft_launch.py",
        "fdagym/train/sft.py",
    ],
)
def test_trainers_set_report_to_from_the_session(source):
    text = (ROOT / source).read_text()
    assert '["report_to"] = session.report_to' in text
    assert "resume" in text
    assert "training-curve" not in text
