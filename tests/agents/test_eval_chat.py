import importlib.util
from pathlib import Path

import pytest

from fdagym.env import TOOL_BUDGET
from tests.env.support import EXAMPLE_TASK

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "eval_chat.py"
    spec = importlib.util.spec_from_file_location("eval_chat_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_available_splits_lists_committed_jsonl_and_omits_missing_names():
    module = _module()
    present = set(module.available_splits(ROOT / "data" / "tasks"))
    assert {"public", "hard_train", "hard_train_holdout", "train", "test"} <= present
    assert "hard" not in present
    assert "sponsor_lookup" not in present


def test_eval_chat_rejects_unknown_split():
    module = _module()
    with pytest.raises(SystemExit):
        module._parse(["--model", "x", "--split", "hard", "--run-id", "r"])


def test_eval_chat_refuses_to_overwrite_without_flag(tmp_path, monkeypatch, capsys):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_split", lambda split: [{"id": "FG-0001", "type": "approval"}])
    monkeypatch.setattr(module, "available_splits", lambda tasks_dir=None: ["public"])
    (tmp_path / "results" / "old-run").mkdir(parents=True)
    code = module.main(["--model", "x", "--split", "public", "--run-id", "old-run"])
    assert code == 2
    assert "overwrite" in capsys.readouterr().err


def test_snapshot_version_prefers_the_task_and_falls_back_when_missing():
    module = _module()
    assert module.snapshot_version_for({"snapshot_version": "2026-11-05"}, "2026-10-15") == "2026-11-05"
    assert module.snapshot_version_for({}, "2026-10-15") == "2026-10-15"
    assert module.snapshot_version_for({"snapshot_version": ""}, "2026-10-15") == "2026-10-15"


def test_eval_chat_cli_snapshot_flag_is_only_a_fallback():
    module = _module()
    args = module._parse(["--model", "x", "--split", "public", "--run-id", "t"])
    assert args.snapshot_version == "2026-10-15"


def test_eval_chat_builds_env_from_task_snapshot_and_shared_budget(monkeypatch, tmp_path):
    module = _module()
    task = {**EXAMPLE_TASK, "id": "FG-0001", "snapshot_version": "2026-11-05", "type": "boxed_warning"}
    constructed: dict = {}

    class FakeSnapshot:
        def __init__(self, version):
            constructed["version"] = version
            self.version = version

    class FakeEnv:
        def __init__(self, snapshot, tasks, **kwargs):
            constructed["tool_budget"] = kwargs.get("tool_budget")
            constructed["snapshot"] = snapshot.version
            self.tool_budget = kwargs.get("tool_budget")

    def fake_episode(env, task_id, complete, **kwargs):
        return {"id": "x", "reward": {"total": 1.0, "correctness": 1.0, "notes": []}}

    monkeypatch.setattr(module, "DiskSnapshot", FakeSnapshot)
    monkeypatch.setattr(module, "FDAEnv", FakeEnv)
    monkeypatch.setattr(module, "load_split", lambda split: [task])
    monkeypatch.setattr(module, "run_chat_episode", fake_episode)
    monkeypatch.setattr(module, "append_trajectory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "available_splits", lambda tasks_dir=None: ["public"])

    assert module.main(["--model", "x", "--split", "public", "--run-id", "t", "--snapshot-version", "2026-10-15"]) == 0
    assert constructed["version"] == "2026-11-05"
    assert constructed["snapshot"] == "2026-11-05"
    assert constructed["tool_budget"] == TOOL_BUDGET


def test_eval_chat_falls_back_when_task_omits_snapshot_version(monkeypatch, tmp_path):
    module = _module()
    task = {**EXAMPLE_TASK, "id": "FG-0001", "type": "boxed_warning"}
    task.pop("snapshot_version", None)
    constructed: dict = {}

    class FakeSnapshot:
        def __init__(self, version):
            constructed["version"] = version
            self.version = version

    class FakeEnv:
        def __init__(self, snapshot, tasks, **kwargs):
            constructed["tool_budget"] = kwargs.get("tool_budget")
            self.tool_budget = kwargs.get("tool_budget")

    def fake_episode(env, task_id, complete, **kwargs):
        return {"id": "x", "reward": {"total": 0.1, "correctness": 0.0, "notes": []}}

    monkeypatch.setattr(module, "DiskSnapshot", FakeSnapshot)
    monkeypatch.setattr(module, "FDAEnv", FakeEnv)
    monkeypatch.setattr(module, "load_split", lambda split: [task])
    monkeypatch.setattr(module, "run_chat_episode", fake_episode)
    monkeypatch.setattr(module, "append_trajectory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "available_splits", lambda tasks_dir=None: ["public"])

    assert module.main(["--model", "x", "--split", "public", "--run-id", "t"]) == 0
    assert constructed["version"] == "2026-10-15"
    assert constructed["tool_budget"] == TOOL_BUDGET
