import importlib.util
from pathlib import Path

import pytest

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
