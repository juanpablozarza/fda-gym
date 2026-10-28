import json
from pathlib import Path

import pytest

from fdagym.train.pipeline import Pipeline, Stage, find_mean, strict_mean, tool_artifact_paths

ROOT = Path(__file__).resolve().parents[2]


def _adapter(p):
    p.mkdir(parents=True)
    (p / "adapter_model.safetensors").write_bytes(b"x")


def test_strict_mean_ignores_dense():
    s = {"models": {"m": {"mean_reward": 0.5}}, "dense_reward": {"mean_reward": 0.9}}
    assert find_mean(s) == [0.5]
    assert strict_mean({}) == 0.0


def test_runs_in_order_and_gates(tmp_path):
    calls = []
    stages = [
        Stage("sft", lambda s: calls.append("sft")),
        Stage("eval", lambda s: {"models": {"m": {"mean_reward": 0.01}}},
              gate=lambda s: strict_mean(s["results"]["eval"]) >= 0.0485),
        Stage("grpo", lambda s: calls.append("grpo")),
    ]
    st = Pipeline(tmp_path, stages).run()
    assert st["stage"] == "gate-failed" and calls == ["sft"]


def test_skips_finished_adapter_and_refuses_restart(tmp_path):
    ad = tmp_path / "sft" / "adapter"
    _adapter(ad)
    calls = []
    p = Pipeline(tmp_path / "pipe", [Stage("sft", lambda s: calls.append(1), output=ad)])
    assert p.run()["stage"] == "done"
    assert calls == []  # never overwrite a finished adapter
    with pytest.raises(RuntimeError):
        p.run()
    assert p.run(force=True)["stage"] == "done"


def test_resume_from_sft_skips_finished_evals_and_keeps_scores(tmp_path):
    """Base 0.0086 and SFT eval 0.5552 stay on disk; GRPO is the first stage that runs."""
    runs = tmp_path / "runs"
    paths = tool_artifact_paths(
        runs,
        baseline_id="baseline-qwen3b-tools",
        sft_id="sft-qwen3b-tools",
        grpo_id="grpo-qwen3b-tools",
        grpo_eval_id="grpo-qwen3b-tools-eval",
    )
    base = {
        "run_id": "baseline-qwen3b-tools",
        "models": {"Qwen/Qwen2.5-3B-Instruct": {"n": 324, "mean_reward": 0.0086}},
        "dense_reward": {"mean_reward": 0.0086},
    }
    start = {
        "run_id": "sft-qwen3b-tools-tools-eval",
        "models": {"Qwen/Qwen2.5-3B-Instruct+tools": {"n": 324, "mean_reward": 0.5552}},
        "dense_reward": {"mean_reward": 0.5653},
    }
    paths["eval_base"].parent.mkdir(parents=True)
    paths["eval_base"].write_text(json.dumps(base))
    paths["eval_start"].parent.mkdir(parents=True)
    paths["eval_start"].write_text(json.dumps(start))
    _adapter(paths["sft_adapter"])
    calls = []

    def gate(status):
        return strict_mean(status["results"].get("eval-tools-start")) >= strict_mean(
            status["results"].get("eval-tools-base")
        ) + 0.03

    stages = [
        Stage("eval-tools-base", lambda s: calls.append("base"), output=paths["eval_base"]),
        Stage("sft-tools", lambda s: calls.append("sft"), output=paths["sft_adapter"]),
        Stage("eval-tools-start", lambda s: calls.append("start"), output=paths["eval_start"], gate=gate),
        Stage("train-tools", lambda s: calls.append("grpo") or {"ok": True}, output=paths["grpo_adapter"]),
        Stage("eval-tools", lambda s: calls.append("eval"), output=paths["eval_grpo"]),
    ]
    status = Pipeline(runs / "pipeline-tools", stages).run()
    assert calls == ["grpo", "eval"]
    assert status["stage"] == "done"
    assert strict_mean(status["results"]["eval-tools-base"]) == 0.0086
    assert strict_mean(status["results"]["eval-tools-start"]) == 0.5552
    assert json.loads(paths["eval_base"].read_text())["models"]["Qwen/Qwen2.5-3B-Instruct"]["mean_reward"] == 0.0086
    assert (
        json.loads(paths["eval_start"].read_text())["models"]["Qwen/Qwen2.5-3B-Instruct+tools"]["mean_reward"]
        == 0.5552
    )
    text = (ROOT / "modal_train.py").read_text()
    assert "tool_artifact_paths" in text
    assert 'output=paths["eval_base"]' in text
    assert 'output=paths["eval_start"]' in text
    assert 'output=paths["eval_grpo"]' in text


def test_tool_artifact_paths_match_the_sft_gate_run_id(tmp_path):
    paths = tool_artifact_paths(
        tmp_path,
        baseline_id="baseline-qwen3b-tools",
        sft_id="sft-qwen3b-tools",
        grpo_id="grpo-qwen3b-tools",
        grpo_eval_id="grpo-qwen3b-tools-eval",
    )
    assert paths["eval_base"] == tmp_path / "baseline-qwen3b-tools" / "summary.json"
    assert paths["sft_adapter"] == tmp_path / "sft-qwen3b-tools" / "adapter"
    assert paths["eval_start"] == tmp_path / "sft-qwen3b-tools-tools-eval" / "summary.json"
    assert paths["grpo_adapter"] == tmp_path / "grpo-qwen3b-tools" / "adapter"
    same = tool_artifact_paths(
        tmp_path,
        baseline_id="baseline-qwen3b-tools",
        sft_id="sft-qwen3b-tools",
        grpo_id="grpo-qwen3b-tools",
        grpo_eval_id="grpo-qwen3b-tools-eval",
        init_adapter=str(paths["sft_adapter"]),
    )
    assert same["eval_start"] == paths["eval_start"]
    assert same["sft_adapter"] == paths["sft_adapter"]


def test_failure_recorded(tmp_path):
    def boom(_):
        raise ValueError("x")

    p = Pipeline(tmp_path, [Stage("a", boom)])
    with pytest.raises(ValueError):
        p.run()
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "failed-a"
    with pytest.raises(RuntimeError):
        p.run()
