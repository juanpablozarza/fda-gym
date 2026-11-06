from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.openfda import DiskSnapshot
from fdagym.train.checkpoints import latest_checkpoint
from fdagym.train.config import QWEN_3B, QWEN_7B, TrainConfig, for_7b, select_grpo_kwargs
from fdagym.train.dataset import build_rows
from fdagym.train.prompt_len import assert_prompts_fit, fit_context
from fdagym.train.snapshot_fs import require_snapshot

ROOT = Path(__file__).resolve().parents[2]


def test_default_3b_vllm_and_50_step_checkpoints():
    cfg = TrainConfig()
    assert cfg.model == QWEN_3B
    assert cfg.num_generations == 8
    assert cfg.max_steps == 400
    assert cfg.max_prompt_length == 8192
    assert cfg.vllm_max_model_len == 8192
    assert cfg.max_completion_length == 256
    assert cfg.save_steps == 50
    assert cfg.use_vllm is True
    assert cfg.dense_reward is True
    assert cfg.init_adapter is None
    payload = cfg.grpo_kwargs("/tmp/out")
    assert payload["max_prompt_length"] == 8192
    assert payload["save_steps"] == 50
    assert payload["use_vllm"] is True
    assert payload["vllm_mode"] == "colocate"
    assert payload["max_completion_length"] == 256


def test_select_grpo_kwargs_drops_unknown_fields():
    class Fake:
        def __init__(self, output_dir, use_vllm=False, max_steps=1, learning_rate=1e-5, **_ignored):
            pass

        # inspect.signature reads __init__; extra kwargs absorbed so we pass a tight stub instead.

    class Tight:
        def __init__(self, output_dir, use_vllm=False, max_steps=400):
            pass

    selected = select_grpo_kwargs(TrainConfig(), "/tmp/out", grpo_config_cls=Tight)
    assert selected["output_dir"] == "/tmp/out"
    assert selected["use_vllm"] is True
    assert "max_prompt_length" not in selected
    assert "vllm_mode" not in selected
    assert "max_steps" in selected


def test_latest_checkpoint_picks_highest_step(tmp_path):
    run = tmp_path / "grpo-qwen3b"
    (run / "checkpoint-50").mkdir(parents=True)
    (run / "checkpoint-150").mkdir()
    (run / "checkpoint-100").mkdir()
    (run / "adapter").mkdir()
    (run / "checkpoint-nope").mkdir()
    assert latest_checkpoint(run).name == "checkpoint-150"
    assert latest_checkpoint(tmp_path / "missing") is None


def test_7b_preset_enables_gradient_checkpointing():
    cfg = for_7b()
    assert cfg.model == QWEN_7B
    assert cfg.gradient_checkpointing is True
    assert cfg.use_vllm is True
    assert cfg.num_generations == 8
    assert cfg.max_steps == 400
    help_text = (ROOT / "scripts" / "grpo_singleturn.py").read_text()
    assert "--preset" in help_text
    assert "7b" in help_text


def test_gpu_extra_pins_the_local_cuda_stack():
    from fdagym.train.gpu_stack import gpu_requirements

    pins = gpu_requirements()
    assert pins == [
        "torch==2.6.0",
        "transformers==4.55.0",
        "accelerate==1.4.0",
        "trl==0.21.0",
        "peft==0.15.2",
        "datasets==3.5.1",
        "vllm==0.8.5",
    ]
    extra = (ROOT / "pyproject.toml").read_text()
    assert "[project.optional-dependencies]" in extra
    cfg_text = (ROOT / "fdagym" / "train" / "config.py").read_text()
    assert "use_vllm: bool = True" in cfg_text
    assert "require_snapshot" in (ROOT / "fdagym" / "train" / "snapshot_fs.py").read_text()


def test_fit_context_ignores_the_four_document_maximum():
    """A short pool must not grow the window from 4 × 6000 characters."""
    untouched, before = fit_context(TrainConfig(max_pack_chars=6000, n_distractors=2))
    assert untouched.vllm_max_model_len == 8192
    assert untouched.max_pack_chars == 6000
    assert before["fits"] is True
    assert "worst_case_prompt_tokens" not in before

    rows = [{"task_id": "FG-short", "prompt": [{"role": "user", "content": "x" * 90}]}]
    fitted, stats = fit_context(TrainConfig(max_pack_chars=6000, n_distractors=2), rows)
    assert stats["measured_prompt_tokens"] + fitted.max_completion_length <= 4096
    assert fitted.vllm_max_model_len == 4096
    assert fitted.max_pack_chars == 6000
    assert stats["fits"] is True


def test_over_cap_prompt_stays_at_8192_and_launch_rejects_it():
    rows = [{"task_id": "FG-long", "prompt": [{"role": "user", "content": "x" * 30000}]}]
    fitted, stats = fit_context(TrainConfig(), rows)
    assert fitted.vllm_max_model_len == 8192
    assert fitted.max_pack_chars == 6000
    assert stats["fits"] is False
    with pytest.raises(RuntimeError, match="FG-long"):
        assert_prompts_fit(rows, fitted)


def test_stage1_pool_fits_4096_after_approval_summaries():
    snapshot = DiskSnapshot("2026-10-15")
    rows, _mix = build_rows(snapshot, load_split("train"), TrainConfig(stage=1))
    fitted, stats = fit_context(TrainConfig(stage=1), rows)
    needed = stats["measured_prompt_tokens"] + fitted.max_completion_length
    assert rows
    # Generic-name approval pages are summaries, so the longest stage-1 prompt
    # no longer needs the 8192 window. fit_context still rounds up to 4096.
    assert needed <= 4096, stats
    assert fitted.vllm_max_model_len == 4096
    assert stats["vllm_max_model_len"] == 4096
    assert stats["fits"] is True
    assert fitted.max_pack_chars == 6000
    assert_prompts_fit(rows, fitted)


def test_snapshot_is_on_disk_for_local_gpu_jobs():
    path = require_snapshot(ROOT)
    assert path.is_file()
    runbook = (ROOT / "docs" / "train.md").read_text()
    assert "results/" in runbook
    assert runbook.index("results/") < runbook.index("compare_runs.py")
    assert "uv sync --extra gpu" in runbook
