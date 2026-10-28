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


def test_modal_app_is_l40s_vllm_and_resumable():
    text = (ROOT / "modal_train.py").read_text()
    assert 'gpu="L40S"' in text or "gpu=GPU_DEFAULT" in text
    assert 'GPU_DEFAULT = "L40S"' in text
    assert "A100-80GB" in text
    assert "use_vllm=True" in text
    assert "fdagym-checkpoints" in text
    assert "resume" in text
    assert "volume.commit" in text or "_commit" in text
    assert "eval-base" in text
    assert "sft-plan-a" in text
    assert "init_adapter" in text
    assert "sft-plan-a-qwen3b" in text
    assert "run_id or RUN_ID_3B" in text
    assert "does nothing until" in text.lower() or "Does nothing until" in text
    assert '".env"' in text
    assert '"*.pyc"' in text
    assert "require_snapshot" in text
    assert "_modal_pip()" in text
    from fdagym.train.gpu_stack import gpu_requirements, modal_pip

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
    assert modal_pip()[: len(pins)] == pins
    assert "wandb>=0.19.0" in modal_pip()
    assert "litellm" not in " ".join(modal_pip())
    ignore_block = text.split("IGNORE = [", 1)[1].split("]", 1)[0]
    assert ".env" in ignore_block
    assert "*.pyc" in ignore_block
    assert '"data"' not in ignore_block
    assert "data/snapshots" not in ignore_block


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


def test_stage1_pool_resolves_to_8192():
    snapshot = DiskSnapshot("2026-10-15")
    rows, _mix = build_rows(snapshot, load_split("train"), TrainConfig(stage=1))
    fitted, stats = fit_context(TrainConfig(stage=1), rows)
    needed = stats["measured_prompt_tokens"] + fitted.max_completion_length
    assert rows
    assert 4096 < needed <= 8192, stats
    assert fitted.vllm_max_model_len == 8192
    assert stats["vllm_max_model_len"] == 8192
    assert stats["fits"] is True
    assert fitted.max_pack_chars == 6000
    assert_prompts_fit(rows, fitted)


def test_snapshot_is_on_disk_for_image_upload():
    path = require_snapshot(ROOT)
    assert path.is_file()
    assert not (ROOT / "Dockerfile").exists()
    assert not (ROOT / ".dockerignore").exists()
    setup = (ROOT / "scripts" / "runpod_setup.sh").read_text()
    assert "data/snapshots/2026-10-15/manifest.jsonl" in setup
    runbook = (ROOT / "docs" / "train.md").read_text()
    assert "modal volume get" in runbook
    assert runbook.index("modal volume get") < runbook.index("compare_runs.py")


def test_runpod_setup_pins_the_gpu_extra_and_does_not_train():
    setup = (ROOT / "scripts" / "runpod_setup.sh").read_text()
    assert "uv sync --extra gpu --frozen --no-dev" in setup
    assert "import vllm" in setup
    assert "torch.cuda.is_available()" in setup
    assert "gpu_requirements" in setup
    assert "run_gpu.py" not in setup.split("cat <<'EOF'", 1)[0]
    assert "--preset 7b" not in setup
    assert "A100" in setup
    assert "Does not start" in setup or "No job was started" in setup
