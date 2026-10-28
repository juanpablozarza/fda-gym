"""Shared Plan A GRPO entry used by the local script, Modal, and RunPod."""

from __future__ import annotations

import json
from pathlib import Path

from fdagym.train.checkpoints import latest_checkpoint
from fdagym.train.config import TrainConfig, select_grpo_kwargs
from fdagym.train.dataset import build_rows
from fdagym.train.prompt_len import assert_prompts_fit, fit_context
from fdagym.train.score import score_completion


def run_grpo(
    snapshot,
    tasks: list[dict],
    config: TrainConfig,
    *,
    run_id: str,
    results_dir: Path,
    resume: bool = True,
    on_save=None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> dict:
    """Fit LoRA with GRPO. Requires ``uv sync --extra train`` plus vLLM when ``use_vllm``."""
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("Install training extras first: uv sync --extra train") from exc

    rows, mix = build_rows(snapshot, tasks, config)
    if not rows:
        raise RuntimeError("stage pool is empty")
    config, length = fit_context(config, rows)
    assert_prompts_fit(rows, config)
    if not length["fits"]:
        raise RuntimeError(
            f"prompt still exceeds vLLM context after packing: {length}. Lower --distractors or max_pack_chars."
        )
    dataset = Dataset.from_list(rows)
    by_id = {task["id"]: task for task in tasks}
    out = results_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    def fda_reward(completions, task_id, **_kwargs):
        scores = []
        for completion, tid in zip(completions, task_id, strict=True):
            task = by_id[tid]
            reward = score_completion(
                snapshot,
                tasks,
                task,
                completion,
                run_id=run_id,
                dense=config.dense_reward,
            )
            scores.append(float(reward["total"]))
        return scores

    class SaveCallback(TrainerCallback):
        def on_save(self, args_, state, control, **kwargs):
            if on_save is not None:
                on_save(state)

    from fdagym.train.wandb_log import build_run_config, start_wandb, step_from_checkpoint

    ckpt = latest_checkpoint(out) if resume else None
    session = start_wandb(
        run_id,
        build_run_config(
            run_id=run_id,
            job="grpo",
            prompt_version=None,
            tool_budget=None,
            hard_train_mix=None,
            seeds=[0],
            adapter=config.init_adapter,
            model=config.model,
            extra={"stage": config.stage, "mix": mix, "max_steps": config.max_steps},
        ),
        project=wandb_project,
        entity=wandb_entity,
        resume_step=step_from_checkpoint(ckpt),
    )
    kwargs = select_grpo_kwargs(config, str(out), grpo_config_cls=GRPOConfig)
    kwargs["report_to"] = session.report_to
    kwargs["run_name"] = run_id
    model, peft_config = _grpo_model(config, resume_from=ckpt, lora_cls=LoraConfig)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=fda_reward,
        args=GRPOConfig(**kwargs),
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[SaveCallback()],
    )
    try:
        trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None)
        adapter = out / "adapter"
        trainer.save_model(str(adapter))
        (out / "mix.json").write_text(json.dumps({"stage": config.stage, "mix": mix, "n": len(rows)}, indent=2))
        if on_save is not None:
            on_save(None)
        return {
            "run_id": run_id,
            "output_dir": str(out),
            "adapter": str(adapter),
            "n_prompts": len(rows),
            "mix": mix,
            "resumed": str(ckpt) if ckpt else None,
            "model": config.model,
            "prompt_length": length,
            "init_adapter": config.init_adapter,
            "dense_reward": config.dense_reward,
        }
    finally:
        session.finish()


def _grpo_model(config, resume_from, *, lora_cls):
    """Fresh LoRA, or the SFT adapter when ``init_adapter`` is set and this run is not resuming."""
    if config.init_adapter and resume_from is None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, config.init_adapter, is_trainable=True)
        return model, None
    return config.model, lora_cls(**config.lora_kwargs())
