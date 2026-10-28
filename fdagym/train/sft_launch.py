"""Fit the SFT warm-up. Torch, Transformers, and PEFT are imported inside ``run_sft``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fdagym.train.rollout import QwenToolCodec, RolloutLimits, RolloutTokenizer
from fdagym.train.sft_config import SFTConfig
from fdagym.train.sft_select import sft_sources, task_ids_digest
from fdagym.train.sft_tools import IGNORE_INDEX, build_sft_rows, split_sft_by_drug, teacher_forced_tool_match


def run_sft(
    snapshot,
    tasks: list[dict],
    config: SFTConfig | None = None,
    *,
    run_id: str,
    results_dir: Path,
    on_save=None,
    hard_tasks: list[dict] | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    init_adapter: str | None = None,
) -> dict:
    """Supervised loss on oracle tool calls. Requires ``uv sync --extra train``.

    ``init_adapter`` continues that LoRA (trainable) instead of attaching a fresh one.
    """
    _require_train_stack()
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    config = config or SFTConfig()
    selected, mix = sft_sources(tasks, config, hard_tasks)
    if not selected:
        raise RuntimeError("stage pool is empty")
    init_adapter = _blank(init_adapter)
    init_sha = adapter_sha256(init_adapter) if init_adapter else None
    # build_sft_rows looks tasks up by id. hard_train rows are not in the train file.
    pool = [*tasks, *(hard_tasks or [])] if config.include_hard_train or config.task_ids is not None else tasks
    train_tasks, val_tasks = split_sft_by_drug(selected, fraction=config.val_drug_fraction, seed=config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rollout_tokenizer = RolloutTokenizer(tokenizer)
    limits = RolloutLimits(
        temperature=0.0,
        tool_budget=config.tool_budget,
        prompt_version=config.prompt_version,
    )
    codec = QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    built = build_sft_rows(
        snapshot,
        pool,
        [task["id"] for task in train_tasks],
        rollout_tokenizer,
        codec,
        limits,
        max_length=config.max_length,
        min_reward=config.min_reward,
        run_id=run_id,
    )
    # fraction 0 holds out no drugs. Skip the eval replay instead of scoring an empty split.
    if val_tasks:
        val_built = build_sft_rows(
            snapshot,
            pool,
            [task["id"] for task in val_tasks],
            rollout_tokenizer,
            codec,
            limits,
            max_length=config.max_length,
            min_reward=config.min_reward,
            run_id=f"{run_id}-val",
        )
    else:
        val_built = {"rows": [], "n_kept": 0, "n_skipped": 0, "skipped": []}
    if not built["rows"]:
        raise RuntimeError(f"SFT pool is empty after skipping rows below {config.min_reward} or over max_length")
    columns = ("input_ids", "attention_mask", "labels")
    dataset = Dataset.from_list([{key: row[key] for key in columns} for row in built["rows"]])
    eval_dataset = (
        Dataset.from_list([{key: row[key] for key in columns} for row in val_built["rows"]])
        if val_built["rows"]
        else None
    )
    model = load_sft_model(config, init_adapter)
    out = Path(results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    class _SaveCallback(TrainerCallback):
        def on_save(self, args_, state, control, **kwargs):
            if on_save is not None:
                on_save(state)

    from fdagym.train.checkpoints import latest_checkpoint
    from fdagym.train.wandb_log import build_run_config, start_wandb, step_from_checkpoint

    adapter = out / "adapter"
    checkpoint = latest_checkpoint(out)
    session = start_wandb(
        run_id,
        build_run_config(
            run_id=run_id,
            job="sft",
            prompt_version=config.prompt_version,
            tool_budget=config.tool_budget,
            hard_train_mix=None,
            seeds=[config.seed],
            adapter=str(adapter),
            model=config.model,
            extra={
                "include_hard_train": config.include_hard_train,
                "mix": mix,
                "init_adapter": init_adapter,
                "init_adapter_sha256": init_sha,
                "task_ids_sha256": task_ids_digest(config.task_ids) if config.task_ids is not None else None,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
                "save_steps": config.save_steps,
                "val_drug_fraction": config.val_drug_fraction,
                "warmup_ratio": config.warmup_ratio,
            },
        ),
        project=wandb_project,
        entity=wandb_entity,
        resume_step=step_from_checkpoint(checkpoint),
    )
    trainer_args = config.trainer_kwargs(str(out), eval_split=eval_dataset is not None)
    trainer_args["report_to"] = session.report_to
    trainer_args["run_name"] = run_id
    trainer = _tool_sft_trainer(Trainer)(
        model=model,
        args=TrainingArguments(**trainer_args),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: _collate(features, tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[_SaveCallback()],
    )
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        if eval_dataset is not None and trainer.state.global_step % config.eval_steps:
            # ~139 steps with eval every 50: score the held-out drugs once more on the final weights.
            trainer.evaluate()
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        report = {
            "run_id": run_id,
            "adapter": str(adapter),
            "n_kept": built["n_kept"],
            "n_skipped": built["n_skipped"],
            "skipped": built["skipped"],
            "n_val": val_built["n_kept"],
            "n_val_skipped": val_built["n_skipped"],
            "val_drugs": sorted({task["entity"]["generic_name"] for task in val_tasks}),
            "min_reward": config.min_reward,
            "epochs": config.num_train_epochs,
            "prompt_version": config.prompt_version,
            "tool_budget": config.tool_budget,
            "include_hard_train": config.include_hard_train,
            "mix": mix,
            "model": config.model,
            "init_adapter": init_adapter,
            "init_adapter_sha256": init_sha,
            "task_ids": list(config.task_ids) if config.task_ids is not None else None,
            "task_ids_sha256": task_ids_digest(config.task_ids) if config.task_ids is not None else None,
            "learning_rate": config.learning_rate,
            "save_steps": config.save_steps,
            "val_drug_fraction": config.val_drug_fraction,
            "warmup_ratio": config.warmup_ratio,
            "eval": _last_eval(trainer.state.log_history),
        }
        saved = {key: value for key, value in report.items() if key != "adapter"}
        (out / "sft.json").write_text(json.dumps(saved, indent=2))
        if on_save is not None:
            on_save(None)
        report["output_dir"] = str(out)
        return report
    finally:
        session.finish()


def _require_train_stack() -> None:
    """Fail before the oracle replay when the train extra is not installed."""
    missing = []
    for name in ("datasets", "peft", "torch", "transformers"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise ImportError(f"Install training extras first: uv sync --extra train (missing {joined})")


def load_sft_model(config: SFTConfig, init_adapter: str | None):
    """Fresh LoRA, or ``init_adapter`` with its existing weights left trainable.

    ``get_peft_model`` would attach a second adapter on top of the warm start.
    ``is_trainable=True`` keeps the loaded LoRA in the optimizer.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() and config.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if init_adapter:
        from peft import PeftModel

        return PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(**config.lora_kwargs()))


def is_hub_adapter(ref: str | Path | None) -> bool:
    """True for a Hugging Face repo id (``org/name``) that is not a local path.

    Local directories that exist win. Absolute, ``.``-relative, and ``~`` paths
    that are missing stay local errors, not Hub lookups.
    """
    if ref is None:
        return False
    text = str(ref).strip()
    if not text:
        return False
    path = Path(text).expanduser()
    if path.exists():
        return False
    if path.is_absolute() or text.startswith((".", "~")):
        return False
    parts = text.split("/")
    return len(parts) == 2 and all(parts) and not any(part in {".", ".."} for part in parts)


def adapter_sha256(path: str | Path) -> str:
    """SHA-256 of ``adapter_model.safetensors``, the file the volume check hashes.

    A Hub repo id hashes ``hub:<org/name>`` so a probe can record the requested
    adapter without downloading weights first. ``PeftModel.from_pretrained``
    still needs ``HF_TOKEN`` for a private repo.
    """
    text = str(path).strip()
    weights = Path(text) / "adapter_model.safetensors"
    if weights.is_file():
        digest = hashlib.sha256()
        with weights.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if is_hub_adapter(text):
        return hashlib.sha256(f"hub:{text}".encode()).hexdigest()
    raise FileNotFoundError(f"init adapter weights not found: {weights}")


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _tool_sft_trainer(trainer_cls):
    """Trainer that logs teacher-forced tool-call match from the eval forward pass."""

    class ToolSFTTrainer(trainer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tf_first = 0
            self._tf_seq = 0
            self._tf_n = 0

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            evaluating = not model.training
            kwargs = {"return_outputs": return_outputs or evaluating}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            result = super().compute_loss(model, inputs, **kwargs)
            if not evaluating:
                return result
            loss, outputs = result
            labels = inputs.get("labels")
            logits = getattr(outputs, "logits", None)
            if labels is not None and logits is not None:
                self._note_match(logits.detach(), labels)
            return (loss, outputs) if return_outputs else loss

        def _note_match(self, logits, labels) -> None:
            # preds[i] predicts labels[i + 1]. Pass the full label row, not a shifted copy.
            predicted = logits[:, :-1, :].argmax(dim=-1)
            for row_pred, row_target in zip(predicted, labels, strict=True):
                match = teacher_forced_tool_match(row_pred.tolist(), row_target.tolist())
                self._tf_n += 1
                self._tf_first += int(match["first_tool_exact"])
                self._tf_seq += int(match["tool_seq_exact"])

        def log(self, logs, *args, **kwargs):
            if isinstance(logs, dict) and "eval_loss" in logs and self._tf_n:
                logs = dict(logs)
                logs["eval_first_tool_exact"] = self._tf_first / self._tf_n
                logs["eval_tool_seq_exact"] = self._tf_seq / self._tf_n
                self._tf_first = 0
                self._tf_seq = 0
                self._tf_n = 0
            return super().log(logs, *args, **kwargs)

    return ToolSFTTrainer


def _last_eval(history: list[dict]) -> dict | None:
    for row in reversed(history):
        if "eval_loss" not in row:
            continue
        keys = ("eval_loss", "eval_first_tool_exact", "eval_tool_seq_exact", "epoch", "step")
        return {key: row[key] for key in keys if key in row}
    return None


def _collate(features: list[dict], pad_id: int):
    import torch

    width = max(len(row["input_ids"]) for row in features)
    input_ids = []
    attention = []
    labels = []
    for row in features:
        gap = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * gap)
        attention.append(row["attention_mask"] + [0] * gap)
        labels.append(row["labels"] + [IGNORE_INDEX] * gap)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
