"""Fit LoRA distill SFT on nested teacher subsets. Torch stays inside ``fit_subset``."""

from __future__ import annotations

import json
from pathlib import Path

from fdagym.train.distill import DistillConfig, data_hash, question_ids, size_run_id, subset_report, teacher_mix
from fdagym.train.distill_sft import build_distill_rows
from fdagym.train.rollout import QwenToolCodec, RolloutLimits, RolloutTokenizer
from fdagym.train.sft_launch import _blank, _collate, _last_eval, _tool_sft_trainer, adapter_sha256, load_sft_model
from fdagym.train.sft_tools import IGNORE_INDEX
from fdagym.train.wandb_log import build_run_config, git_sha, start_wandb, step_from_checkpoint


def subset_manifest(
    *,
    run_id: str,
    size: str,
    rows,
    config: DistillConfig,
    filter_stats: dict,
    n_tokenized: int,
    n_skipped: int,
    init_adapter: str | None,
    init_sha: str | None,
) -> dict:
    report = subset_report(size, rows)
    return {
        "run_id": run_id,
        "size": size,
        "n": report["n"],
        "n_tokenized": n_tokenized,
        "n_token_skipped": n_skipped,
        "data_hash": report["data_hash"],
        "question_ids": report["question_ids"],
        "trajectory_ids": report["trajectory_ids"],
        "per_template_counts": report["per_template"],
        "per_variant_counts": report["per_variant"],
        "teacher_mix": report["teacher_mix"],
        "hyperparameters": config.hyperparameters(),
        "filter": filter_stats,
        "git_sha": git_sha(),
        "seed": config.seed,
        "prompt_version": config.prompt_version,
        "tool_budget": config.tool_budget,
        "model": config.model,
        "init_adapter": init_adapter,
        "init_adapter_sha256": init_sha,
        "eval_after": config.eval_after,
        "eval_split": config.eval_split if config.eval_after else None,
    }


def write_manifest(directory: Path, manifest: dict) -> Path:
    path = Path(directory) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def trainer_kwargs(config: DistillConfig, output_dir: str) -> dict:
    payload = config.sft_config().trainer_kwargs(output_dir, eval_split=False)
    if config.max_steps:
        payload["max_steps"] = config.max_steps
    return payload


def fit_subset(
    trajectories,
    config: DistillConfig,
    *,
    run_id: str,
    results_dir: Path,
    size: str,
    filter_stats: dict,
    on_save=None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    tokenizer=None,
) -> dict:
    """One LoRA adapter for one nested subset. Requires ``uv sync --extra train`` or ``gpu``."""
    _require_train_stack()
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    init_adapter = _blank(config.init_adapter)
    init_sha = adapter_sha256(init_adapter) if init_adapter else None
    if tokenizer is None:
        hf_tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if hf_tokenizer.pad_token_id is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        tokenizer = RolloutTokenizer(hf_tokenizer)
        pad_id = tokenizer.pad_token_id
        processing = hf_tokenizer
    else:
        pad_id = tokenizer.pad_token_id
        processing = tokenizer
    limits = RolloutLimits(
        temperature=0.0,
        tool_budget=config.tool_budget,
        prompt_version=config.prompt_version,
    )
    codec = QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    built = build_distill_rows(trajectories, tokenizer, codec, limits, max_length=config.max_length)
    if not built["rows"]:
        raise RuntimeError(f"{run_id}: no tokenized rows (skipped {built['skipped']})")
    columns = ("input_ids", "attention_mask", "labels")
    dataset = Dataset.from_list([{key: row[key] for key in columns} for row in built["rows"]])
    sft_config = config.sft_config()
    model = load_sft_model(sft_config, init_adapter)
    out = Path(results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = subset_manifest(
        run_id=run_id,
        size=size,
        rows=trajectories,
        config=config,
        filter_stats=filter_stats,
        n_tokenized=built["n_kept"],
        n_skipped=built["n_skipped"],
        init_adapter=init_adapter,
        init_sha=init_sha,
    )
    write_manifest(out, manifest)

    class _SaveCallback(TrainerCallback):
        def on_save(self, args_, state, control, **kwargs):
            if on_save is not None:
                on_save(state)

    from fdagym.train.checkpoints import latest_checkpoint

    adapter = out / "adapter"
    checkpoint = latest_checkpoint(out)
    session = start_wandb(
        run_id,
        build_run_config(
            run_id=run_id,
            job="distill-sft",
            prompt_version=config.prompt_version,
            tool_budget=config.tool_budget,
            hard_train_mix=None,
            seeds=[config.seed],
            adapter=str(adapter),
            model=config.model,
            extra={
                "size": size,
                "n": len(trajectories),
                "data_hash": manifest["data_hash"],
                "per_template_counts": manifest["per_template_counts"],
                "teacher_mix": manifest["teacher_mix"],
                "init_adapter": init_adapter,
                "init_adapter_sha256": init_sha,
                **config.hyperparameters(),
            },
        ),
        project=wandb_project,
        entity=wandb_entity,
        resume_step=step_from_checkpoint(checkpoint),
    )
    args = trainer_kwargs(config, str(out))
    args["report_to"] = session.report_to
    args["run_name"] = run_id
    trainer = _tool_sft_trainer(Trainer)(
        model=model,
        args=TrainingArguments(**args),
        train_dataset=dataset,
        data_collator=lambda features: _collate(features, pad_id),
        processing_class=processing,
        callbacks=[_SaveCallback()],
    )
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(adapter))
        if hasattr(processing, "save_pretrained"):
            processing.save_pretrained(str(adapter))
        report = {
            **manifest,
            "adapter": str(adapter),
            "output_dir": str(out),
            "skipped_rows": built["skipped"],
            "eval": _last_eval(trainer.state.log_history),
        }
        saved = {key: value for key, value in report.items() if key not in {"adapter", "output_dir"}}
        (out / "sft.json").write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        if on_save is not None:
            on_save(None)
        return report
    finally:
        session.finish()


def dry_tokenize(prepared: dict, tokenizer) -> dict:
    """CPU path: tokenize every nested subset and check the assistant-only mask."""
    from fdagym.train.distill_sft import assert_assistant_only_labels

    config: DistillConfig = prepared["config"]
    limits = RolloutLimits(temperature=0.0, tool_budget=config.tool_budget, prompt_version=config.prompt_version)
    codec = QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    checks = {}
    for name, rows in prepared["subsets"].items():
        built = build_distill_rows(rows, tokenizer, codec, limits, max_length=config.max_length)
        for row, encoded in zip(built["rows"], built["encoded"], strict=True):
            assert_assistant_only_labels(row, encoded)
        supervised = [
            sum(1 for label in row["labels"] if label != IGNORE_INDEX) for row in built["rows"]
        ]
        checks[name] = {
            **prepared["reports"][name],
            "n_tokenized": built["n_kept"],
            "n_token_skipped": built["n_skipped"],
            "supervised_tokens": supervised,
            "skipped": built["skipped"],
        }
    return checks


def _require_train_stack() -> None:
    missing = []
    for name in ("datasets", "peft", "torch", "transformers"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise ImportError(f"Install training extras first: uv sync --extra gpu (missing {joined})")


def default_eval_run_id(train_run_id: str, split: str) -> str:
    return f"{train_run_id}-{split}"


# Re-export helpers tests/jobs import without pulling torch.
__all__ = [
    "IGNORE_INDEX",
    "data_hash",
    "default_eval_run_id",
    "dry_tokenize",
    "fit_subset",
    "question_ids",
    "size_run_id",
    "subset_manifest",
    "teacher_mix",
    "trainer_kwargs",
    "write_manifest",
]
