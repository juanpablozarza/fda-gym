"""SFT warm-up knobs. Torch stays out of this module."""

from __future__ import annotations

from dataclasses import dataclass

from fdagym.train.config import QWEN_3B, TrainConfig
from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET
from fdagym.train.sft_tools import MIN_SFT_REWARD, VAL_DRUG_FRACTION


@dataclass(frozen=True)
class SFTConfig:
    """Oracle-trajectory SFT. LoRA matches ``TrainConfig``.

    ``val_drug_fraction`` of the drugs are held out (by drug, seed ``seed``)
    and scored with a teacher-forced tool-call match. ``0`` holds nothing out,
    so the trainer has no eval split. ``task_ids`` replaces the stage pool with
    exactly those rows from train and hard_train.
    """

    model: str = QWEN_3B
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 8192
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 0
    stage: int = 1
    min_reward: float = MIN_SFT_REWARD
    val_drug_fraction: float = VAL_DRUG_FRACTION
    eval_steps: int = 50
    # Same knobs as tool GRPO. New warm-ups use v4 and 8 calls.
    tool_budget: int = DEFAULT_TOOL_BUDGET
    prompt_version: str = DEFAULT_PROMPT_VERSION
    # Off by default: the v3 warm-up was the current train split only.
    # On, append hard_train oracle rows. Holdout and hard eval stay out either way.
    include_hard_train: bool = False
    # Exact pool. None keeps stage_pool (and include_hard_train). The sha is the
    # digest of the sorted ids joined by newlines; blank means do not check.
    task_ids: tuple[str, ...] | None = None
    task_ids_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.task_ids is not None:
            if isinstance(self.task_ids, str) or not isinstance(self.task_ids, (list, tuple)):
                raise TypeError("task_ids must be a list of ids")
            object.__setattr__(self, "task_ids", tuple(self.task_ids))
        sha = self.task_ids_sha256
        if isinstance(sha, str):
            sha = sha.strip().lower() or None
            object.__setattr__(self, "task_ids_sha256", sha)
        if self.save_steps < 1:
            raise ValueError("save_steps must be >= 1")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be > 0")
        if not 0 <= self.val_drug_fraction < 1:
            raise ValueError("val_drug_fraction must be in [0, 1)")
        if self.warmup_ratio < 0:
            raise ValueError("warmup_ratio must be >= 0")
        if self.task_ids_sha256 and self.task_ids is None:
            raise ValueError("task_ids_sha256 requires task_ids")
        if self.task_ids is not None and self.include_hard_train:
            raise ValueError("task_ids replaces include_hard_train; pass only the explicit list")

    def lora_kwargs(self) -> dict:
        return TrainConfig(model=self.model).lora_kwargs()

    def trainer_kwargs(self, output_dir: str, *, eval_split: bool = True) -> dict:
        payload = {
            "output_dir": output_dir,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "logging_steps": self.logging_steps,
            "save_strategy": "steps",
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "report_to": [],
            "seed": self.seed,
            "remove_unused_columns": False,
        }
        if eval_split:
            # Every 50 steps, one forward over the held-out drugs. No generate().
            payload["eval_strategy"] = "steps"
            payload["eval_steps"] = self.eval_steps
            payload["per_device_eval_batch_size"] = self.per_device_train_batch_size
            payload["prediction_loss_only"] = True
        return payload
