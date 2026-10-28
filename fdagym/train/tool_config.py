"""GPU shape for tool-call GRPO. Plan A ``TrainConfig()`` defaults stay put."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from fdagym.memorization import MemorizationThresholds
from fdagym.train.config import TrainConfig
from fdagym.train.curriculum import GRPO_POOL_TYPES, TOOL_MIX_ROWS, TOOL_TYPE_MIX, parse_pool_types
from fdagym.train.group_sample import optimizer_batch, resolve_hard_signal
from fdagym.train.prompt_filter import PromptFilter, hard_prompt_filter
from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET, RolloutLimits

TOOL_PARSE_BONUS = 0.05
# Distill-adapter pass@8 at 0.8 was 69% unparseable (unclosed <tool_call>).
# Greedy hard eval of the same adapter was 10% format errors. 0.65 is the
# next GRPO / probe default. Override with --temperature; probe and train
# share the flag so they stay identical.
TOOL_TEMPERATURE = 0.65
PARSE_STOP_STEPS = 20
PARSE_STOP_MIN = 0.9


def with_prompts_per_step(config: TrainConfig, prompts_per_step: int) -> TrainConfig:
    """Set the micro-batch to one group and accumulate ``prompts_per_step`` groups.

    ``grpo_kwargs`` still sets ``generation_batch_size`` to ``num_generations``,
    so TRL rolls out one group at a time. See ``optimizer_batch``.
    """
    batch = optimizer_batch(config.num_generations, prompts_per_step)
    return replace(
        config,
        num_generations=batch["num_generations"],
        per_device_train_batch_size=batch["per_device_train_batch_size"],
        gradient_accumulation_steps=batch["gradient_accumulation_steps"],
    )


def tool_train_config(**overrides) -> TrainConfig:
    """8 generations at TOOL_TEMPERATURE, 150 steps, gradient checkpointing, engine context 8192 on one A100-80GB.

    TRL 0.21 sizes the colocated engine as ``max_prompt_length + max_completion_length``.
    It does not read ``vllm_max_model_len`` for that. The two lengths sum to 8192.
    """
    payload = {
        "max_prompt_length": 2048,
        "max_completion_length": 6144,
        "num_generations": 8,
        "max_steps": 150,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "vllm_gpu_memory_utilization": 0.30,
        "vllm_max_model_len": 8192,
        "use_vllm": True,
        "dense_reward": True,
        "temperature": TOOL_TEMPERATURE,
        "extra": {"log_completions": False},
    }
    extra = dict(payload["extra"])
    if "extra" in overrides:
        extra.update(overrides.pop("extra"))
    payload.update(overrides)
    payload["extra"] = extra
    return TrainConfig(**payload)


@dataclass(frozen=True)
class ToolTrainConfig:
    train: TrainConfig = field(default_factory=tool_train_config)
    tool_parse_bonus: float = TOOL_PARSE_BONUS
    # Explicit. New tool-mode runs use v4 and 8 calls. v3 and 6 stay selectable.
    tool_budget: int = DEFAULT_TOOL_BUDGET
    prompt_version: str = DEFAULT_PROMPT_VERSION
    limits: RolloutLimits | None = None
    # GRPO prompt pool by task type (JP, 2026-10-19). None falls back to stage_pool.
    type_mix: dict | None = field(default_factory=lambda: dict(TOOL_TYPE_MIX))
    mix_rows: int = TOOL_MIX_ROWS
    mix_seed: int = 0
    # None keeps the current-train type mix. A new hard RL run passes
    # {"hard": 0.60, "current": 0.40} (see HARD_TRAIN_MIX). The Modal flag
    # defaults to 0, which stays None, so an unflagged train-tools run does not mix.
    hard_train_mix: dict | None = None
    # Measurement only. The callback reads every_n_steps. It does not change the pool.
    memorization: MemorizationThresholds = field(default_factory=MemorizationThresholds)
    # Stop rule: mean parse rate over the first parse_stop_steps steps below parse_stop_min.
    parse_stop_steps: int = PARSE_STOP_STEPS
    parse_stop_min: float = PARSE_STOP_MIN
    # Questions accumulated into one optimizer step. 1 is the unflagged tool run
    # (the shape that fit on the A100). A hard-mix run uses 4.
    prompts_per_step: int = 1
    # DAPO-style resample of tied groups. Off unless a hard-mix run turns it on.
    dynamic_sampling: bool = False
    max_resample_attempts: int = 3
    # Pre-run drop of saturated current prompts. Off unless a hard-mix run turns it on.
    prompt_filter: PromptFilter = field(default_factory=PromptFilter)
    # Current-train types in the GRPO prompt pool. Hard-train compare / sponsor_hop
    # are selected by mix_hard_train, not this list. Default drops composite.
    pool_types: tuple[str, ...] = field(default_factory=lambda: GRPO_POOL_TYPES)

    def __post_init__(self) -> None:
        if self.tool_parse_bonus < 0:
            raise ValueError("tool_parse_bonus must be >= 0")
        if self.limits is None:
            object.__setattr__(
                self,
                "limits",
                RolloutLimits(
                    temperature=self.train.temperature,
                    tool_budget=self.tool_budget,
                    prompt_version=self.prompt_version,
                ),
            )
        if self.limits.tool_budget != self.tool_budget:
            raise ValueError(f"rollout tool_budget {self.limits.tool_budget} != config tool_budget {self.tool_budget}")
        if self.limits.prompt_version != self.prompt_version:
            raise ValueError(
                f"rollout prompt_version {self.limits.prompt_version!r} != config prompt_version {self.prompt_version!r}"
            )
        if self.limits.temperature != self.train.temperature:
            raise ValueError(
                f"rollout temperature {self.limits.temperature} != GRPO temperature {self.train.temperature}"
            )
        context = self.train.max_prompt_length + self.train.max_completion_length
        if context != self.limits.max_context_tokens:
            raise ValueError(f"prompt + completion is {context}, rollout context is {self.limits.max_context_tokens}")
        if self.prompts_per_step < 1:
            raise ValueError("prompts_per_step must be >= 1")
        if self.train.gradient_accumulation_steps != self.prompts_per_step:
            raise ValueError(
                f"gradient_accumulation_steps is {self.train.gradient_accumulation_steps}, "
                f"prompts_per_step is {self.prompts_per_step}. "
                "One micro-batch is one question; accumulation is how questions share an optimizer step."
            )
        if self.train.per_device_train_batch_size != self.train.num_generations:
            raise ValueError(
                f"per_device_train_batch_size {self.train.per_device_train_batch_size} must equal "
                f"num_generations {self.train.num_generations} so the log-prob forward stays one group wide."
            )
        if self.max_resample_attempts < 0:
            raise ValueError("max_resample_attempts must be >= 0")


def configure_tool_run(
    *,
    hard: bool,
    max_steps: int = 0,
    prompts_per_step: int | None = None,
    dynamic_sampling: bool | None = None,
    difficulty_filter: bool | None = None,
    max_resample_attempts: int | None = None,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    hard_train_mix: dict | None = None,
    save_steps: int = 50,
    temperature: float | None = None,
    pool_types: str | tuple[str, ...] | list[str] | None = None,
) -> ToolTrainConfig:
    """Build the tool GRPO config. A hard mix turns the signal knobs on unless overridden.

    ``save_steps`` 50 is the unflagged checkpoint interval. A warm-up GRPO run passes 10.
    """
    if save_steps < 1:
        raise ValueError("save_steps must be >= 1")
    signal = resolve_hard_signal(
        hard,
        prompts_per_step=prompts_per_step,
        dynamic_sampling=dynamic_sampling,
        difficulty_filter=difficulty_filter,
        max_resample_attempts=max_resample_attempts,
    )
    train_kwargs: dict = {"save_steps": save_steps}
    if max_steps:
        train_kwargs["max_steps"] = max_steps
    if temperature is not None:
        train_kwargs["temperature"] = float(temperature)
    train = tool_train_config(**train_kwargs)
    train = with_prompts_per_step(train, signal["prompts_per_step"])
    prompt_filter = hard_prompt_filter() if signal["difficulty_filter"] else PromptFilter(enabled=False)
    return ToolTrainConfig(
        train=train,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        hard_train_mix=hard_train_mix,
        prompts_per_step=signal["prompts_per_step"],
        dynamic_sampling=signal["dynamic_sampling"],
        max_resample_attempts=signal["max_resample_attempts"],
        prompt_filter=prompt_filter,
        pool_types=parse_pool_types(pool_types),
    )
