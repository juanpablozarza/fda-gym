"""LoRA / GRPO knobs. Torch stays out of this module so tests import it offline."""

from dataclasses import dataclass, field
from inspect import signature

from fdagym.train.token_logps import DEFAULT_LOGPROB_CHUNK_SIZE

QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class TrainConfig:
    model: str = QWEN_3B
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 1e-5
    num_generations: int = 8
    max_completion_length: int = 256
    max_prompt_length: int | None = 8192
    max_steps: int = 400
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    beta: float = 0.04
    max_pack_chars: int = 6000
    n_distractors: int = 2
    stage: int = 1
    skip_types: tuple[str, ...] = ("label_open",)
    snapshot_version: str = "2026-10-15"
    lora_bias: str = "none"
    task_type: str = "CAUSAL_LM"
    temperature: float = 0.7
    save_steps: int = 50
    save_total_limit: int = 8
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_model_len: int = 8192
    gradient_checkpointing: bool = False
    bf16: bool = True
    dense_reward: bool = True
    init_adapter: str | None = None
    # Token rows of the lm_head scored at once during GRPO log-probs. Not a
    # GRPOConfig field. 256 keeps the logit tensor near 156 MiB instead of the
    # ~17 GiB full-vocab window that OOMed the 8-generation A100 step.
    logprob_chunk_size: int = DEFAULT_LOGPROB_CHUNK_SIZE
    extra: dict = field(default_factory=dict)

    def lora_kwargs(self) -> dict:
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.lora_target_modules),
            "bias": self.lora_bias,
            "task_type": self.task_type,
        }

    def grpo_kwargs(self, output_dir: str) -> dict:
        payload = {
            "output_dir": output_dir,
            "learning_rate": self.learning_rate,
            "num_generations": self.num_generations,
            # One group per generation call. Questions share an optimizer step via
            # gradient_accumulation_steps (see fdagym.train.group_sample.optimizer_batch).
            # Widening this to num_generations * prompts_per_step would make TRL roll
            # every question out in one forward.
            "generation_batch_size": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "beta": self.beta,
            "remove_unused_columns": False,
            "logging_steps": 1,
            "report_to": [],
            "temperature": self.temperature,
            "save_strategy": "steps",
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "use_vllm": self.use_vllm,
            "vllm_mode": self.vllm_mode,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_max_model_len": self.vllm_max_model_len,
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.bf16,
            "logging_first_step": True,
        }
        if self.max_prompt_length is not None:
            payload["max_prompt_length"] = self.max_prompt_length
        payload.update(self.extra)
        return payload


def select_grpo_kwargs(config: TrainConfig, output_dir: str, *, grpo_config_cls=None) -> dict:
    """Drop keys the installed GRPOConfig does not accept (max_prompt_length went away in TRL 0.28)."""
    payload = config.grpo_kwargs(output_dir)
    if grpo_config_cls is None:
        return payload
    accepted = set(signature(grpo_config_cls).parameters)
    return {key: value for key, value in payload.items() if key in accepted}


def for_7b(**overrides) -> TrainConfig:
    """A100 80GB defaults: same 8×400 schedule, vLLM + gradient checkpointing."""
    payload = {
        "model": QWEN_7B,
        "use_vllm": True,
        "gradient_checkpointing": True,
        "vllm_gpu_memory_utilization": 0.30,
    }
    payload.update(overrides)
    return TrainConfig(**payload)
