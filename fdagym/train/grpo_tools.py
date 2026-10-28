"""Tool-call GRPO on TRL 0.21.

``ToolGRPOTrainer`` replaces single-turn generation with ``rollout_batch``.
``completion_mask`` stays the attention mask, including tool tokens. ``loss_mask``
is the only mask the loss, the KL, and the clip ratios reduce over. Liger fuses
those into one kernel and would train on the tool tokens, so it is refused.
Log-probs use the same temperature and selective log-softmax as TRL 0.21, but
the vocabulary logits are scored in chunks (``logprob_chunk_size``) so an
8-by-~4k step does not allocate the full ``[batch, time, vocab]`` tensor.
``hf_smoke=True`` is the CPU path that generates without vLLM so a test can
run that log-prob and the loss.

Importing this module does not import torch. TRL is optional at import time so
the CPU tests can read the class. ``run_tool_grpo`` is what a GPU job calls.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

from fdagym.train.checkpoints import latest_checkpoint
from fdagym.train.config import TrainConfig, select_grpo_kwargs
from fdagym.train.curriculum import TOOL_MIX_ROWS, filter_pool_types, parse_pool_types, stage_pool, type_mix_pool
from fdagym.train.group_sample import PromptPool, collapse_group_counts, fill_zero_std_groups
from fdagym.train.hard_mix import rollout_tasks
from fdagym.train.prompt_filter import (
    apply_protected_floor,
    blocked_eval_ids,
    count_by_type,
    filter_current_tasks,
)
from fdagym.train.rollout import QwenToolCodec, RolloutLimits, RolloutTokenizer, VLLMGenerator, rollout_batch
from fdagym.train.token_logps import DEFAULT_LOGPROB_CHUNK_SIZE, chunked_token_logps
from fdagym.train.tool_guard import ParseRateGuard, ParseRateStop
from fdagym.train.tool_score import episode_metrics, episode_scores, frac_zero_std, pad_rows

try:
    from trl import GRPOTrainer as _GRPOTrainer
except ImportError:  # pragma: no cover - exercised only when the train extra is installed
    _GRPOTrainer = object


def _unwrap_causal_lm(model):
    """Peft and accelerator wrappers hide the causal LM that owns ``lm_head``."""
    seen: set[int] = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "module", None)
        if inner is not None and inner is not current:
            current = inner
            continue
        getter = getattr(current, "get_base_model", None)
        if callable(getter):
            current = getter()
            continue
        break
    if not hasattr(current, "lm_head") or not hasattr(current, "model"):
        raise TypeError(f"Tool GRPO log-probs need a causal LM with lm_head, got {type(current).__name__}")
    return current


class ToolGRPOTrainer(_GRPOTrainer):
    """GRPOTrainer that rolls out FDAEnv episodes inside ``_generate_and_score_completions``."""

    def __init__(
        self,
        *args,
        snapshot,
        tasks,
        codec,
        limits: RolloutLimits,
        tool_parse_bonus: float = 0.05,
        dense_reward: bool = True,
        run_id: str = "grpo-tools",
        hf_smoke: bool = False,
        parse_guard=None,
        memorization_box=None,
        memorization_every: int = 10,
        logprob_chunk_size: int = DEFAULT_LOGPROB_CHUNK_SIZE,
        dynamic_sampling: bool = False,
        max_resample_attempts: int = 3,
        prompt_pool: PromptPool | None = None,
        resample_seed: int = 0,
        **kwargs,
    ):
        if _GRPOTrainer is object:
            raise ImportError("Install training extras first: uv sync --extra train")
        super().__init__(*args, **kwargs)
        if self.use_liger_loss:
            raise RuntimeError("Tool GRPO cannot use Liger loss; it ignores loss_mask and would train on tool tokens.")
        # TRL 0.21 defaults. Sequence-level ratios and the entropy mask are not implemented;
        # the approved run leaves both at the default, so this does not change that job.
        if self.importance_sampling_level != "token":
            raise NotImplementedError(
                "Tool GRPO implements token-level importance sampling. "
                f"importance_sampling_level is {self.importance_sampling_level!r}."
            )
        if self.top_entropy_quantile < 1.0:
            raise NotImplementedError("Tool GRPO does not apply TRL's entropy mask. Leave top_entropy_quantile at 1.0.")
        self.hf_smoke = hf_smoke
        self.snapshot = snapshot
        self.tasks = tasks
        self.codec = codec
        self.limits = limits
        self.tool_parse_bonus = tool_parse_bonus
        self.dense_reward = dense_reward
        self.run_id = run_id
        self.parse_guard = parse_guard
        self.memorization_box = memorization_box if memorization_box is not None else {"fn": None}
        self.memorization_every = memorization_every
        if logprob_chunk_size < 1:
            raise ValueError(f"logprob_chunk_size must be >= 1, got {logprob_chunk_size}")
        self.logprob_chunk_size = logprob_chunk_size
        if max_resample_attempts < 0:
            raise ValueError(f"max_resample_attempts must be >= 0, got {max_resample_attempts}")
        if dynamic_sampling and prompt_pool is None:
            raise ValueError("dynamic_sampling requires a training PromptPool")
        self.dynamic_sampling = dynamic_sampling
        self.max_resample_attempts = max_resample_attempts
        self.prompt_pool = prompt_pool
        self.resample_seed = resample_seed
        self.rollout_tokenizer = RolloutTokenizer(self.processing_class)

    def log(self, logs, *args, **kwargs):
        """Let the hook see averaged reward metrics, then let TRL log them once.

        TRL 0.21 folds the per-step metric lists into the dict inside ``GRPOTrainer.log``,
        after this override. The dict Hugging Face passes in is loss, grad norm,
        and learning rate, so a hook that runs first never sees ``reward`` or
        ``rewards/parse_rate``. Average the same way TRL does, run the hook on
        that view, and copy only the keys TRL does not already log (memorization
        and ``parse_stop``) back onto ``logs``. ``GRPOTrainer.log`` still writes
        each reward metric once. The full report is also written to
        ``output_dir/memorization-<step>.json``.
        """
        from fdagym.train.wandb_log import attach_grpo_train_logs

        if self.model.training and isinstance(logs, dict):
            # Counts are one per question. Sum them so the logged step is the optimizer step.
            collapse_group_counts(self._metrics["train"])
            averaged = {key: sum(values) / len(values) for key, values in self._metrics["train"].items()}
            preview = {**logs, **averaged}
            attach_grpo_train_logs(
                preview,
                step=int(getattr(self.state, "global_step", 0) or 0),
                measure=(self.memorization_box or {}).get("fn"),
                every=self.memorization_every,
                guard=self.parse_guard,
                control=getattr(self, "control", None),
                report_dir=getattr(self.args, "output_dir", None),
            )
            for key, value in preview.items():
                if key not in averaged:
                    logs[key] = value
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            raise RuntimeError("Tool GRPO cannot use Liger loss; it ignores loss_mask and would train on tool tokens.")
        return self._compute_loss(model, inputs)

    def _generate_and_score_completions(self, inputs: list[dict]) -> dict:
        import torch

        if not self.use_vllm:
            if not self.hf_smoke:
                raise RuntimeError(
                    "Tool GRPO requires colocated vLLM (use_vllm=True, vllm_mode='colocate'). "
                    "use_vllm=False is only the CPU smoke path (hf_smoke=True)."
                )
            return self._generate_and_score_hf(inputs)
        if self.vllm_mode != "colocate":
            raise RuntimeError("Tool GRPO requires colocated vLLM (use_vllm=True, vllm_mode='colocate')")
        if self.vllm_tensor_parallel_size > 1:
            raise RuntimeError("Tool GRPO does not support vLLM tensor parallel in this build")
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        task_ids = [example["task_id"] for example in inputs]
        rollouts, scored, sampled = self._sample_groups(task_ids, mode)
        pad_id = self.rollout_tokenizer.pad_token_id
        prompt_ids, prompt_mask = pad_rows([list(item.prompt_ids) for item in rollouts], pad_value=pad_id, side="left")
        completion_ids, _presence = pad_rows(
            [list(item.completion_ids) for item in rollouts], pad_value=pad_id, side="right"
        )
        completion_mask, _presence = pad_rows(
            [list(item.attention_mask) for item in rollouts], pad_value=0, side="right"
        )
        loss_rows, _presence = pad_rows([list(item.loss_mask) for item in rollouts], pad_value=0, side="right")
        prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        prompt_mask_t = torch.tensor(prompt_mask, dtype=torch.long, device=device)
        completion_ids_t = torch.tensor(completion_ids, dtype=torch.long, device=device)
        completion_mask_t = torch.tensor(completion_mask, dtype=torch.long, device=device)
        loss_mask_t = torch.tensor(loss_rows, dtype=torch.long, device=device)

        env_totals = [row[0] for row in scored]
        dense_totals = [row[1] for row in scored]
        shaped = [row[2] for row in scored]
        hop_rates = [row[3] for row in scored]
        shaped_t = self.accelerator.gather(torch.tensor(shaped, dtype=torch.float32, device=device))
        env_t = self.accelerator.gather(torch.tensor(env_totals, dtype=torch.float32, device=device))
        dense_t = self.accelerator.gather(torch.tensor(dense_totals, dtype=torch.float32, device=device))
        hops_t = self.accelerator.gather(torch.tensor(hop_rates, dtype=torch.float32, device=device))
        if shaped_t.numel() % self.num_generations != 0:
            raise RuntimeError(f"{shaped_t.numel()} rewards is not divisible by num_generations={self.num_generations}")
        advantages = _group_advantages(shaped_t, self.num_generations, scale=self.scale_rewards)
        start = self.accelerator.process_index * len(inputs)
        advantages = advantages[start : start + len(inputs)]

        old_per_token_logps = None
        if self.num_iterations > 1 and completion_ids_t.numel():
            input_ids = torch.cat([prompt_ids_t, completion_ids_t], dim=1)
            attention_mask = torch.cat([prompt_mask_t, completion_mask_t], dim=1)
            old_per_token_logps = self._completion_logps(
                self.model, input_ids, attention_mask, completion_ids_t.size(1)
            )

        if mode == "train" and completion_mask_t.numel():
            attention_mask = torch.cat([prompt_mask_t, completion_mask_t], dim=1)
            seen = self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            self.state.num_input_tokens_seen += seen
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
            lengths = self.accelerator.gather_for_metrics(completion_mask_t.sum(1))
            self._metrics[mode]["completions/mean_length"].append(lengths.float().mean().item())
            loss_lengths = self.accelerator.gather_for_metrics(loss_mask_t.sum(1))
            self._metrics[mode]["loss_tokens/mean"].append(loss_lengths.float().mean().item())
        self._metrics[mode]["reward"].append(shaped_t.mean().item())
        self._metrics[mode]["reward_std"].append(shaped_t.view(-1, self.num_generations).std(dim=1).mean().item())
        self._metrics[mode]["rewards/env/mean"].append(env_t.mean().item())
        self._metrics[mode]["rewards/dense/mean"].append(dense_t.mean().item())
        self._metrics[mode]["rewards/hops/mean"].append(hops_t.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(frac_zero_std(shaped_t.tolist(), self.num_generations))
        if mode == "train":
            self._append_group_stats(sampled.stats)
        # Single process (colocated vLLM, no tensor parallel), so local rollouts are the whole batch.
        for key, value in episode_metrics(rollouts).items():
            self._metrics[mode][key].append(value)
        example_mask_t = torch.tensor(sampled.example_mask, dtype=torch.float32, device=device)
        return {
            "prompt_ids": prompt_ids_t,
            "prompt_mask": prompt_mask_t,
            "completion_ids": completion_ids_t,
            "completion_mask": completion_mask_t,
            "loss_mask": loss_mask_t,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "example_mask": example_mask_t,
        }

    def _sample_groups(self, task_ids: list[str], mode: str):
        """Roll the dataloader group. In train mode, replace tied groups up to the cap.

        Round 0 uses ``global_step`` as the rollout seed, the same seed a run
        without resampling used. Later rounds add ``10007 * round`` so a redraw
        is a new sample of that prompt.
        """
        base = int(self.state.global_step)

        def rollout_fn(ids: list[str], round_index: int):
            rollouts, scored = self._rollout_scored(ids, base + round_index * 10007)
            return list(zip(rollouts, scored, strict=True)), [row[2] for row in scored]

        enabled = self.dynamic_sampling and mode == "train"
        sampled = fill_zero_std_groups(
            task_ids,
            self.num_generations,
            rollout_fn,
            self.prompt_pool,
            enabled=enabled,
            max_attempts=self.max_resample_attempts,
            rng=random.Random(self.resample_seed + base),
        )
        rollouts = [payload[0] for payload in sampled.payloads]
        scored = [payload[1] for payload in sampled.payloads]
        return rollouts, scored, sampled

    def _rollout_scored(self, task_ids: list[str], seed: int):
        # Keep tool_budget and prompt_version. Rebuilding the dataclass by hand dropped new fields.
        limits = replace(self.limits, temperature=self.temperature)
        rollouts = rollout_batch(
            self.snapshot,
            self.tasks,
            task_ids,
            self.rollout_tokenizer,
            VLLMGenerator(self.llm),
            self.codec,
            limits,
            run_id=self.run_id,
            seed=seed,
        )
        by_id = {task["id"]: task for task in self.tasks}
        scored = [
            episode_scores(
                item,
                by_id[item.task_id],
                self.snapshot,
                dense=self.dense_reward,
                bonus=self.tool_parse_bonus,
            )
            for item in rollouts
        ]
        return rollouts, scored

    def _append_group_stats(self, stats: dict) -> None:
        mode = "train"
        for key in ("dropped", "resampled", "resample_rounds", "masked", "trained"):
            self._metrics[mode][f"groups/{key}"].append(stats[key])
        n_groups = stats["n_groups"]
        initial = stats["initial_zero_std"] / n_groups if n_groups else 0.0
        self._metrics[mode]["groups/initial_frac_zero_std"].append(initial)

    def _generate_and_score_hf(self, inputs: list[dict]) -> dict:
        """CPU smoke generation. No vLLM, no FDAEnv, no tool tokens.

        The GPU job sets ``use_vllm=True`` and never enters here. The dict has the
        same keys ``_compute_loss`` reads, so the log-prob, the loss, and the
        optimizer step are the TRL path that crashed on Modal.
        """
        import torch
        from transformers import Trainer
        from trl.models import unwrap_model_for_generation

        device = self.accelerator.device
        prompts = []
        for example in inputs:
            prompt = example["prompt"]
            if not isinstance(prompt, str):
                raise TypeError("The CPU smoke path expects a plain string prompt")
            prompts.append(prompt)
        prompt_inputs = self.processing_class(
            text=prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        # Trainer._prepare_inputs, not GRPOTrainer._prepare_inputs: the GRPO
        # override is what called us, and calling it again would recurse.
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        with (
            torch.no_grad(),
            unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped,
        ):
            generated = unwrapped.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )
        completion_ids = generated[:, prompt_ids.size(1) :]
        completion_mask = (completion_ids != self.pad_token_id).to(dtype=torch.long)
        # No tool tokens on this path: every non-pad completion token is a model token.
        loss_mask = completion_mask.clone()
        rewards = completion_ids.float().sum(dim=1)
        gathered = self.accelerator.gather(rewards)
        if gathered.numel() % self.num_generations != 0:
            raise RuntimeError(f"{gathered.numel()} rewards is not divisible by num_generations={self.num_generations}")
        advantages = _group_advantages(gathered, self.num_generations, scale=self.scale_rewards)
        start = self.accelerator.process_index * len(inputs)
        advantages = advantages[start : start + len(inputs)]
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["reward"].append(gathered.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(frac_zero_std(gathered.tolist(), self.num_generations))
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids.to(device),
            "completion_mask": completion_mask.to(device),
            "loss_mask": loss_mask.to(device),
            "advantages": advantages.to(device),
            "old_per_token_logps": None,
        }

    def _completion_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Per-token completion log-probs, without a full-vocabulary logit tensor.

        Same window and temperature as TRL 0.21: drop the last position, keep
        the completion tokens, divide logits by ``self.temperature``, then
        selective log-softmax. The backbone runs once, so LoRA dropout matches
        a single forward. ``lm_head`` is applied ``logprob_chunk_size`` token
        rows at a time. Entropy is not computed.
        """
        causal = _unwrap_causal_lm(model)
        outputs = causal.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state
        hidden = hidden[:, :-1, :]
        hidden = hidden[:, -logits_to_keep:, :]
        target_ids = input_ids[:, -logits_to_keep:]
        if hidden.shape[1] != target_ids.shape[1]:
            raise RuntimeError(f"log-prob window {hidden.shape[1]} does not match completion ids {target_ids.shape[1]}")
        return chunked_token_logps(hidden, target_ids, causal.lm_head, float(self.temperature), self.logprob_chunk_size)

    def _compute_loss(self, model, inputs):
        import torch

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        token_mask = inputs["loss_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # completion_mask is attention: 1 on model tokens and on tool tokens, 0 on pad.
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._completion_logps(model, input_ids, attention_mask, logits_to_keep)

        per_token_kl = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._completion_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._completion_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        # Missing mask (the CPU smoke path) trains every row. A 0 marks a tied group
        # that dynamic sampling could not replace: it is not a training example.
        # ``weighted_mean`` in group_sample.py is the same reduction, without torch.
        example_mask = inputs.get("example_mask")
        if example_mask is None:
            weights = torch.ones(advantages.size(0), device=advantages.device, dtype=advantages.dtype)
        else:
            weights = example_mask.to(device=advantages.device, dtype=advantages.dtype)
        advantages = advantages * weights
        token_mask = token_mask * weights.unsqueeze(1)
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # TRL 0.21 applies delta after coef_2 is taken from the unclamped ratio. Default is None.
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            per_seq = (per_token_loss * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1.0)
            # All-ones weights match a plain mean. A masked group adds nothing to either side.
            loss = (per_seq * weights).sum() / weights.sum().clamp(min=1.0)
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * token_mask).sum() / token_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * token_mask).sum() / (weights.sum().clamp(min=1.0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        denom = token_mask.sum().clamp(min=1.0)
        if per_token_kl is not None:
            mean_kl = (per_token_kl * token_mask).sum() / denom
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = (is_low_clipped * token_mask).sum() / denom
        high_clip = (is_high_clipped * token_mask).sum() / denom
        clip_ratio = (is_region_clipped * token_mask).sum() / denom
        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(_nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(_nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss


def _nanmin(tensor):
    """Min ignoring NaN. Torch 2.6 has ``Tensor.nanmean`` and no ``torch.nanmin``."""
    import torch

    flat = tensor.reshape(-1)
    if torch.isnan(flat).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(flat[~torch.isnan(flat)])


def _nanmax(tensor):
    """Max ignoring NaN. Same gap as ``_nanmin`` on torch 2.6."""
    import torch

    flat = tensor.reshape(-1)
    if torch.isnan(flat).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(flat[~torch.isnan(flat)])


def _group_advantages(rewards, group_size: int, *, scale: bool):
    """TRL's grouped advantage on a 1-D tensor. Unbiased std, then divide by std + 1e-4."""
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if not scale:
        return centered.reshape(-1)
    std = grouped.std(dim=1, keepdim=True)
    return (centered / (std + 1e-4)).reshape(-1)


def unused_reward(*_args, **_kwargs):
    """GRPOTrainer requires a reward function. Tool GRPO scores inside the rollout."""
    raise RuntimeError("Tool GRPO scores episodes inside the rollout. This reward function must not run.")


def build_tool_rows(
    tasks: list[dict],
    config: TrainConfig,
    codec,
    *,
    type_mix: dict | None = None,
    mix_rows: int = TOOL_MIX_ROWS,
    mix_seed: int = 0,
    hard_train_mix: dict | None = None,
    hard_tasks: list[dict] | None = None,
    prompt_filter=None,
    blocked_ids: set[str] | frozenset[str] | None = None,
    pool_types: str | tuple[str, ...] | list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Question-only prompts. No packed documents and no distractors.

    ``type_mix`` (the tool run passes ``ToolTrainConfig.type_mix``) draws ``mix_rows``
    rows by task type; rows can repeat. Without it, the Plan A stage pool is used.

    ``hard_train_mix`` (for example ``{"hard": 0.60, "current": 0.40}``) instead draws
    from ``hard_tasks`` (the hard training split) and ``tasks`` (the current train
    pool). Leave it None on the default path. It refuses ``split == "hard"``.

    An enabled ``prompt_filter`` edits only that current side, then tops
    ``boxed_warning`` up to the configured floor of the current slice. The
    hard-train count stays the mix weight. Eval, holdout, and test ids are not
    rows and raise if they appear in the current list.

    ``pool_types`` is the current-train allowlist (default: no composite).
    Hard-train compare / sponsor_hop are not filtered by it.
    """
    allowed = parse_pool_types(pool_types)
    if hard_train_mix is not None:
        from fdagym.train.hard_mix import mix_hard_train, prove_training_pool

        prove_training_pool(tasks, hard_tasks or [])
        current, type_report = filter_pool_types(tasks, allowed)
        filter_report = None
        if prompt_filter is not None and prompt_filter.enabled:
            if blocked_ids is None:
                blocked_ids = blocked_eval_ids()
            current, filter_report = filter_current_tasks(current, prompt_filter, blocked_ids=blocked_ids)
        selected, mix = mix_hard_train(
            current, hard_tasks or [], weights=hard_train_mix, n_rows=mix_rows, seed=mix_seed
        )
        mix["pool_types"] = type_report
        if prompt_filter is not None and prompt_filter.enabled and prompt_filter.protected_floor > 0:
            selected, floor_report = apply_protected_floor(
                selected, current, config=prompt_filter, seed=mix_seed + 1
            )
            mix["floor"] = floor_report
        if filter_report is not None:
            mix["filter"] = filter_report
            mix["by_type"] = count_by_type(selected)
    elif type_mix is not None:
        weights = {kind: weight for kind, weight in type_mix.items() if kind in allowed}
        if not weights:
            shown = ", ".join(allowed)
            raise RuntimeError(f"pool_types [{shown}] left the type mix empty")
        selected, mix = type_mix_pool(tasks, weights, n_rows=mix_rows, seed=mix_seed)
        mix["pool_types"] = {"allowlist": list(allowed)}
    else:
        selected, mix = stage_pool(tasks, config.stage)
    if not selected:
        raise RuntimeError("stage pool is empty")
    rows = [{"prompt": codec.prompt(task["question"]), "task_id": task["id"]} for task in selected]
    return rows, mix


def resolve_tool_start(output_dir: str | Path, sft_adapter: str | Path | None, *, resume: bool) -> dict:
    """Resume a GRPO checkpoint, otherwise require the SFT adapter. Never start from the raw base.

    ``sft_adapter`` is a local directory or a Hugging Face repo id
    (``org/name``). Hub ids need ``HF_TOKEN`` for a private repo. A missing
    local path does not fall back to the base model.
    """
    from fdagym.train.sft_launch import is_hub_adapter

    out = Path(output_dir)
    checkpoint = latest_checkpoint(out) if resume else None
    if checkpoint is not None:
        return {"mode": "resume", "checkpoint": str(checkpoint), "sft_adapter": None}
    if sft_adapter is None or not str(sft_adapter).strip():
        raise FileNotFoundError(
            f"SFT adapter not found at {sft_adapter}. Run the warm-up before tool GRPO. "
            "A missing adapter does not fall back to the base model."
        )
    text = str(sft_adapter).strip()
    path = Path(text)
    if path.is_dir():
        return {"mode": "sft", "checkpoint": None, "sft_adapter": str(path)}
    if is_hub_adapter(text):
        return {"mode": "sft", "checkpoint": None, "sft_adapter": text}
    raise FileNotFoundError(
        f"SFT adapter not found at {sft_adapter}. Run the warm-up before tool GRPO. "
        "A missing adapter does not fall back to the base model."
    )


def load_sft_policy(base_name: str, adapter: str):
    """Load the warm-up LoRA for more training. vLLM must see the base model id, not the adapter path."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, trust_remote_code=True)
    base.config._name_or_path = base_name
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    inner = policy.get_base_model()
    inner.config._name_or_path = base_name
    inner.config.use_cache = False
    # PeftModel has no name_or_path property. TRL passes model.name_or_path to vLLM,
    # and an adapter directory is not a base model the engine can load.
    policy.name_or_path = base_name
    return policy


def run_tool_grpo(
    snapshot,
    tasks: list[dict],
    config: TrainConfig,
    *,
    run_id: str,
    results_dir: Path,
    sft_adapter: str | Path | None,
    tool_parse_bonus: float = 0.05,
    limits: RolloutLimits | None = None,
    resume: bool = True,
    on_save=None,
    type_mix: dict | None = None,
    mix_rows: int = TOOL_MIX_ROWS,
    mix_seed: int = 0,
    guard: ParseRateGuard | None = None,
    hard_train_mix: dict | None = None,
    hard_tasks: list[dict] | None = None,
    measure_memorization=None,
    memorization_every: int = 10,
    memorization: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    dynamic_sampling: bool = False,
    max_resample_attempts: int = 3,
    prompt_filter=None,
    prompts_per_step: int = 1,
    pool_types: str | tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Fit tool-call LoRA with GRPO. Requires ``uv sync --extra train`` and a colocated vLLM."""
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, TrainerCallback
        from trl import GRPOConfig
    except ImportError as exc:
        raise ImportError("Install training extras first: uv sync --extra train") from exc

    limits = limits or RolloutLimits()
    codec = QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    filtering = prompt_filter is not None and prompt_filter.enabled
    blocked = blocked_eval_ids() if dynamic_sampling or filtering else None
    rows, mix = build_tool_rows(
        tasks,
        config,
        codec,
        type_mix=type_mix,
        mix_rows=mix_rows,
        mix_seed=mix_seed,
        hard_train_mix=hard_train_mix,
        hard_tasks=hard_tasks,
        prompt_filter=prompt_filter,
        blocked_ids=blocked,
        pool_types=pool_types,
    )
    if isinstance(mix, dict) and "by_type" in mix:
        print("hard-rl post-filter mix " + json.dumps(mix["by_type"]))
    prompt_pool = PromptPool([row["task_id"] for row in rows], blocked or set()) if dynamic_sampling else None
    out = Path(results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    start = resolve_tool_start(out, sft_adapter, resume=resume)
    if start["mode"] == "sft":
        model = load_sft_policy(config.model, start["sft_adapter"])
        peft_config = None
    else:
        model = config.model
        peft_config = LoraConfig(**config.lora_kwargs())
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    guard = guard if guard is not None else ParseRateGuard()
    # The live measurer is installed after the trainer exists. Unflagged runs leave this None.
    measure_box = {"fn": measure_memorization}
    from fdagym.train.wandb_log import (
        build_run_config,
        start_wandb,
        step_from_checkpoint,
    )

    session = start_wandb(
        run_id,
        build_run_config(
            run_id=run_id,
            job="grpo",
            prompt_version=limits.prompt_version,
            tool_budget=limits.tool_budget,
            hard_train_mix=hard_train_mix,
            seeds=[mix_seed],
            adapter=None if sft_adapter is None else str(sft_adapter),
            model=config.model,
            extra={
                "stage": config.stage,
                "mix": mix,
                "max_steps": config.max_steps,
                "prompts_per_step": prompts_per_step,
                "num_generations": config.num_generations,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "dynamic_sampling": dynamic_sampling,
                "max_resample_attempts": max_resample_attempts,
                "difficulty_filter": filtering,
                "temperature": config.temperature,
                "pool_types": list(parse_pool_types(pool_types)),
                "post_filter_by_type": mix.get("by_type") if isinstance(mix, dict) else None,
            },
        ),
        project=wandb_project,
        entity=wandb_entity,
        resume_step=step_from_checkpoint(start["checkpoint"]),
    )

    class TrainSaveCallback(TrainerCallback):
        def on_save(self, args_, state, control, **kwargs):
            if on_save is not None:
                on_save(state)

    kwargs = select_grpo_kwargs(config, str(out), grpo_config_cls=GRPOConfig)
    kwargs["report_to"] = session.report_to
    kwargs["run_name"] = run_id
    env_tasks = rollout_tasks(tasks, hard_tasks if hard_train_mix is not None else None)
    trainer = ToolGRPOTrainer(
        model=model,
        reward_funcs=unused_reward,
        args=GRPOConfig(**kwargs),
        train_dataset=Dataset.from_list(rows),
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[TrainSaveCallback()] if on_save is not None else [],
        snapshot=snapshot,
        tasks=env_tasks,
        codec=codec,
        limits=limits,
        tool_parse_bonus=tool_parse_bonus,
        dense_reward=config.dense_reward,
        run_id=run_id,
        parse_guard=guard,
        memorization_box=measure_box,
        memorization_every=memorization_every,
        logprob_chunk_size=config.logprob_chunk_size,
        dynamic_sampling=dynamic_sampling,
        max_resample_attempts=max_resample_attempts,
        prompt_pool=prompt_pool,
        resample_seed=mix_seed,
    )
    if memorization and measure_box["fn"] is None:
        from fdagym.train.eval_tools import live_memorization_measure

        measure_box["fn"] = live_memorization_measure(trainer, limits=limits, codec=codec, run_id=run_id)
    try:
        trainer.train(resume_from_checkpoint=start["checkpoint"])
        if guard.reason:
            # Not "adapter": the pipeline treats that directory as a finished stage.
            stopped = out / "adapter-stopped"
            trainer.save_model(str(stopped))
            (out / "stop.json").write_text(
                json.dumps({"reason": guard.reason, "step": trainer.state.global_step}, indent=2)
            )
            if on_save is not None:
                on_save(None)
            raise ParseRateStop(f"tool GRPO stopped at step {trainer.state.global_step}: {guard.reason}")
        adapter = out / "adapter"
        trainer.save_model(str(adapter))
        mix_record = {
            "stage": config.stage,
            "mix": mix,
            "n": len(rows),
            "protocol": "tool_rollout",
            "tool_parse_bonus": tool_parse_bonus,
            "dense_reward": config.dense_reward,
            "temperature": config.temperature,
            "type_mix": type_mix,
            "start": start["mode"],
        }
        if hard_train_mix is not None:
            mix_record["hard_train_mix"] = hard_train_mix
        mix_record["prompts_per_step"] = prompts_per_step
        mix_record["dynamic_sampling"] = dynamic_sampling
        mix_record["max_resample_attempts"] = max_resample_attempts
        mix_record["difficulty_filter"] = filtering
        (out / "mix.json").write_text(json.dumps(mix_record, indent=2))
        if on_save is not None:
            on_save(None)
        return {
            "run_id": run_id,
            "output_dir": str(out),
            "adapter": str(adapter),
            "n_prompts": len(rows),
            "mix": mix,
            "resumed": start["checkpoint"],
            "sft_adapter": start["sft_adapter"],
            "model": config.model,
            "protocol": "tool_rollout",
        }
    finally:
        session.finish()
