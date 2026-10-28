"""One real GRPO step on TRL 0.21.0 with a tiny random Qwen2, on CPU.

Generation is the isolated ``hf_smoke`` path (``use_vllm=False``). vLLM 0.8.5 does not
run here. The log-prob, the loss, and the optimizer step go through ``ToolGRPOTrainer``.
Log-probs are ``chunked_token_logps`` (TRL 0.21 removed ``_get_per_token_logps`` and
its replacement materializes the full vocabulary).

Run with the pinned interpreter (torch 2.6 CPU, transformers 4.55, trl 0.21, peft 0.15.2):

    /tmp/fdagym-trl021/bin/python tests/train/trl_grpo_smoke.py
"""

from __future__ import annotations

import inspect
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import GRPOConfig, GRPOTrainer

from fdagym.train.grpo_tools import ToolGRPOTrainer, unused_reward
from fdagym.train.rollout import RolloutLimits


def _tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for i in range(4, vocab_size):
        vocab[f"t{i}"] = i
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id != 0 or tokenizer.eos_token_id != 2:
        raise RuntimeError(f"special ids drifted: pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}")
    return tokenizer


def main() -> None:
    import transformers
    import trl

    if trl.__version__ != "0.21.0":
        raise SystemExit(f"expected trl 0.21.0, found {trl.__version__}")
    if not transformers.__version__.startswith("4.55."):
        raise SystemExit(f"expected transformers 4.55.x, found {transformers.__version__}")
    if not torch.__version__.startswith("2.6."):
        raise SystemExit(f"expected torch 2.6.x, found {torch.__version__}")
    if hasattr(GRPOTrainer, "_get_per_token_logps"):
        raise SystemExit("TRL 0.21.0 unexpectedly still has _get_per_token_logps")
    if not hasattr(GRPOTrainer, "_get_per_token_logps_and_entropies"):
        raise SystemExit("TRL 0.21.0 is missing _get_per_token_logps_and_entropies")
    if not hasattr(GRPOTrainer, "_move_model_to_vllm"):
        raise SystemExit("TRL 0.21.0 is missing _move_model_to_vllm")
    logps_src = inspect.getsource(GRPOTrainer._get_per_token_logps_and_entropies)
    if "self.temperature" not in logps_src or "selective_log_softmax" not in logps_src:
        raise SystemExit("log-prob method is not the TRL 0.21 implementation")

    vocab_size = 128
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    config._name_or_path = "tiny-random-qwen2"
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.config.use_cache = True
    tokenizer = _tokenizer(vocab_size)
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    rows = [
        {"prompt": "t4 t5 t6 t7", "task_id": "a"},
        {"prompt": "t8 t9 t10 t11", "task_id": "b"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        args = GRPOConfig(
            output_dir=tmp,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_generations=2,
            generation_batch_size=2,
            max_completion_length=4,
            max_prompt_length=32,
            max_steps=1,
            beta=0.04,
            temperature=0.8,
            top_k=50,
            min_p=0.0,
            bf16=False,
            fp16=False,
            use_vllm=False,
            gradient_checkpointing=False,
            logging_steps=1,
            logging_first_step=True,
            report_to="none",
            save_strategy="no",
            remove_unused_columns=False,
            seed=0,
            dataloader_pin_memory=False,
            disable_dropout=True,
        )
        trainer = ToolGRPOTrainer(
            model=model,
            reward_funcs=unused_reward,
            args=args,
            train_dataset=Dataset.from_list(rows),
            peft_config=peft_config,
            processing_class=tokenizer,
            snapshot=None,
            tasks=[],
            codec=None,
            limits=RolloutLimits(temperature=0.8, max_new_tokens=4, max_context_tokens=64),
            hf_smoke=True,
            logprob_chunk_size=2,
        )
        for name in (
            "use_vllm",
            "vllm_mode",
            "vllm_tensor_parallel_size",
            "use_liger_loss",
            "temperature",
            "num_generations",
            "num_iterations",
            "scale_rewards",
            "beta",
            "epsilon_low",
            "epsilon_high",
            "loss_type",
            "max_completion_length",
            "importance_sampling_level",
            "top_entropy_quantile",
            "model_kwarg_keys",
            "generation_config",
            "_metrics",
            "ref_model",
        ):
            if not hasattr(trainer, name):
                raise SystemExit(f"ToolGRPOTrainer is missing {name}, which the GPU step reads")
        if trainer.use_vllm:
            raise SystemExit("smoke trainer must run with use_vllm=False")
        if not hasattr(trainer.state, "num_input_tokens_seen"):
            raise SystemExit("TrainerState is missing num_input_tokens_seen")
        calls = {"n": 0, "chunks": []}
        from fdagym.train import grpo_tools

        real = grpo_tools.chunked_token_logps

        def _counted(*args, **kwargs):
            calls["n"] += 1
            calls["chunks"].append(kwargs.get("chunk_size", args[4] if len(args) > 4 else None))
            return real(*args, **kwargs)

        grpo_tools.chunked_token_logps = _counted
        # lora_A's gradient is zero on the first step (lora_B starts at 0). lora_B moves.
        watched = next(name for name, _param in trainer.model.named_parameters() if "lora_B" in name)
        before = trainer.model.get_parameter(watched).detach().clone()
        trainer.train()
        after = trainer.model.get_parameter(watched).detach()
        losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
        if trainer.state.global_step != 1:
            raise SystemExit(f"expected global_step 1, got {trainer.state.global_step}")
        if not losses or not math.isfinite(float(losses[-1])):
            raise SystemExit(f"loss was not finite: {losses}")
        if calls["n"] < 2:
            raise SystemExit(f"expected policy and reference log-probs, counted {calls['n']}")
        if calls["chunks"] != [2, 2]:
            raise SystemExit(f"expected chunk_size 2 on policy and reference, got {calls['chunks']}")
        if torch.equal(before, after):
            raise SystemExit(f"optimizer step left {watched} unchanged")
        logged = next(row for row in trainer.state.log_history if "frac_reward_zero_std" in row)
        for key in ("loss", "kl", "reward", "frac_reward_zero_std", "clip_ratio/low_min", "clip_ratio/high_max"):
            if key not in logged or not math.isfinite(float(logged[key])):
                raise SystemExit(f"missing or non-finite log field {key}: {logged}")
        print(
            "PASS"
            f" trl={trl.__version__}"
            f" transformers={transformers.__version__}"
            f" torch={torch.__version__}"
            f" step={trainer.state.global_step}"
            f" loss={float(losses[-1]):.6f}"
            f" logprob_calls={calls['n']}"
            f" updated={watched}"
        )


if __name__ == "__main__":
    main()
