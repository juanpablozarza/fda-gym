"""TRL 0.21 ``GRPOTrainer.log`` sees reward metrics only inside that method.

Hugging Face calls ``log`` with loss, grad norm, and learning rate. The reward
and parse-rate lists live on ``_metrics`` until ``GRPOTrainer.log`` averages
them. This drives that path: realistic incoming logs, populated ``_metrics``,
then ``ToolGRPOTrainer.log``.

    /tmp/fdagym-trl021/bin/python tests/train/trl_grpo_log_flow.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import Trainer
from trl import GRPOConfig

from fdagym.train.grpo_tools import ToolGRPOTrainer, unused_reward
from fdagym.train.rollout import RolloutLimits
from fdagym.train.tool_guard import ParseRateGuard
from tests.train.trl_grpo_smoke import _tokenizer


def _trainer(tmp: str, *, every: int, guard: ParseRateGuard, measure) -> ToolGRPOTrainer:
    vocab_size = 128
    from transformers import Qwen2Config, Qwen2ForCausalLM

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
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.train()
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
        beta=0.0,
        temperature=0.8,
        bf16=False,
        fp16=False,
        use_vllm=False,
        gradient_checkpointing=False,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        seed=0,
        dataloader_pin_memory=False,
        log_completions=False,
    )
    trainer = ToolGRPOTrainer(
        model=model,
        reward_funcs=unused_reward,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": "t4 t5", "task_id": "a"}, {"prompt": "t6 t7", "task_id": "b"}]),
        peft_config=LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        processing_class=_tokenizer(vocab_size),
        snapshot=None,
        tasks=[],
        codec=None,
        limits=RolloutLimits(temperature=0.8, max_new_tokens=4, max_context_tokens=64),
        hf_smoke=True,
        parse_guard=guard,
        memorization_box={"fn": measure},
        memorization_every=every,
    )
    trainer.model.train()
    return trainer


def _fill(trainer: ToolGRPOTrainer, reward: list[float], parse_rate: float, turn_rate: float) -> None:
    bucket = trainer._metrics["train"]
    bucket.clear()
    bucket["reward"].extend(reward)
    bucket["rewards/parse_rate"].append(parse_rate)
    bucket["rewards/turn_parse_rate"].append(turn_rate)
    bucket["rewards/env/mean"].append(0.2)


REPORT = {
    "gap": {"strict": 0.2},
    "known": {"mean_tool_calls": 3.0, "strict": 0.5},
    "unseen": {"mean_tool_calls": 4.5, "strict": 0.3},
    "swapped_facts": {"n": 30, "follow": 12, "follow_rate": 0.4, "original": 8, "other": 10},
    "answered_without_looking": {"rate": 0.0, "rule": "gold value or gold_citation id in a tool output"},
    "flags": {"level": "warn", "swapped_facts": "warn"},
    "auto_stop": False,
}


def main() -> None:
    import trl

    if trl.__version__ != "0.21.0":
        raise SystemExit(f"expected trl 0.21.0, found {trl.__version__}")
    seen: list[int] = []

    def measure(step: int) -> dict:
        seen.append(step)
        return REPORT

    guard = ParseRateGuard(window=2, min_rate=0.9)
    logged: list[dict] = []
    real_log = Trainer.log

    def _capture(self, logs, *args, **kwargs):
        logged.append(dict(logs))
        return real_log(self, logs, *args, **kwargs)

    Trainer.log = _capture
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, every=2, guard=guard, measure=measure)
            if not trainer.model.training:
                raise SystemExit("trainer must be in train mode so the hook reads train metrics")
            hf_logs = {"loss": 1.5, "grad_norm": 0.3, "learning_rate": 1e-5}
            trainer.state.global_step = 1
            _fill(trainer, [0.4, 0.6], 0.5, 0.25)
            before = len(trainer.state.log_history)
            trainer.log(dict(hf_logs))
            if seen:
                raise SystemExit(f"memorization ran on step 1: {seen}")
            if guard.seen.get("rewards/parse_rate") != [0.5]:
                raise SystemExit(f"parse guard missed step 1: {guard.seen}")
            if guard.seen.get("rewards/turn_parse_rate") != [0.25]:
                raise SystemExit(f"turn parse guard missed step 1: {guard.seen}")
            if len(logged) != 1:
                raise SystemExit(f"Trainer.log ran {len(logged)} times on one step")
            row = logged[0]
            if row.get("reward") != 0.5:
                raise SystemExit(f"reward was not the average of _metrics: {row.get('reward')}")
            if row.get("rewards/parse_rate") != 0.5:
                raise SystemExit(f"parse rate was not logged once from _metrics: {row}")
            if row.get("loss") != 1.5:
                raise SystemExit(f"loss dropped: {row}")
            if any(key.startswith("memorization/") for key in row):
                raise SystemExit(f"memorization logged off cadence: {row}")
            if len(trainer.state.log_history) != before + 1:
                raise SystemExit("log history did not grow by one row")
            if trainer._metrics["train"]:
                raise SystemExit(f"_metrics was not cleared: {dict(trainer._metrics['train'])}")

            trainer.state.global_step = 2
            _fill(trainer, [0.2], 0.4, 0.2)
            trainer.log(dict(hf_logs))
            if seen != [2]:
                raise SystemExit(f"memorization cadence was {seen}, expected [2]")
            if len(logged) != 2:
                raise SystemExit(f"Trainer.log ran {len(logged)} times over two steps")
            second = logged[1]
            if second.get("reward") != 0.2:
                raise SystemExit(f"second reward logged wrong: {second.get('reward')}")
            if second.get("memorization/gap/strict") != 0.2:
                raise SystemExit(f"memorization missing from the logged dict: {second}")
            if second.get("memorization/known/mean_tool_calls") != 3.0:
                raise SystemExit(f"known tool calls missing: {second}")
            if second.get("memorization/unseen/mean_tool_calls") != 4.5:
                raise SystemExit(f"unseen tool calls missing: {second}")
            if second.get("memorization/swapped/follow_rate") != 0.4:
                raise SystemExit(f"swapped follow rate missing: {second}")
            if second.get("memorization/swapped/follow") != 12:
                raise SystemExit(f"swapped follow count missing: {second}")
            if second.get("memorization/swapped/n") != 30:
                raise SystemExit(f"swapped n missing: {second}")
            if "memorization/answered_without_looking/rule" in second:
                raise SystemExit(f"look rule was logged as a metric: {second}")
            if second.get("memorization/auto_stop") is not False:
                raise SystemExit(f"memorization auto_stop changed: {second}")
            saved = Path(tmp) / "memorization-2.json"
            if json.loads(saved.read_text()) != REPORT:
                raise SystemExit(f"full report was not saved: {saved}")
            if (Path(tmp) / "memorization-1.json").exists():
                raise SystemExit("wrote a memorization report off cadence")
            if "parse_stop" not in second:
                raise SystemExit(f"parse guard did not fire: {second}")
            if not trainer.control.should_training_stop:
                raise SystemExit("parse guard did not set should_training_stop")
            reward_keys = [key for key in second if key == "reward"]
            if reward_keys != ["reward"]:
                raise SystemExit(f"reward key repeated: {list(second)}")

            trainer.model.eval()
            trainer.state.global_step = 2
            trainer._metrics["eval"]["reward"].append(0.9)
            trainer._metrics["eval"]["rewards/parse_rate"].append(1.0)
            calls = len(seen)
            trainer.log(dict(hf_logs))
            if len(seen) != calls:
                raise SystemExit("eval log ran a memorization measurement")
            if logged[-1].get("eval_reward") != 0.9:
                raise SystemExit(f"eval metrics were not logged by TRL: {logged[-1]}")
            if "reward" in logged[-1]:
                raise SystemExit(f"eval log also wrote an unprefixed reward: {logged[-1]}")
            written = sorted(path.name for path in Path(tmp).glob("memorization-*.json"))
            if written != ["memorization-2.json"]:
                raise SystemExit(f"eval changed the saved reports: {written}")
    finally:
        Trainer.log = real_log
    print("PASS trl=0.21.0 log_flow steps=2 memorization=1 parse_stop=1 trainer_log_calls=1")


if __name__ == "__main__":
    main()
