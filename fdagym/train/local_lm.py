"""HuggingFace generate() wrapper. Imported only on GPU boxes; tests inject a fake instead."""

from __future__ import annotations

EVAL_TEMPERATURE = 0.0


def seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generator(
    model_name: str,
    *,
    adapter: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = EVAL_TEMPERATURE,
    seed: int = 0,
):
    """Return ``generate(messages, seed=...) -> str``. Greedy when temperature is 0."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_torch(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(messages: list[dict], seed: int = seed) -> str:
        seed_torch(seed)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate
