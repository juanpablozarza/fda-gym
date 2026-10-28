"""Hugging Face generate() as a rollout generator. Torch is imported inside the calls."""

from __future__ import annotations

from fdagym.train.rollout import RolloutTokenizer, TurnGeneration


class HFRolloutGenerator:
    """Left-padded ``model.generate``. Greedy when temperature is 0. The runner cuts at EOS."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt_token_ids, *, max_tokens: int, temperature: float):
        import torch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        width = max(len(row) for row in prompt_token_ids)
        input_rows = []
        mask_rows = []
        for row in prompt_token_ids:
            gap = width - len(row)
            input_rows.append([pad_id] * gap + list(row))
            mask_rows.append([0] * gap + [1] * len(row))
        device = next(self.model.parameters()).device
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=pad_id,
            )
        turns = []
        for index in range(len(prompt_token_ids)):
            new_tokens = tuple(int(token) for token in out[index, width:].tolist())
            turns.append(TurnGeneration(new_tokens, None))
        return turns


def load_hf_rollout(model_name: str, adapter: str | None = None):
    """Return ``(HFRolloutGenerator, RolloutTokenizer)``. Greedy eval loads the LoRA with PEFT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return HFRolloutGenerator(model, tokenizer), RolloutTokenizer(tokenizer)
