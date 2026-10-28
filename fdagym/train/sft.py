"""Plan A supervised warm-up: copy a gold answer that is already in the prompt.

Train-split tasks only. The assistant target is submit_answer JSON: the gold answer
and one citation per gold entry (endpoint, id, field, and a quote that is a verbatim
span of the rendered user message). A task is skipped when that answer text or that
citation is not in the user message. The system prompt is not searched, so NOT_FOUND
there does not count. Boolean tasks count as visible when a gold keyword is in the
documents; a yes/no task with no keyword is skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from inspect import signature
from pathlib import Path

from fdagym.rewards.citation import _TOOL_ENDPOINT, _collapse, _dicts, _render, _resolve
from fdagym.snapshot import Snapshot
from fdagym.train.config import TrainConfig
from fdagym.train.context import assemble_documents
from fdagym.train.gold import _accepted_id, gold_answer
from fdagym.train.prompt import render_messages
from fdagym.train.prompt_len import prompt_tokens

SFT_RUN_ID = "sft-plan-a-qwen3b"
SFT_LR = 2e-4
SFT_EPOCHS = 1
SFT_MAX_LENGTH = 8192


def build_sft_rows(
    snapshot: Snapshot,
    tasks: list[dict],
    config: TrainConfig | None = None,
) -> tuple[list[dict], dict]:
    """Chat rows for SFT. ``stats`` counts skips; nothing here imports torch."""
    config = config or TrainConfig()
    train = [task for task in tasks if task.get("split") != "test"]
    rows = []
    skipped: Counter[str] = Counter()
    for task in train:
        docs = assemble_documents(
            snapshot,
            tasks,
            task,
            n_distractors=config.n_distractors,
            max_chars=config.max_pack_chars,
        )
        messages = render_messages(task, docs)
        user = messages[1]["content"]
        payload, reason = target_from_docs(task, docs, user)
        if payload is None:
            skipped[reason] += 1
            continue
        completion = json.dumps(payload, ensure_ascii=False)
        chat = [*messages, {"role": "assistant", "content": completion}]
        if prompt_tokens(chat) > SFT_MAX_LENGTH:
            skipped["over_length"] += 1
            continue
        rows.append({"messages": chat, "task_id": task["id"], "completion": completion})
    stats = {
        "n_train": len(train),
        "n_kept": len(rows),
        "skipped": dict(skipped),
    }
    return rows, stats


def target_from_docs(task: dict, docs: list[dict], user: str) -> tuple[dict | None, str]:
    if not answer_in_prompt(task, user):
        return None, "answer_not_in_prompt"
    citations = []
    for gold in task.get("gold_citations") or []:
        found = _citation(task, docs, user, gold)
        if found is None:
            return None, "citation_not_in_prompt"
        citations.append(found)
    if not citations:
        return None, "citation_not_in_prompt"
    return {"answer": gold_answer(task), "citations": citations}, "kept"


def answer_in_prompt(task: dict, user: str) -> bool:
    """Gold answer text is in the user message. The system prompt is not consulted."""
    answer = gold_answer(task)
    if isinstance(answer, bool):
        return False
    text = str(answer).strip()
    if not text:
        return False
    if text.upper() == "NOT_FOUND":
        return "NOT_FOUND" in user
    gold = task.get("gold") or {}
    if gold.get("kind") == "boolean":
        keywords = [str(item) for item in gold.get("keywords") or [] if str(item).strip()]
        if not keywords:
            return False
        folded = user.lower()
        return all(item.lower() in folded for item in keywords)
    return text in user or text.lower() in user.lower()


def verbatim_quote(field_text: str, prompt: str) -> str | None:
    """A span of ``field_text`` that occurs unchanged in the rendered prompt."""
    collapsed = _collapse(field_text)
    if not collapsed:
        return None
    if len(collapsed) <= 12:
        if collapsed in prompt:
            return collapsed
        return None
    for source in (field_text, collapsed):
        found = _window_in(source, prompt)
        if found:
            return found
    return None


def _window_in(source: str, prompt: str) -> str | None:
    text = source.strip()
    if len(text) >= 4 and text in prompt:
        return text[:200]
    size = min(80, len(text))
    if size < 4:
        return None
    for start in range(0, max(1, len(text) - 3), 20):
        chunk = text[start : start + size].strip()
        if len(chunk) >= 4 and chunk in prompt:
            return chunk[:200]
    prefix = text[:200]
    while len(prefix) >= 4:
        if prefix in prompt:
            return prefix
        prefix = prefix[:-1]
    return None


def _citation(task: dict, docs: list[dict], user: str, gold: dict) -> dict | None:
    for pack in docs:
        if pack.get("task_id") != task["id"]:
            continue
        if _TOOL_ENDPOINT.get(pack.get("tool")) != gold.get("endpoint"):
            continue
        for node in _dicts(pack.get("output")):
            if not isinstance(node, dict):
                continue
            cid = _accepted_id(node, gold)
            if cid is None or cid not in user:
                continue
            for field in gold.get("fields") or []:
                text = _render(_resolve(node, field))
                if not text:
                    continue
                quote = verbatim_quote(text, user)
                if quote is None:
                    continue
                return {"endpoint": gold["endpoint"], "id": cid, "field": field, "quote": quote[:500]}
    return None


def run_sft(
    snapshot: Snapshot,
    tasks: list[dict],
    config: TrainConfig | None = None,
    *,
    output_dir: Path,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> dict:
    """One epoch of LoRA SFT. Imports torch/TRL only when called."""
    config = config or TrainConfig()
    rows, stats = build_sft_rows(snapshot, tasks, config)
    if not rows:
        raise RuntimeError(f"sft pool is empty: {stats}")
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError("Install training extras first: uv sync --extra train") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = out / "adapter"
    from fdagym.train.checkpoints import latest_checkpoint
    from fdagym.train.wandb_log import build_run_config, start_wandb, step_from_checkpoint

    checkpoint = latest_checkpoint(out)
    session = start_wandb(
        SFT_RUN_ID,
        build_run_config(
            run_id=SFT_RUN_ID,
            job="sft",
            prompt_version=None,
            tool_budget=None,
            hard_train_mix=None,
            seeds=[0],
            adapter=str(adapter),
            model=config.model,
        ),
        project=wandb_project,
        entity=wandb_entity,
        resume_step=step_from_checkpoint(checkpoint),
    )
    payload = _sft_kwargs(out, SFTConfig)
    payload["report_to"] = session.report_to
    payload["run_name"] = SFT_RUN_ID
    trainer = SFTTrainer(
        model=config.model,
        args=SFTConfig(**payload),
        train_dataset=Dataset.from_list([{"messages": row["messages"]} for row in rows]),
        peft_config=LoraConfig(**config.lora_kwargs()),
    )
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(adapter))
        (out / "sft_stats.json").write_text(json.dumps({"run_id": SFT_RUN_ID, **stats}, indent=2) + "\n")
        return {"run_id": SFT_RUN_ID, "adapter": str(adapter), "n": len(rows), "stats": stats}
    finally:
        session.finish()


def _sft_kwargs(output_dir: Path, config_cls) -> dict:
    payload = {
        "output_dir": str(output_dir),
        "num_train_epochs": SFT_EPOCHS,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": SFT_LR,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": [],
        "max_length": SFT_MAX_LENGTH,
        "max_seq_length": SFT_MAX_LENGTH,
        "packing": False,
    }
    accepted = set(signature(config_cls).parameters)
    selected = {key: value for key, value in payload.items() if key in accepted}
    if "max_length" not in selected and "max_seq_length" not in selected:
        raise RuntimeError("SFTConfig accepts neither max_length nor max_seq_length; refusing to truncate")
    return selected
