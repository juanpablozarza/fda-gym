"""Tokenize teacher messages into SFT rows. Loss is assistant turns only.

The renderer is ``QwenToolCodec`` (prompt v4 / budget 8 by default), the same
objects ``FDAEnv`` eval uses. System and first user become the prompt and are
masked. Tool observations are masked. Assistant tokens (plus EOS) are labels.
Torch stays out of this module.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fdagym.train.rollout import (
    TOOL_MARK,
    QwenToolCodec,
    RolloutLimits,
    action_from_obj,
    json_objects,
    observation_payload,
)
from fdagym.train.sft_tools import IGNORE_INDEX, sft_example

_ASSISTANT = "assistant"
_TOOL_ROLES = frozenset({"tool", "function"})


def user_question(messages: list[dict] | tuple[dict, ...]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("messages have no user question")


def assistant_text(message: dict, codec) -> str:
    """Hermes ``<tool_call>`` block, or JSON ``{tool, arguments}`` rendered through the codec."""
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("assistant content must be a string")
    text = content.strip()
    if not text:
        raise ValueError("assistant content is empty")
    if "<tool_call>" in text:
        return text
    for obj in json_objects(text):
        action = action_from_obj(obj)
        if action is not None:
            return codec.action_text(action)
    return text


def tool_observation(message: dict) -> dict:
    content = message.get("content")
    name = message.get("name")
    if isinstance(content, dict):
        payload = dict(content)
    elif isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        payload = dict(parsed) if isinstance(parsed, dict) else {"output": content}
    else:
        payload = {}
    if "tool" not in payload:
        payload["tool"] = name or payload.get("name") or "unknown"
    payload.setdefault("output", None)
    payload.setdefault("error", None)
    return payload


def messages_from_rollout(rollout, tokenizer) -> list[dict]:
    """Recover structured messages from an oracle/GRPO rollout (for fixtures and tests)."""
    prompt = tokenizer.decode(rollout.prompt_ids)
    question = _question_from_prompt(prompt)
    from fdagym.train.rollout import system_prompt

    messages = [
        {"role": "system", "content": system_prompt("v4", 8)},
        {"role": "user", "content": question},
    ]
    for span in rollout.spans:
        if span.role == "model":
            messages.append({"role": "assistant", "content": span.text})
        elif span.role == "tool":
            payload = _tool_payload(span.text)
            obs = json.loads(payload)
            messages.append({"role": "tool", "name": obs.get("tool") or "unknown", "content": payload})
    return messages


def _question_from_prompt(prompt: str) -> str:
    marker = "<|im_start|>user\n"
    start = prompt.rfind(marker)
    if start < 0:
        raise ValueError("prompt has no user turn")
    body = prompt[start + len(marker) :]
    end = body.find("<|im_end|>")
    if end < 0:
        raise ValueError("user turn is not closed")
    return body[:end]


def _tool_payload(text: str) -> str:
    start = text.find(TOOL_MARK)
    end = text.find("</tool_response>")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("tool span is missing <tool_response> wrappers")
    return text[start + len(TOOL_MARK) : end].strip()


def encode_messages(
    messages: list[dict] | tuple[dict, ...],
    tokenizer,
    codec=None,
    limits: RolloutLimits | None = None,
) -> SimpleNamespace:
    """Prompt ids plus completion ids/loss_mask in the same layout as a ``Rollout``."""
    codec = codec or QwenToolCodec()
    limits = limits or RolloutLimits(temperature=0.0)
    question = user_question(messages)
    prompt = codec.prompt(question, tool_budget=limits.tool_budget, prompt_version=limits.prompt_version)
    prompt_ids = list(tokenizer.encode(prompt))
    completion_ids: list[int] = []
    loss_mask: list[int] = []
    started = False
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user" and not started:
            started = True
            continue
        if role == _ASSISTANT:
            text = assistant_text(message, codec)
            token_ids = list(tokenizer.encode(text))
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                token_ids.append(eos)
            if not token_ids:
                raise ValueError("assistant turn encoded to no tokens")
            completion_ids.extend(token_ids)
            loss_mask.extend([1] * len(token_ids))
            continue
        if role in _TOOL_ROLES or (role == "user" and isinstance(message.get("content"), str) and TOOL_MARK in message["content"]):
            if role == "user":
                obs_text = message["content"]
                start = obs_text.find(TOOL_MARK)
                end = obs_text.find("</tool_response>")
                payload = obs_text[start + len(TOOL_MARK) : end].strip() if start >= 0 and end > start else obs_text
                try:
                    obs = json.loads(payload)
                except json.JSONDecodeError:
                    obs = {"tool": "unknown", "output": payload, "error": None}
            else:
                obs = tool_observation(message)
            text = codec.tool_message(obs, max_chars=limits.max_tool_chars)
            token_ids = list(tokenizer.encode(text))
            completion_ids.extend(token_ids)
            loss_mask.extend([0] * len(token_ids))
            continue
        raise ValueError(f"unsupported message role {role!r}")
    if not any(loss_mask):
        raise ValueError("no assistant tokens to supervise")
    return SimpleNamespace(
        prompt_ids=tuple(prompt_ids),
        completion_ids=tuple(completion_ids),
        loss_mask=tuple(loss_mask),
        task_id="",
    )


def row_from_messages(
    messages: list[dict] | tuple[dict, ...],
    tokenizer,
    codec=None,
    limits: RolloutLimits | None = None,
    *,
    task_id: str = "",
) -> dict:
    encoded = encode_messages(messages, tokenizer, codec, limits)
    encoded.task_id = task_id
    row = sft_example(encoded)
    row["task_id"] = task_id
    return row


def build_distill_rows(
    trajectories,
    tokenizer,
    codec=None,
    limits: RolloutLimits | None = None,
    *,
    max_length: int = 8192,
) -> dict:
    """Tokenize kept teacher rows. Skip empty/over-length conversations."""
    codec = codec or QwenToolCodec(
        prompt_version=(limits.prompt_version if limits is not None else "v4"),
        tool_budget=(limits.tool_budget if limits is not None else 8),
    )
    limits = limits or RolloutLimits(temperature=0.0)
    rows = []
    skipped = []
    encoded_rows = []
    for trajectory in trajectories:
        task_id = getattr(trajectory, "question_id", None) or trajectory.get("question_id")
        messages = getattr(trajectory, "messages", None) or trajectory.get("messages")
        try:
            encoded = encode_messages(messages, tokenizer, codec, limits)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"id": getattr(trajectory, "id", task_id), "reason": "messages", "detail": str(exc)})
            continue
        length = len(encoded.prompt_ids) + len(encoded.completion_ids)
        if length > max_length:
            skipped.append({"id": getattr(trajectory, "id", task_id), "reason": "length", "length": length})
            continue
        encoded.task_id = task_id
        row = sft_example(encoded)
        row["task_id"] = task_id
        row["trajectory_id"] = getattr(trajectory, "id", "")
        rows.append(row)
        encoded_rows.append(encoded)
    return {
        "rows": rows,
        "encoded": encoded_rows,
        "skipped": skipped,
        "n_kept": len(rows),
        "n_skipped": len(skipped),
    }


def assert_assistant_only_labels(row: dict, encoded) -> None:
    """Prompt and tool tokens are ``IGNORE_INDEX``; assistant tokens equal the ids."""
    prompt_n = len(encoded.prompt_ids)
    if row["labels"][:prompt_n] != [IGNORE_INDEX] * prompt_n:
        raise AssertionError("prompt tokens are not masked")
    completion = row["labels"][prompt_n:]
    supervised = 0
    for token_id, bit, label in zip(encoded.completion_ids, encoded.loss_mask, completion, strict=True):
        if bit:
            if label != token_id:
                raise AssertionError("assistant label does not match token id")
            supervised += 1
        elif label != IGNORE_INDEX:
            raise AssertionError("non-assistant token is not masked")
    if supervised != sum(encoded.loss_mask):
        raise AssertionError("supervised count does not match the loss mask")
    if supervised < 1:
        raise AssertionError("no assistant tokens")


def observation_json(observation: dict) -> str:
    """Same payload ``QwenToolCodec.tool_message`` wraps."""
    return observation_payload(observation)
