"""Pull a submit_answer payload out of a model completion."""

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict) and message.get("content") is not None:
                parts.append(str(message["content"]))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    if isinstance(completion, dict) and completion.get("content") is not None:
        return str(completion["content"])
    return str(completion)


def parse_answer(completion) -> dict | None:
    """Return `{answer, citations}` if a JSON object with those keys is in the text, else None."""
    text = completion_text(completion).strip()
    if not text:
        return None
    fenced = _FENCE.search(text)
    if fenced:
        parsed = _object(fenced.group(1))
        if parsed is not None:
            return parsed
    parsed = _object(text)
    if parsed is not None:
        return parsed
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if _is_answer(obj):
            return obj
    return None


def _object(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if _is_answer(obj) else None


def _is_answer(obj) -> bool:
    return isinstance(obj, dict) and "answer" in obj and "citations" in obj
