"""Quote and field helpers shared by the grader and the question-only solver."""

from __future__ import annotations

import json
import re


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def resolve(node: object, path: str) -> object:
    current = node
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def render(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
