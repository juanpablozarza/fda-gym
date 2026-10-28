"""One FDAEnv episode driven by an OpenAI-style Chat Completions callable."""

import json
import time

from fdagym.env import FDAEnv
from fdagym.tools import TOOLS

PUBLIC_PROMPT = """\
Answer only from tool results in this conversation.
Do not use memorized facts.
Call submit_answer when you are done.
Search before submitting NOT_FOUND.
"""

_SUBMIT = next(tool for tool in TOOLS if tool["function"]["name"] == "submit_answer")
_BUDGET_ERROR = {"error": "tool budget exhausted; call submit_answer"}
_SUBMIT_CHOICE = {"type": "function", "function": {"name": "submit_answer"}}


def run_chat_episode(
    env: FDAEnv,
    task_id: str,
    complete,
    *,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Run one episode and return the FDAEnv trajectory.

    ``complete`` takes the OpenAI Chat Completions keywords ``model``,
    ``messages``, ``tools``, ``tool_choice``, and ``temperature``. The OpenAI
    SDK method matches that shape::

        from openai import OpenAI
        client = OpenAI()
        run_chat_episode(env, task_id, client.chat.completions.create, model="gpt-4.1")

    Any callable with the same keywords works::

        def complete(*, model, messages, tools, tool_choice="auto", temperature=0.0):
            return my_backend(model, messages, tools, tool_choice, temperature)

        run_chat_episode(env, task_id, complete, model="lab-model")
    """
    env.model = {"name": model, "kind": "baseline", "checkpoint_step": None}
    env.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "latency_ms": 0}
    observation = env.reset(task_id)
    messages: list[dict] = [
        {"role": "system", "content": PUBLIC_PROMPT},
        {"role": "user", "content": observation["question"]},
    ]
    for _ in range(env.tool_budget + 1):
        trajectory = _round(env, complete, messages, model=model, temperature=temperature)
        if trajectory is not None:
            return trajectory
    return _submit(env, _forced_arguments(), thought=None, latency_ms=0, stamp_latency=False)


def _round(env: FDAEnv, complete, messages: list[dict], *, model: str, temperature: float) -> dict | None:
    force_submit = env.tool_calls >= env.tool_budget
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": [_SUBMIT] if force_submit else TOOLS,
        "temperature": temperature,
    }
    if force_submit:
        kwargs["tool_choice"] = _SUBMIT_CHOICE
    started = time.perf_counter()
    response = complete(**kwargs)
    latency_ms = (time.perf_counter() - started) * 1000
    _accumulate_usage(env, response, latency_ms)

    message = _choice_message(response)
    content = _text(_get(message, "content"))
    actions = _actions_from_message(message)
    if not actions:
        arguments = _submit_from_text(content) or _forced_arguments()
        return _submit(env, arguments, thought=content, latency_ms=latency_ms, stamp_latency=True)

    assistant_calls = []
    tool_messages = []
    done = False
    info: dict = {}
    stamp_latency = True
    for action in actions:
        assistant_calls.append(
            {
                "id": action["call_id"],
                "type": "function",
                "function": {"name": action["tool"], "arguments": action["arguments_json"]},
            }
        )
        if action["tool"] != "submit_answer" and env.tool_calls >= env.tool_budget:
            tool_messages.append(_tool_message(action["call_id"], _BUDGET_ERROR))
            continue
        thought = content if stamp_latency else None
        observation, done, info = _step(env, action["tool"], action["arguments"], thought, latency_ms, stamp_latency)
        stamp_latency = False
        tool_messages.append(_tool_message(action["call_id"], observation.get("output")))
        if done:
            break
    if not done and force_submit:
        _step(env, "submit_answer", _forced_arguments(), content if stamp_latency else None, latency_ms, stamp_latency)
        done = True
        info = {"trajectory": env.trajectory}
    messages.append({"role": "assistant", "content": content, "tool_calls": assistant_calls})
    messages.extend(tool_messages)
    if done:
        return info["trajectory"]
    return None


def _submit(env: FDAEnv, arguments: dict, *, thought: str | None, latency_ms: float, stamp_latency: bool) -> dict:
    _observation, _done, info = _step(env, "submit_answer", arguments, thought, latency_ms, stamp_latency)
    return info["trajectory"]


def _step(env: FDAEnv, tool: str, arguments: dict, thought: str | None, latency_ms: float, stamp_latency: bool):
    observation, _reward, done, info = env.step({"tool": tool, "arguments": arguments, "thought": thought})
    if stamp_latency:
        env.steps[-1]["latency_ms"] = round(latency_ms)
    return observation, done, info


def _forced_arguments() -> dict:
    return {"answer": "NOT_FOUND", "citations": []}


def _actions_from_message(message) -> list[dict]:
    actions = []
    for index, call in enumerate(_get(message, "tool_calls") or []):
        function = _get(call, "function")
        name = _get(function, "name")
        if not isinstance(name, str) or not name:
            continue
        raw = _get(function, "arguments")
        arguments = _parse_arguments(raw)
        actions.append(
            {
                "tool": name,
                "arguments": arguments,
                "arguments_json": raw if isinstance(raw, str) and raw.strip() else json.dumps(arguments),
                "call_id": _get(call, "id") or f"call_{index}",
            }
        )
    return actions


def _submit_from_text(text: str | None) -> dict | None:
    if not text:
        return None
    blob = text.strip()
    if blob.startswith("```"):
        blob = blob.strip("`")
        blob = blob.removeprefix("json").strip()
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        start, end = blob.find("{"), blob.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(blob[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    if "answer" in data and "citations" in data:
        return {"answer": data["answer"], "citations": data["citations"]}
    nested = data.get("arguments")
    if data.get("tool") == "submit_answer" and isinstance(nested, dict):
        return nested
    return None


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_message(call_id: str, output) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(output, default=str)}


def _choice_message(response):
    choices = _get(response, "choices") or []
    if not choices:
        return {}
    return _get(choices[0], "message") or {}


def _accumulate_usage(env: FDAEnv, response, latency_ms: float) -> None:
    usage = _get(response, "usage")
    if usage is not None:
        env.usage["prompt_tokens"] += int(_get(usage, "prompt_tokens") or 0)
        env.usage["completion_tokens"] += int(_get(usage, "completion_tokens") or 0)
    env.usage["latency_ms"] = round(env.usage["latency_ms"] + latency_ms)


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _text(value) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
