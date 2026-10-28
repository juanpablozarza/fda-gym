"""Multi-turn LiteLLM tool loop over FDAEnv. Six tools, then a forced submit_answer."""

import json
import time

from fdagym.env import FDAEnv
from fdagym.tools import TOOLS

TEMPERATURE = 0.2
FORCED_SUBMIT = {"answer": "NOT_FOUND", "citations": []}

SYSTEM_PROMPT = """\
You are an FDA-Gym agent. Answer only from tool outputs returned in THIS episode.
Do not use memorized FDA facts. If a tool result does not contain the answer, call another tool or submit NOT_FOUND.

Call submit_answer when you are done. Its arguments must match schemas/answer.schema.json:
  {"answer": <string|number|boolean>, "citations": [{"endpoint": "...", "id": "...", "field": "...", "quote": "..."}]}
- endpoint is one of drug/label, drug/event, drug/drugsfda.
- id is a set_id, application_number, or "step:<n>" for FAERS aggregates and empty searches (n is the tool-call index that returned them).
- field is a path the tool actually returned: boxed_warning, sections, submissions, products, sponsor_name, results, meta.results.total, ...
- search_approvals accepts application_number (NDA020977, NDA 020977, 020977, or a BLA/ANDA number) as well as generic_name. Use the application number when the question names one.
- search_approvals returns one page (page starts at 1). NDA and BLA rows come before ANDA, and submissions list only the original approval.
- quote is copied from that field in THIS episode's tool output (verbatim span, not a paraphrase). Quotes shorter than 4 characters never count.
- MedDRA reaction terms are uppercase with British spelling; copy the term from results, do not translate it.
- A missing boxed warning is cited from search_drug_label: id = the label's set_id, field = sections.
- NOT_FOUND is only correct when the snapshot has no record, and only after you searched.
- You get at most 6 tool calls, then you must submit_answer.

Worked citation example (warfarin boxed warning):
1. search_drug_label({"generic_name": "warfarin", "has_section": "boxed_warning"})
   returns set_id 0cbce382-9c88-4f58-ae0f-532a841e8f95.
2. get_label_section({"set_id": "0cbce382-9c88-4f58-ae0f-532a841e8f95", "section": "boxed_warning"})
   returns text containing "Warfarin sodium can cause major or fatal bleeding".
3. submit_answer({
     "answer": "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
     "citations": [{
       "endpoint": "drug/label",
       "id": "0cbce382-9c88-4f58-ae0f-532a841e8f95",
       "field": "boxed_warning",
       "quote": "Warfarin sodium can cause major or fatal bleeding"
     }]
   })
"""

_SUBMIT = next(tool for tool in TOOLS if tool["function"]["name"] == "submit_answer")


def _completion(*args, **kwargs):
    from litellm import completion

    return completion(*args, **kwargs)


class LiteLLMAgent:
    def __init__(
        self,
        model: str,
        *,
        completion=None,
        temperature: float = TEMPERATURE,
        kind: str = "baseline",
        system_prompt: str | None = None,
    ):
        self.model = model
        self.completion = completion
        self.temperature = temperature
        self.kind = kind
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt

    def run(self, env: FDAEnv, task_id: str) -> dict:
        return run_episode(
            env,
            task_id,
            model=self.model,
            completion=self.completion,
            temperature=self.temperature,
            kind=self.kind,
            system_prompt=self.system_prompt,
        )


def run_episode(
    env: FDAEnv,
    task_id: str,
    *,
    model: str,
    completion=None,
    temperature: float = TEMPERATURE,
    kind: str = "baseline",
    system_prompt: str | None = None,
) -> dict:
    complete = completion or _completion
    prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    prompt = prompt.replace(
        "- You get at most 6 tool calls, then you must submit_answer.",
        f"- You get at most {env.tool_budget} tool calls, then you must submit_answer.",
    )
    env.model = {"name": model, "kind": kind, "checkpoint_step": None}
    env.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "latency_ms": 0}
    obs = env.reset(task_id)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": obs["question"]},
    ]
    for _round in range(env.tool_budget + 1):
        force_submit = env.tool_calls >= env.tool_budget
        tools = [_SUBMIT] if force_submit else TOOLS
        started = time.perf_counter()
        response = _call_llm(
            complete,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            force_submit=force_submit,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        _accumulate_usage(env, response, latency_ms)
        llm_error = getattr(response, "_fdagym_error", None)
        if llm_error:
            _obs, _reward, _done, info = env.step({"tool": "submit_answer", "arguments": {}})
            env.steps[-1]["error"] = llm_error
            return info["trajectory"]
        thought = _text(_get(_choice_message(response), "content"))
        actions = _actions_from_response(response, force_submit=force_submit)
        actions.sort(key=lambda row: row["tool"] == "submit_answer")
        assistant_tool_calls = []
        tool_messages = []
        done = False
        info = {}
        for index, action in enumerate(actions):
            call_id = action["_call_id"]
            payload = {"tool": action["tool"], "arguments": action["arguments"], "thought": thought if index == 0 else None}
            assistant_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": action["tool"], "arguments": json.dumps(action["arguments"])},
                }
            )
            obs, _reward, done, info = env.step(payload)
            env.steps[-1]["latency_ms"] = 0 if index else round(latency_ms)
            tool_messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": json.dumps(obs.get("output"), default=str)}
            )
            if done:
                break
        messages.append({"role": "assistant", "content": thought, "tool_calls": assistant_tool_calls})
        messages.extend(tool_messages)
        if done:
            return info["trajectory"]
    raise RuntimeError(f"{task_id}: episode did not finish after {env.tool_budget + 1} LLM rounds")


def _call_llm(complete, *, model, messages, tools, temperature, force_submit):
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": 2048,
        "timeout": 40,
        "drop_params": True,
    }
    from litellm.exceptions import LITELLM_EXCEPTION_TYPES

    llm_errors = tuple(LITELLM_EXCEPTION_TYPES) + (TimeoutError, OSError)
    for attempt in range(2):
        try:
            return complete(**kwargs)
        except llm_errors as exc:
            name = type(exc).__name__
            text = str(exc)
            retryable = any(token in text for token in ("429", "500", "502", "503", "504", "timeout", "Timeout")) or "RateLimit" in name
            if retryable and attempt == 0:
                time.sleep(15)
                continue
            print(f"llm {name}: {text}; recording episode error", flush=True)
            return _empty_response(error=f"{name}: {text}")


def _empty_response(error: str | None = None):
    from types import SimpleNamespace

    message = SimpleNamespace(content=None, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        _hidden_params={"response_cost": 0.0},
        _fdagym_error=error,
    )


def _actions_from_response(response, *, force_submit: bool) -> list[dict]:
    message = _choice_message(response)
    actions = []
    for index, call in enumerate(_get(message, "tool_calls") or []):
        name = _get(_get(call, "function"), "name")
        arguments = _parse_arguments(_get(_get(call, "function"), "arguments"))
        call_id = _get(call, "id") or f"call_{index}"
        if not isinstance(name, str) or not name:
            continue
        actions.append({"tool": name, "arguments": arguments, "_call_id": call_id})
    if not actions:
        parsed = _submit_from_text(_text(_get(message, "content")))
        if parsed is not None:
            actions.append(parsed)
    if force_submit:
        actions = [row for row in actions if row["tool"] == "submit_answer"]
        if not actions:
            actions.append(
                {"tool": "submit_answer", "arguments": dict(FORCED_SUBMIT), "_call_id": "call_forced", "thought": None}
            )
    if not actions:
        actions.append(
            {"tool": "submit_answer", "arguments": dict(FORCED_SUBMIT), "_call_id": "call_forced"}
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
        return {
            "tool": "submit_answer",
            "arguments": {"answer": data["answer"], "citations": data["citations"]},
            "_call_id": "call_text",
        }
    if data.get("tool") == "submit_answer" and isinstance(data.get("arguments"), dict):
        return {"tool": "submit_answer", "arguments": data["arguments"], "_call_id": "call_text"}
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


def _choice_message(response):
    choices = _get(response, "choices") or []
    if not choices:
        return {}
    return _get(choices[0], "message") or {}


def _accumulate_usage(env: FDAEnv, response, latency_ms: float) -> None:
    usage = _get(response, "usage")
    prompt = int(_get(usage, "prompt_tokens") or 0)
    completion = int(_get(usage, "completion_tokens") or 0)
    cost = _cost_of(response)
    env.usage["prompt_tokens"] += prompt
    env.usage["completion_tokens"] += completion
    env.usage["latency_ms"] = round(env.usage["latency_ms"] + latency_ms)
    if cost is not None:
        env.usage["cost_usd"] = round((env.usage["cost_usd"] or 0.0) + float(cost), 6)


def _cost_of(response):
    hidden = _get(response, "_hidden_params") or {}
    if isinstance(hidden, dict) and hidden.get("response_cost") is not None:
        return hidden["response_cost"]
    usage = _get(response, "usage")
    for key in ("cost", "total_cost"):
        value = _get(usage, key)
        if value is not None:
            return value
    return None


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
