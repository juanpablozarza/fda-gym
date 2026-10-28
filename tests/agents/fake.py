"""Scripted LiteLLM stand-in: one OpenAI-shaped tool_call per round."""

import json
from types import SimpleNamespace


def scripted_response(name: str, arguments: dict, *, thought: str | None = None, cost: float = 0.0001):
    call = SimpleNamespace(
        id=f"call_{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    message = SimpleNamespace(content=thought, tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        _hidden_params={"response_cost": cost},
    )


class ScriptedCompletion:
    def __init__(self, calls: list[dict], *, cost: float = 0.0001):
        self.calls = list(calls)
        self.cost = cost
        self.n = 0
        self.models: list[str] = []
        self.kwargs: list[dict] = []

    def __call__(self, **kwargs):
        self.models.append(kwargs["model"])
        self.kwargs.append(kwargs)
        if self.n >= len(self.calls):
            raise AssertionError(f"extra LLM call after {len(self.calls)} scripted tools")
        call = self.calls[self.n]
        self.n += 1
        return scripted_response(call["tool"], call["arguments"], thought=call.get("thought"), cost=self.cost)
