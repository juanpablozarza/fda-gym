"""Answer-independent label trajectory for boxed_warning and composite tasks.

The stored ``oracle_calls`` add ``has_section=boxed_warning`` only when the label
has that section. Replaying those calls teaches the model to guess before it
searches. This module builds the label steps from the search result instead:

1. ``search_drug_label`` with the generic name only (no section filter).
2. Keep labels of this drug (single ingredient, and the product type the question
   names). A combination on the same page is not this drug.
3. ``get_label_section`` for ``boxed_warning`` only when one of those labels lists
   that section.

The decision reads the question and the records the search returned. It does not
read the answer.
"""

from __future__ import annotations

from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.snapshot import Snapshot
from fdagym.tasks.generators import _label_is_single, _product_type
from fdagym.tools import PAGE_SIZE

EVIDENCE_TYPES = frozenset({"boxed_warning", "composite"})
LABEL_TOOLS = frozenset({"search_drug_label", "get_label_section"})
_QUESTION_SCOPE = {
    "prescription": "HUMAN PRESCRIPTION DRUG",
    "over-the-counter": "HUMAN OTC DRUG",
}


class ScriptError(RuntimeError):
    """The evidence trajectory could not be played through the env."""


def evidence_oracle(task: dict) -> bool:
    return task.get("type") in EVIDENCE_TYPES


def question_scope(question: str) -> str | None:
    """Product type named in the question, if it names one."""
    for phrase, product_type in _QUESTION_SCOPE.items():
        if f"the {phrase} FDA label" in question:
            return product_type
    return None


def tool_actions(snapshot: Snapshot, tasks: list[dict], task: dict) -> tuple[list[dict], FDAEnv]:
    """Tool calls before submit, and the env those calls were played on.

    Non-label calls (the FAERS count on a composite task) are copied from
    ``oracle_calls``. Label calls are rebuilt from the search result.
    """
    env = FDAEnv(snapshot, tasks, run_id="oracle-script")
    env.reset(task["id"])
    actions: list[dict] = []
    for call in task.get("oracle_calls") or []:
        if call.get("tool") in LABEL_TOOLS:
            continue
        actions.append(_play(env, call["tool"], dict(call["arguments"])))
    actions.extend(_label_actions(env, task))
    return actions, env


def _label_actions(env: FDAEnv, task: dict) -> list[dict]:
    drug = (task.get("entity") or {}).get("generic_name")
    if not isinstance(drug, str) or not drug:
        raise ScriptError("task has no generic name")
    scope = question_scope(str(task.get("question") or ""))
    actions: list[dict] = []
    page = 1
    chosen: str | None = None
    saw = False
    while True:
        arguments = {"generic_name": drug}
        if page > 1:
            arguments["page"] = page
        _obs, output = _observe(env, "search_drug_label", arguments)
        actions.append({"tool": "search_drug_label", "arguments": arguments})
        for row in output.get("results") or []:
            if not isinstance(row, dict):
                continue
            set_id = row.get("set_id")
            record = env.state.labels.get(set_id)
            if not isinstance(set_id, str) or not isinstance(record, dict):
                continue
            if not _matches(record, drug, scope):
                continue
            saw = True
            sections = row.get("sections") or []
            if chosen is None and "boxed_warning" in sections:
                chosen = set_id
        total = ((output.get("meta") or {}).get("results") or {}).get("total")
        if saw or not isinstance(total, int) or page * PAGE_SIZE >= total:
            break
        page += 1
    if not saw:
        raise ScriptError(f"no label for {drug} was on the searched pages")
    if chosen is not None:
        arguments = {"set_id": chosen, "section": "boxed_warning"}
        _observe(env, "get_label_section", arguments)
        actions.append({"tool": "get_label_section", "arguments": arguments})
    return actions


def _matches(record: dict, drug: str, scope: str | None) -> bool:
    if not _label_is_single(record, drug):
        return False
    return scope is None or _product_type(record) == scope


def _observe(env: FDAEnv, tool: str, arguments: dict) -> tuple[dict, dict]:
    if env.tool_calls >= TOOL_BUDGET:
        raise ScriptError("tool budget exhausted")
    obs, _reward, done, _info = env.step({"tool": tool, "arguments": arguments})
    if done or obs.get("error"):
        raise ScriptError(obs.get("error") or f"{tool} ended the episode")
    output = obs.get("output")
    if not isinstance(output, dict):
        raise ScriptError(f"{tool} returned no record")
    return obs, output


def _play(env: FDAEnv, tool: str, arguments: dict) -> dict:
    _observe(env, tool, arguments)
    return {"tool": tool, "arguments": arguments}
