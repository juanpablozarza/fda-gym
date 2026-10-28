"""Citation reward: valid / cited, after deduping endpoint + id + field.

A citation is valid when its id showed up in a tool output of this trajectory, its field is one
the task accepts, and its quote fuzzy-matches that field's text at >= 0.9. A quote longer than
the field text does not match. step: ids, and gold entries with an empty id list, must come
from a tool call about the task's drug. On a compare question that drug may be any drug the
question names. A boxed_none question also accepts step:<n> meta.results.total, or results, from a
search_drug_label call with has_section boxed_warning whose total is 0.
"""

import json
import re
from dataclasses import dataclass

from rapidfuzz.fuzz import partial_ratio, ratio

from fdagym.queries import label_search

_ID_KEYS = ("id", "set_id", "application_number", "safetyreportid")
_TOOL_ENDPOINT = {
    "search_drug_label": "drug/label",
    "get_label_section": "drug/label",
    "query_adverse_events": "drug/event",
    "search_approvals": "drug/drugsfda",
}
_FUZZY = 90
_SHORT_FIELD = 12
_MIN_QUOTE = 4


# Share of a citation group earned when the id is a gold id but the field or quote is not.
ID_ONLY = 0.5


@dataclass(frozen=True)
class CitationScore:
    score: float
    valid: int
    total: int
    empty_grounded: bool
    notes: list[str]
    id_hits: int = 0


def score_citations(steps: list[dict], task: dict, citations: list[dict], *, dense: bool = False) -> CitationScore:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for citation in citations:
        key = (citation["endpoint"], citation["id"], citation["field"])
        groups.setdefault(key, []).append(citation)

    notes: list[str] = []
    dropped = sum(len(group) - 1 for group in groups.values())
    if dropped:
        notes.append(f"dropped {dropped} duplicate citations (same endpoint, id, and field)")

    valid = 0
    id_hits = 0
    credit = 0.0
    empty_grounded = False
    for group in groups.values():
        ok, reason, empty = _group_ok(group, steps, task)
        notes.append(reason)
        if ok:
            valid += 1
            id_hits += 1
            credit += 1.0
            empty_grounded = empty_grounded or empty
        elif dense and _gold_id_hit(group[0], steps, task):
            id_hits += 1
            credit += ID_ONLY
            label = f"{group[0]['id']} {group[0]['field']}"
            notes.append(f"partial: {label} id matches a gold citation id; field or quote does not")

    total = len(groups)
    score = (credit / total) if (dense and total) else (valid / total if total else 0.0)
    notes.insert(0, f"citation {valid}/{total} valid" + (f", dense credit {score:.2f}" if dense else ""))
    return CitationScore(score, valid, total, empty_grounded, notes, id_hits)


def _group_ok(group: list[dict], steps: list[dict], task: dict) -> tuple[bool, str, bool]:
    reasons = []
    for citation in group:
        ok, reason, empty = _one(citation, steps, task)
        if ok:
            return True, reason, empty
        reasons.append(reason)
    return False, reasons[0], False


def _one(citation: dict, steps: list[dict], task: dict) -> tuple[bool, str, bool]:
    endpoint, cid, field = citation["endpoint"], citation["id"], citation["field"]
    label = f"{cid} {field}"
    needs_drug = _needs_drug_check(task, endpoint, cid, field)
    if needs_drug is None:
        ok, reason = _empty_boxed_citation(citation, steps, task)
        if ok:
            return True, reason, True
        return False, f"invalid: {label} field or id not accepted by this task", False

    candidates = _candidates(steps, endpoint, cid, field)
    if not candidates:
        if _id_appeared(steps, endpoint, cid):
            return False, f"invalid: {label} field not in the tool output", False
        return False, f"invalid: {label} id did not appear in this trajectory", False

    if needs_drug:
        candidates = [item for item in candidates if _about_task_drug(item[0], steps, task)]
        if not candidates:
            return False, f"invalid: {label} is not about {task['entity']['generic_name']}", False

    quote = citation.get("quote")
    if not isinstance(quote, str):
        return False, f"invalid: {label} quote is not a string", False
    for step, text in candidates:
        matched, detail = _quote_matches(quote, text)
        if matched:
            empty = _is_empty(step.get("output"))
            return True, f"valid: {label} ({detail})", empty
    return False, f"invalid: {label} {_quote_matches(quote, candidates[0][1])[1]}", False


def _gold_id_hit(citation: dict, steps: list[dict], task: dict) -> bool:
    """True when this citation's id is one the task's gold citations name, and it was retrieved."""
    endpoint, cid = citation.get("endpoint"), citation.get("id")
    if not isinstance(endpoint, str) or not isinstance(cid, str):
        return False
    if not _id_appeared(steps, endpoint, cid):
        return False
    for gold in task.get("gold_citations") or []:
        if gold.get("endpoint") != endpoint:
            continue
        ids = gold.get("ids") or []
        if ids:
            if cid in ids:
                return True
            continue
        if not cid.startswith("step:"):
            continue
        for step in steps:
            if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != endpoint:
                continue
            if not any(_node_has_id(node, cid) for node in _dicts(step.get("output"))):
                continue
            if _about_task_drug(step, steps, task):
                return True
    return False


def _needs_drug_check(task: dict, endpoint: str, cid: str, field: str) -> bool | None:
    """None if no gold entry accepts this citation. True if a drug check is also required."""
    matches = [
        gold
        for gold in task["gold_citations"]
        if gold["endpoint"] == endpoint and field in gold["fields"] and (not gold["ids"] or cid in gold["ids"])
    ]
    if not matches:
        return None
    explicit = any(gold["ids"] and cid in gold["ids"] for gold in matches)
    return cid.startswith("step:") or not explicit


def _candidates(steps: list[dict], endpoint: str, cid: str, field: str) -> list[tuple[dict, str]]:
    found = []
    for step in steps:
        if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != endpoint:
            continue
        output = step.get("output")
        for node in _dicts(output):
            if not _node_has_id(node, cid):
                continue
            text = _render(_resolve(node, field))
            if text is not None:
                found.append((step, text))
                break
    return found


def _id_appeared(steps: list[dict], endpoint: str, cid: str) -> bool:
    for step in steps:
        if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != endpoint:
            continue
        if any(_node_has_id(node, cid) for node in _dicts(step.get("output"))):
            return True
    return False


def _about_task_drug(step: dict, steps: list[dict], task: dict) -> bool:
    drug = _drug_of(step, steps)
    if drug is None:
        return False
    return any(label_search(drug) == label_search(name) for name in _topic_drugs(task))


def _topic_drugs(task: dict) -> list[str]:
    """Drugs a step citation may be about. Compare questions name every drug that counts."""
    entity = task["entity"]["generic_name"]
    if task.get("type") != "compare":
        return [entity]
    from fdagym.tasks.hard_solver import Unsolvable, parse_question

    try:
        drugs = parse_question(task["question"]).get("drugs") or []
    except (Unsolvable, KeyError, TypeError, ValueError):
        return [entity]
    return [drug for drug in drugs if isinstance(drug, str) and drug.strip()] or [entity]


def _empty_boxed_citation(citation: dict, steps: list[dict], task: dict) -> tuple[bool, str]:
    """A 0-result boxed-warning search is evidence on a boxed_none question.

    Gold lists label set_ids and field sections. The search that shows a drug has no boxed
    warning is cited as step:<n> meta.results.total or results. The quote may be the leaf
    (0 or []) or a JSON span of the tool output that contains it. The search may be for
    any drug the question names.
    """
    label = f"{citation.get('id')} {citation.get('field')}"
    if task.get("type") != "compare" or citation.get("endpoint") != "drug/label":
        return False, ""
    if citation.get("field") not in {"meta.results.total", "results"} or not str(citation.get("id", "")).startswith("step:"):
        return False, ""
    from fdagym.tasks.hard_solver import Unsolvable, parse_question

    try:
        spec = parse_question(task["question"])
    except (Unsolvable, KeyError, TypeError, ValueError):
        return False, ""
    if spec.get("variant") != "boxed_none":
        return False, ""
    step = _step_with_id(steps, citation["id"])
    if step is None or step.get("tool") != "search_drug_label":
        return False, ""
    arguments = step.get("arguments") or {}
    if arguments.get("has_section") != "boxed_warning":
        return False, ""
    output = step.get("output")
    if not isinstance(output, dict) or _resolve(output, "meta.results.total") != 0:
        return False, ""
    drug = _drug_of(step, steps)
    if drug is None or not any(label_search(drug) == label_search(name) for name in _topic_drugs(task)):
        return False, ""
    quote = citation.get("quote")
    if not isinstance(quote, str) or not _quotes_empty_search(quote, output, citation["field"]):
        return False, ""
    return True, f"valid: {label} (empty boxed_warning search)"


def _step_with_id(steps: list[dict], cid: str) -> dict | None:
    for step in steps:
        if step.get("error"):
            continue
        output = step.get("output")
        if isinstance(output, dict) and output.get("id") == cid:
            return step
    return None


def _quotes_empty_search(quote: str, output: dict, field: str) -> bool:
    """The leaf is `0` or `[]`. Models also quote the JSON span around that leaf."""
    leaf = "0" if field == "meta.results.total" else "[]"
    if _quote_matches(quote, leaf)[0]:
        return True
    span = _collapse(quote)
    blob = _collapse(json.dumps(output, ensure_ascii=False))
    marker = '"total": 0' if field == "meta.results.total" else '"results": []'
    return bool(span) and span in blob and marker in span


def _drug_of(step: dict, steps: list[dict]) -> str | None:
    name = (step.get("arguments") or {}).get("generic_name")
    if isinstance(name, str) and name.strip():
        return name
    set_id = (step.get("arguments") or {}).get("set_id")
    if not isinstance(set_id, str):
        return None
    for prev in steps:
        if prev.get("tool") != "search_drug_label" or prev.get("error"):
            continue
        if _id_appeared([prev], "drug/label", set_id):
            prev_name = (prev.get("arguments") or {}).get("generic_name")
            if isinstance(prev_name, str):
                return prev_name
    return None


def _is_empty(output: object) -> bool:
    return isinstance(output, dict) and _resolve(output, "meta.results.total") == 0


def _quote_matches(quote: str, field_text: str) -> tuple[bool, str]:
    left = _collapse(quote)
    right = _collapse(field_text)
    if not left:
        return False, "empty quote"
    if len(left) > len(right):
        return False, "quote longer than field text"
    if len(right) <= _SHORT_FIELD:
        score = ratio(left, right, processor=str.lower)
    else:
        if len(left) < _MIN_QUOTE:
            return False, "quote shorter than 4 characters"
        score = partial_ratio(left, right, processor=str.lower)
    if score < _FUZZY:
        return False, f"quote fuzzy {score / 100:.2f} < 0.90"
    return True, f"quote fuzzy {score / 100:.2f}"


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _resolve(node: object, path: str) -> object:
    current = node
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _render(value: object) -> str | None:
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


def _dicts(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dicts(child)


def _node_has_id(node: dict, cid: str) -> bool:
    return any(node.get(key) == cid for key in _ID_KEYS)
