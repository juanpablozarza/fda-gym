"""Build a submit_answer payload that the oracle would send. Used by dry-run and tests."""

from fdagym.rewards.citation import _TOOL_ENDPOINT, _collapse, _dicts, _render, _resolve


def gold_answer(task: dict):
    if not task["answerable"]:
        return "NOT_FOUND"
    gold = task["gold"]
    if gold["kind"] == "boolean":
        if gold["value"]:
            keywords = gold.get("keywords") or []
            if keywords:
                return f"Yes. {keywords[0]}."
            return "Yes."
        return "No."
    return gold["value"]


def citations_from_outputs(steps: list[dict], task: dict) -> list[dict]:
    """Cite a gold field from a tool output in this episode.

    A single-drug trajectory keeps step order. A compare that calls FAERS once
    per candidate would otherwise cite the first call. Those rows name the
    answer drug on ``entity.generic_name``, so that drug's calls are tried first.
    An empty ``results: []`` on a count call is skipped when a later call has
    the reaction list.
    """
    ordered = _entity_first(steps, task)
    citations = []
    for gold in task["gold_citations"]:
        found = None
        for index, step in enumerate(ordered):
            if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != gold["endpoint"]:
                continue
            for node in _dicts(step.get("output")):
                if not isinstance(node, dict):
                    continue
                cid = _accepted_id(node, gold)
                if cid is None:
                    continue
                for field in gold["fields"]:
                    text = _render(_resolve(node, field))
                    if not text:
                        continue
                    if text == "[]" and _later_field(ordered, index, gold["endpoint"], field):
                        continue
                    found = (gold["endpoint"], cid, field, _quote(text))
                    break
                if found:
                    break
            if found:
                break
        if found is None:
            raise RuntimeError(f"{task['id']} gold field {gold['fields']} never appeared")
        endpoint, cid, field, quote = found
        citations.append({"endpoint": endpoint, "id": cid, "field": field, "quote": quote})
    return citations


def _entity_first(steps: list[dict], task: dict) -> list[dict]:
    drug = ((task.get("entity") or {}).get("generic_name") or "").casefold()
    names = {
        name.casefold()
        for step in steps
        if isinstance(name := (step.get("arguments") or {}).get("generic_name"), str) and name.strip()
    }
    if not drug or len(names) < 2:
        return list(steps)
    matched = [
        step
        for step in steps
        if ((step.get("arguments") or {}).get("generic_name") or "").casefold() == drug
    ]
    matched_ids = {id(step) for step in matched}
    return matched + [step for step in steps if id(step) not in matched_ids]


def _later_field(steps: list[dict], index: int, endpoint: str, field: str) -> bool:
    for step in steps[index + 1 :]:
        if step.get("error") or _TOOL_ENDPOINT.get(step.get("tool")) != endpoint:
            continue
        for node in _dicts(step.get("output")):
            text = _render(_resolve(node, field))
            if text and text != "[]":
                return True
    return False


def gold_submit(task: dict, steps: list[dict]) -> dict:
    return {"answer": gold_answer(task), "citations": citations_from_outputs(steps, task)}


def _quote(text: str) -> str:
    collapsed = _collapse(text)
    if len(collapsed) <= 12 or len(collapsed) < 20:
        return collapsed
    return collapsed[:20]


def _accepted_id(node: dict, gold: dict) -> str | None:
    for key in ("set_id", "application_number", "safetyreportid", "id"):
        value = node.get(key)
        if not isinstance(value, str) or not value:
            continue
        if gold["ids"]:
            if value in gold["ids"]:
                return value
        elif value.startswith("step:"):
            return value
    return None
