"""Memorization measurements for a hard-style RL run.

Measurement only. This module does not score rewards, draw the training pool,
or stop the trainer. ``eval_tool_tasks`` and the GRPO curve callback both call
``memorization_report``.

Answered without looking
------------------------
A submission is correct when ``trajectory["reward"]["correctness"] > 0``.

Tool outputs are ``step["output"]`` on steps whose tool is not ``submit_answer``
and whose output is a dict. The question and the submitted answer are not tool
outputs. Swapped-fact episodes are not part of this rate: their outputs were
rewritten, and the reward still uses the original gold.

The episode looked when any tool output meets either clause. Both clauses read
hidden ``gold`` and ``gold_citations`` (and ``gold["aliases"]``):

1. Record. A non-empty citation id equals some string in the output tree,
   compared casefold. That is the record holding the gold (``set_id``,
   ``application_number``, or any other cited id).

2. Gold value.
   - ``number``: some output has ``meta.results.total`` equal to that integer.
     Years, page indexes, and reaction counts do not count.
   - ``boolean``: this clause does not fire. A yes/no gold counts as looked
     only through clause 1 (the label ``set_id``). The strings ``no``, ``yes``,
     ``true``, and ``false`` do not count.
   - ``exact``: the gold string, or an alias, is contained in the output.
     Strings of 4 characters or more match as a casefolded substring of the
     JSON. Shorter strings match only as a whole string value in the tree.

3. FAERS aggregate. ``gold_citations`` for a count or a reaction list use
   ``ids: []`` because the payload has no record id of its own. That record is
   the ``query_adverse_events`` output whose ``generic_name`` argument equals
   hidden ``entity.generic_name`` and whose body has the cited field
   (``meta.results.total`` is an int, or ``results`` is a list). A count for a
   different drug is not that record. The drug name is often absent from the
   payload, which is why the argument and the entity are read together.

The rate is (correct and not looked) / correct, pooled over the known-drug
holdout and the unseen hard eval. No correct submissions makes the rate null,
and a null rate does not warn.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from fdagym.tasks.hard_solver import same_answer

Measure = Callable[[int], dict | None]


@dataclass(frozen=True)
class MemorizationThresholds:
    """Defaults the training callback and the eval summary share.

    ``level`` is ``stop and review`` when the strict gap and the no-look rate
    both warn. Nothing in this module sets ``should_training_stop``.
    """

    every_n_steps: int = 10
    gap_warn: float = 0.15
    no_look_warn: float = 0.20
    follow_warn: float = 0.70

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        for name in ("gap_warn", "no_look_warn", "follow_warn"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")


def memorization_report(
    *,
    known: list[tuple[dict, dict]],
    unseen: list[tuple[dict, dict]],
    swapped: list[tuple[dict, dict, object]] | None = None,
    thresholds: MemorizationThresholds | None = None,
) -> dict:
    """Strict, pass, no-look, tool calls, and swapped-fact follow rates.

    ``known`` is the known-drug holdout. ``unseen`` is the hard eval.
    ``swapped`` rows are ``(task, trajectory, swapped_gold)``.
    Gap is known minus unseen. ``auto_stop`` is always false.
    """
    limits = thresholds or MemorizationThresholds()
    known_stats = _group(known)
    unseen_stats = _group(unseen)
    gap_strict = _subtract(known_stats["strict"], unseen_stats["strict"])
    gap_pass = _subtract(known_stats["pass"], unseen_stats["pass"])
    no_look = _no_look([*known, *unseen])
    no_look_known = _no_look(known)
    no_look_unseen = _no_look(unseen)
    swapped_stats = _swapped(swapped or [])
    gap_flag = "warn" if gap_strict is not None and gap_strict > limits.gap_warn else "ok"
    no_look_flag = "warn" if no_look["rate"] is not None and no_look["rate"] > limits.no_look_warn else "ok"
    follow_flag = (
        "warn"
        if swapped_stats["follow_rate"] is not None and swapped_stats["follow_rate"] < limits.follow_warn
        else "ok"
    )
    if gap_flag == "warn" and no_look_flag == "warn":
        level = "stop and review"
    elif gap_flag == "warn" or no_look_flag == "warn" or follow_flag == "warn":
        level = "warn"
    else:
        level = "ok"
    return {
        "known": known_stats,
        "unseen": unseen_stats,
        "gap": {"strict": gap_strict, "pass": gap_pass},
        "answered_without_looking": {
            "rate": no_look["rate"],
            "correct": no_look["correct"],
            "without_looking": no_look["without_looking"],
            "known": no_look_known,
            "unseen": no_look_unseen,
            "rule": "gold value or gold_citation id in a tool output; see fdagym.memorization",
        },
        "mean_tool_calls": {"known": known_stats["mean_tool_calls"], "unseen": unseen_stats["mean_tool_calls"]},
        "swapped_facts": swapped_stats,
        "flags": {
            "gap": gap_flag,
            "answered_without_looking": no_look_flag,
            "swapped_facts": follow_flag,
            "level": level,
        },
        "auto_stop": False,
        "thresholds": {
            "every_n_steps": limits.every_n_steps,
            "gap_warn": limits.gap_warn,
            "no_look_warn": limits.no_look_warn,
            "follow_warn": limits.follow_warn,
        },
    }


def looked(task: dict, trajectory: dict) -> bool:
    """True when a tool output contained the gold value or the record holding it."""
    gold = (task.get("gold") or {})
    value = gold.get("value")
    kind = gold.get("kind")
    aliases = [alias for alias in gold.get("aliases") or [] if isinstance(alias, str)]
    ids = _citation_ids(task)
    for step in trajectory.get("steps") or []:
        if step.get("tool") == "submit_answer":
            continue
        output = step.get("output")
        if not isinstance(output, dict):
            continue
        if ids and _has_id(output, ids):
            return True
        if kind == "number" and isinstance(value, int) and not isinstance(value, bool) and _total_equals(output, value):
            return True
        if kind == "exact" and isinstance(value, str) and (
            _contains_text(output, value) or any(_contains_text(output, alias) for alias in aliases)
        ):
            return True
        if _faers_record(task, step, output):
            return True
    return False


def attach_memorization(summary: dict, report: dict) -> dict:
    """Copy ``summary`` and set ``memorization``. Existing keys are not rewritten."""
    attached = dict(summary)
    attached["memorization"] = report
    return attached


def record_training_memorization(row: dict | None, step: int, measure: Measure | None, *, every: int = 10) -> dict | None:
    """Attach a report on steps 10, 20, 30, ... Do not stop training."""
    if row is None or measure is None or every < 1 or step < 1 or step % every != 0:
        return row
    report = measure(step)
    if report is not None:
        row["memorization"] = report
    return row


def _group(episodes: list[tuple[dict, dict]]) -> dict:
    n = len(episodes)
    if n == 0:
        return {"n": 0, "strict": None, "pass": None, "mean_tool_calls": None}
    strict = sum(_reward(trajectory)["total"] for _task, trajectory in episodes) / n
    passed = sum(1 for _task, trajectory in episodes if _reward(trajectory)["correctness"] > 0) / n
    calls = sum(_tool_calls(trajectory) for _task, trajectory in episodes) / n
    return {"n": n, "strict": strict, "pass": passed, "mean_tool_calls": calls}


def _no_look(episodes: list[tuple[dict, dict]]) -> dict:
    correct = [(task, trajectory) for task, trajectory in episodes if _reward(trajectory)["correctness"] > 0]
    without = sum(1 for task, trajectory in correct if not looked(task, trajectory))
    rate = (without / len(correct)) if correct else None
    return {"correct": len(correct), "without_looking": without, "rate": rate}


def _swapped(episodes: list[tuple[dict, dict, object]]) -> dict:
    n = len(episodes)
    follow = 0
    original = 0
    other = 0
    for task, trajectory, swapped_gold in episodes:
        answer = _submitted(trajectory)
        if same_answer(answer, swapped_gold):
            follow += 1
        elif same_answer(answer, (task.get("gold") or {}).get("value")):
            original += 1
        else:
            other += 1
    return {
        "n": n,
        "follow": follow,
        "follow_rate": (follow / n) if n else None,
        "original": original,
        "original_rate": (original / n) if n else None,
        "other": other,
    }


def _subtract(left, right):
    if left is None or right is None:
        return None
    return left - right


def _reward(trajectory: dict) -> dict:
    reward = trajectory.get("reward") or {}
    return {"total": float(reward.get("total") or 0.0), "correctness": float(reward.get("correctness") or 0.0)}


def _tool_calls(trajectory: dict) -> int:
    return sum(1 for step in trajectory.get("steps") or [] if step.get("tool") != "submit_answer")


def _submitted(trajectory: dict):
    final = trajectory.get("final_answer") or {}
    if isinstance(final, dict) and "answer" in final:
        return final["answer"]
    return None


def _citation_ids(task: dict) -> set[str]:
    found: set[str] = set()
    for cite in task.get("gold_citations") or []:
        for cid in cite.get("ids") or []:
            if isinstance(cid, str) and cid.strip():
                found.add(cid.strip().casefold())
    return found


def _faers_record(task: dict, step: dict, output: dict) -> bool:
    """Empty-id FAERS citation: the payload for ``entity.generic_name`` is the record."""
    entity = str((task.get("entity") or {}).get("generic_name") or "").strip().casefold()
    asked = str((step.get("arguments") or {}).get("generic_name") or "").strip().casefold()
    if not entity or asked != entity or step.get("tool") != "query_adverse_events":
        return False
    fields: list[str] = []
    for cite in task.get("gold_citations") or []:
        if cite.get("ids"):
            continue
        fields.extend(field for field in cite.get("fields") or [] if isinstance(field, str))
    if "meta.results.total" in fields and _total_is_int(output):
        return True
    return "results" in fields and isinstance(output.get("results"), list)


def _total_is_int(output: dict) -> bool:
    meta = output.get("meta")
    if not isinstance(meta, dict):
        return False
    results = meta.get("results")
    if not isinstance(results, dict):
        return False
    total = results.get("total")
    return isinstance(total, int) and not isinstance(total, bool)


def _has_id(output: dict, ids: set[str]) -> bool:
    return any(isinstance(value, str) and value.strip().casefold() in ids for value in _strings(output))


def _total_equals(output: dict, gold: int) -> bool:
    meta = output.get("meta")
    if not isinstance(meta, dict):
        return False
    results = meta.get("results")
    if not isinstance(results, dict):
        return False
    return results.get("total") == gold


def _contains_text(output: dict, text: str) -> bool:
    token = text.strip()
    if not token:
        return False
    if len(token) >= 4:
        return token.casefold() in json.dumps(output, ensure_ascii=False).casefold()
    return any(value.casefold() == token.casefold() for value in _strings(output))


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
