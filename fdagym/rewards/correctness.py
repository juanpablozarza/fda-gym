"""Correctness reward for exact, number, boolean, and judge gold.

A right answer with no valid citation scores 0. Invalid format scores 0.
NOT_FOUND scores 1 only on answerable:false tasks that have a grounded empty-result citation.
"""

import re
from typing import Protocol

from fdagym.rewards.topics import MAX_TOPICS, topics_named

_OR = re.compile(r"\bor\b", re.IGNORECASE)
_YES = re.compile(r"^yes\b", re.IGNORECASE)
_NO = re.compile(r"^no\b", re.IGNORECASE)


class Judge(Protocol):
    def score(self, *, question: str, answer: str, gold: dict) -> float:
        """Return a score in [0, 1] for an open answer."""


class UnimplementedJudge:
    def score(self, *, question: str, answer: str, gold: dict) -> float:
        raise NotImplementedError("LLM judge is not configured")


def score_correctness(
    answer: object,
    task: dict,
    *,
    valid_citations: int,
    format_ok: bool,
    empty_grounded: bool,
    judge: Judge | None = None,
) -> tuple[float, list[str]]:
    if not format_ok:
        return 0.0, ["correctness forced to 0: invalid format"]
    # Break-it: change `< 1` to `< 0`. An ungrounded guess then keeps its raw score.
    if valid_citations < 1:
        return 0.0, ["correctness gated: no valid citation"]
    return _match(answer, task, empty_grounded=empty_grounded, judge=judge or UnimplementedJudge())


def _match(answer: object, task: dict, *, empty_grounded: bool, judge: Judge) -> tuple[float, list[str]]:
    if _is_not_found(answer):
        if task["answerable"]:
            return 0.0, ["NOT_FOUND on an answerable task"]
        if not empty_grounded:
            return 0.0, ["NOT_FOUND without a grounded empty-result citation"]
        return 1.0, ["abstained after a grounded empty result"]

    gold = task["gold"]
    kind = gold["kind"]
    if kind == "exact":
        return _exact(answer, gold)
    if kind == "number":
        return _number(answer, gold)
    if kind == "boolean":
        return _boolean(answer, gold)
    if kind == "judge":
        return _judge(answer, task, judge)
    return 0.0, [f"unknown gold kind {kind}"]


def _is_not_found(answer: object) -> bool:
    return isinstance(answer, str) and answer.strip().lower() == "not_found"


def _norm(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text).strip(" .")


def _exact(answer: object, gold: dict) -> tuple[float, list[str]]:
    if isinstance(answer, str) and _OR.search(answer):
        return 0.0, ["exact: answer lists alternatives; gold is a single value"]
    wanted = {_norm(gold["value"]), *(_norm(alias) for alias in gold.get("aliases") or [])}
    if _norm(answer) in wanted:
        return 1.0, ["exact: normalized match"]
    return 0.0, [f"exact: {_norm(answer)!r} matches none of {sorted(wanted)}"]


def _number(answer: object, gold: dict) -> tuple[float, list[str]]:
    pred = _as_number(answer)
    target = _as_number(gold["value"])
    if pred is None or target is None:
        return 0.0, ["number: answer is not a number"]
    tolerance = float(gold.get("tolerance") or 0)
    if target == 0:
        score = 1.0 if pred == 0 else 0.0
        return score, [f"number: pred {pred:g} vs gold 0"]
    rel = abs(pred - target) / abs(target)
    if rel <= 1e-12:
        return 1.0, [f"number: exact {pred:g}"]
    if tolerance <= 0 or rel > tolerance:
        return 0.0, [f"number: relative error {rel:.4f} outside tolerance {tolerance:g}"]
    score = 1.0 - rel / tolerance
    return score, [f"number: relative error {rel:.4f} within {tolerance:g}, score {score:.4f}"]


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _boolean(answer: object, gold: dict) -> tuple[float, list[str]]:
    keywords = [str(k) for k in gold.get("keywords") or []]
    value = bool(gold["value"])
    if isinstance(answer, bool):
        if keywords:
            return 0.0, ["boolean: keywords require a yes/no sentence"]
        ok = answer is value
        return (1.0 if ok else 0.0), [f"boolean: {answer} vs {value}"]
    if not isinstance(answer, str):
        return 0.0, ["boolean: answer is not a string"]
    polarity = _YES.search(answer.strip()) if value else _NO.search(answer.strip())
    if polarity is None:
        expect = "yes" if value else "no"
        return 0.0, [f"boolean: answer does not start with {expect}"]
    if keywords and not any(k.lower() in answer.lower() for k in keywords):
        return 0.0, [f"boolean: missing keyword, need one of {keywords}"]
    if keywords:
        named = topics_named(answer)
        if len(named) > MAX_TOPICS:
            return 0.0, [f"correctness: answer names {len(named)} boxed-warning topics (max {MAX_TOPICS})"]
        return 1.0, ["boolean: polarity and keyword match"]
    return 1.0, ["boolean: polarity match"]


def _judge(answer: object, task: dict, judge: Judge) -> tuple[float, list[str]]:
    text = answer if isinstance(answer, str) else str(answer)
    raw = float(judge.score(question=task["question"], answer=text, gold=task["gold"]))
    score = min(1.0, max(0.0, raw))
    return score, [f"judge: {score:.4f}"]
