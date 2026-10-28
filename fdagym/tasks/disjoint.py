"""Drugs, application numbers, and sponsors that a hard-training task must not reuse.

A string appears in a task when it is in the question, the gold answer, or the hidden
metadata (entity, oracle arguments, citation ids). A sponsor appears when it is the gold
of a "Who is the sponsor" question, or the snapshot ``sponsor_name`` of an application
cited with that field. A sponsor that only happens to market a named drug, and is never
cited, does not appear.

Training tasks are also charged for every universe drug that owns an application number
written in the task. That stops a sibling application from walking back into a held-out
drug during a tool call.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from fdagym.queries import approvals_search
from fdagym.tasks.hard_solver import Unsolvable, parse_question
from fdagym.tasks.plan import load_universe

_APP = re.compile(r"\b(?:NDA|BLA|ANDA)\d{6}\b")
_CANON = re.compile(r"^(NDA|BLA|ANDA)\d{6}$")


class OverlapError(RuntimeError):
    """Training tasks reuse a drug, application, or sponsor from the held-out splits."""

    def __init__(self, overlaps: dict[str, set[str]]):
        self.overlaps = {key: sorted(values) for key, values in overlaps.items() if values}
        parts = []
        for key, values in self.overlaps.items():
            shown = ", ".join(values[:8])
            extra = f" (+{len(values) - 8} more)" if len(values) > 8 else ""
            parts.append(f"{len(values)} {key}: {shown}{extra}")
        listing = "; ".join(parts) if parts else "none"
        super().__init__(f"hard-train overlaps held-out entities — {listing}")


@dataclass(frozen=True)
class EntitySet:
    drugs: frozenset[str]
    applications: frozenset[str]
    sponsors: frozenset[str]


@dataclass(frozen=True)
class ApprovalMaps:
    """application number -> universe drugs that return it, and -> sponsor_name."""

    owners: dict[str, frozenset[str]]
    sponsors: dict[str, str]


def approval_maps(snapshot) -> ApprovalMaps:
    cached = getattr(snapshot, "_hard_approval_maps", None)
    if isinstance(cached, ApprovalMaps):
        return cached
    owners: dict[str, set[str]] = defaultdict(set)
    sponsors: dict[str, str] = {}
    for drug in load_universe():
        query = approvals_search(drug)
        body = snapshot.get("drug/drugsfda", search=query.search, limit=query.limit)
        for row in body.get("results") or []:
            number = str(row.get("application_number") or "").upper()
            if _CANON.fullmatch(number) is None:
                continue
            owners[number].add(drug.casefold())
            sponsor = str(row.get("sponsor_name") or "").strip().casefold()
            if sponsor and number not in sponsors:
                sponsors[number] = sponsor
    maps = ApprovalMaps(
        owners={key: frozenset(values) for key, values in owners.items()},
        sponsors=sponsors,
    )
    try:
        snapshot._hard_approval_maps = maps
    except AttributeError:
        pass
    return maps


def holdout_entities(tasks: list[dict], snapshot) -> EntitySet:
    """Union of what appears in the held-out tasks. Drugs are surface names only."""
    maps = approval_maps(snapshot)
    universe = {drug.casefold() for drug in load_universe()}
    drugs: set[str] = set()
    apps: set[str] = set()
    sponsors: set[str] = set()
    for task in tasks:
        drugs |= _surface_drugs(task, universe)
        apps |= _applications(task)
        sponsors |= _sponsors(task, maps)
    return EntitySet(frozenset(drugs), frozenset(apps), frozenset(sponsors))


def task_entities(task: dict, snapshot) -> EntitySet:
    """What a training task would leak, including drugs that own its application numbers."""
    maps = approval_maps(snapshot)
    universe = {drug.casefold() for drug in load_universe()}
    drugs = _surface_drugs(task, universe)
    apps = _applications(task)
    for app in apps:
        drugs |= set(maps.owners.get(app, ()))
    sponsors = _sponsors(task, maps)
    return EntitySet(frozenset(drugs), frozenset(apps), frozenset(sponsors))


def overlaps(train_tasks: list[dict], holdout: EntitySet, snapshot) -> dict[str, set[str]]:
    drugs: set[str] = set()
    apps: set[str] = set()
    sponsors: set[str] = set()
    for task in train_tasks:
        found = task_entities(task, snapshot)
        drugs |= set(found.drugs) & set(holdout.drugs)
        apps |= set(found.applications) & set(holdout.applications)
        sponsors |= set(found.sponsors) & set(holdout.sponsors)
    return {"drugs": drugs, "applications": apps, "sponsors": sponsors}


def assert_disjoint(train_tasks: list[dict], holdout_tasks: list[dict], snapshot) -> dict:
    """Raise ``OverlapError`` listing every overlapping drug, application, and sponsor."""
    if not train_tasks:
        raise RuntimeError("hard_train is empty; refusing a disjointness pass on no tasks")
    holdout = holdout_entities(holdout_tasks, snapshot)
    found = overlaps(train_tasks, holdout, snapshot)
    if any(found.values()):
        raise OverlapError(found)
    train = EntitySet(
        frozenset().union(*(task_entities(task, snapshot).drugs for task in train_tasks)),
        frozenset().union(*(task_entities(task, snapshot).applications for task in train_tasks)),
        frozenset().union(*(task_entities(task, snapshot).sponsors for task in train_tasks)),
    )
    return {
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(holdout_tasks),
        "train": _counts(train),
        "holdout": _counts(holdout),
        "overlap": {"drugs": [], "applications": [], "sponsors": []},
    }


def _counts(found: EntitySet) -> dict[str, int]:
    return {
        "drugs": len(found.drugs),
        "applications": len(found.applications),
        "sponsors": len(found.sponsors),
    }


def _surface_drugs(task: dict, universe: set[str]) -> set[str]:
    found: set[str] = set()
    question = task.get("question") or ""
    try:
        spec = parse_question(question)
    except Unsolvable:
        spec = {}
    for drug in spec.get("drugs") or []:
        if isinstance(drug, str) and drug.strip():
            found.add(drug.strip().casefold())
    entity = ((task.get("entity") or {}).get("generic_name") or "").strip().casefold()
    if entity:
        found.add(entity)
    gold = (task.get("gold") or {}).get("value")
    if isinstance(gold, str) and gold.strip().casefold() in universe:
        found.add(gold.strip().casefold())
    for call in task.get("oracle_calls") or []:
        name = (call.get("arguments") or {}).get("generic_name")
        if isinstance(name, str) and name.strip():
            found.add(name.strip().casefold())
    return found


def _applications(task: dict) -> set[str]:
    return set(_APP.findall(json.dumps(task, ensure_ascii=False)))


def _sponsors(task: dict, maps: ApprovalMaps) -> set[str]:
    found: set[str] = set()
    question = str(task.get("question") or "")
    gold = (task.get("gold") or {}).get("value")
    if isinstance(gold, str) and question.lower().startswith("who is the sponsor"):
        found.add(gold.strip().casefold())
    for cite in task.get("gold_citations") or []:
        fields = cite.get("fields") or []
        if "sponsor_name" not in fields:
            continue
        for app in cite.get("ids") or []:
            sponsor = maps.sponsors.get(str(app).upper())
            if sponsor:
                found.add(sponsor)
    return found
