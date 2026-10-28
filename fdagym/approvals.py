"""Application-number lookup over Drugs@FDA records the snapshot already stored.

Approvals are frozen under `products.active_ingredients.name:"<generic>"`. Questions
that only name an NDA/BLA/ANDA cannot issue that query. The lookup reads the cached
bodies instead of asking openFDA for a search the freeze does not contain.
"""

import re
from collections.abc import Iterable

_APPLICATION = re.compile(r"^(NDA|ANDA|BLA)?(\d{4,6})$")
_CANONICAL = re.compile(r"^(NDA|ANDA|BLA)(\d{6})$")
_PREFIX_ORDER = ("NDA", "BLA", "ANDA")


def parse_application_number(raw: object) -> tuple[str | None, str] | None:
    """Return `(prefix or None, 6-digit stem)`.

    Accepts `NDA020977`, `NDA 020977`, `NDA-020977`, `020977`, and the same
    shapes for BLA and ANDA. Fewer than 6 digits are zero-padded. Anything
    else is not an application number.
    """
    if not isinstance(raw, str):
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    match = _APPLICATION.fullmatch(text)
    if match is None:
        return None
    prefix, digits = match.group(1), match.group(2).zfill(6)
    return prefix, digits


def candidate_numbers(parsed: tuple[str | None, str]) -> list[str]:
    """Ids to try, NDA then BLA then ANDA when the caller omitted the prefix."""
    prefix, digits = parsed
    if prefix:
        return [f"{prefix}{digits}"]
    return [f"{kind}{digits}" for kind in _PREFIX_ORDER]


def approval_index(snapshot) -> dict[str, dict]:
    """Map `NDA020977` to the cached application record. Built once per snapshot."""
    cached = getattr(snapshot, "_approval_index", None)
    if isinstance(cached, dict):
        return cached
    index: dict[str, dict] = {}
    for record in _iter_approval_records(snapshot):
        number = record.get("application_number")
        if not isinstance(number, str) or not number:
            continue
        key = number.upper()
        if _CANONICAL.fullmatch(key) is None:
            continue
        prev = index.get(key)
        if prev is None or _richer(record, prev):
            index[key] = record
    try:
        snapshot._approval_index = index
    except AttributeError:
        pass
    return index


def sponsor_drugs(snapshot, sponsor_name: str) -> list[dict]:
    """Snapshot drugs whose NDA or BLA names this sponsor.

    Each row is ``generic_name`` plus the application numbers that carried the
    match. The name is the full sponsor string, case ignored. A sponsor the
    freeze does not have returns an empty list.
    """
    key = sponsor_name.strip().casefold()
    if not key:
        return []
    return [dict(row) for row in _sponsor_index(snapshot).get(key, ())]


def _sponsor_index(snapshot) -> dict[str, list[dict]]:
    cached = getattr(snapshot, "_sponsor_index", None)
    if isinstance(cached, dict):
        return cached
    from fdagym.drugnames import names_match
    from fdagym.tasks.plan import load_universe

    universe = load_universe()
    grouped: dict[str, dict[str, set[str]]] = {}
    for record in _iter_approval_records(snapshot):
        number = str(record.get("application_number") or "").upper()
        if not number.startswith(("NDA", "BLA")):
            continue
        sponsor = str(record.get("sponsor_name") or "").strip().casefold()
        if not sponsor:
            continue
        for product in record.get("products") or []:
            if not isinstance(product, dict):
                continue
            for ingredient in product.get("active_ingredients") or []:
                raw = ingredient.get("name") if isinstance(ingredient, dict) else None
                if not isinstance(raw, str):
                    continue
                for drug in universe:
                    if names_match(raw, drug):
                        grouped.setdefault(sponsor, {}).setdefault(drug, set()).add(number)
    index = {
        sponsor: [
            {"generic_name": drug, "application_numbers": sorted(numbers)}
            for drug, numbers in sorted(drugs.items())
        ]
        for sponsor, drugs in grouped.items()
    }
    try:
        snapshot._sponsor_index = index
    except AttributeError:
        pass
    return index


def find_applications(snapshot, raw: str) -> list[dict] | None:
    """Matching records, or None when `raw` is not an application number.

    An unknown but well-formed number returns an empty list.
    """
    parsed = parse_application_number(raw)
    if parsed is None:
        return None
    index = approval_index(snapshot)
    return [index[number] for number in candidate_numbers(parsed) if number in index]


def _iter_approval_records(snapshot) -> Iterable[dict]:
    loader = getattr(snapshot, "drugsfda_records", None)
    if callable(loader):
        yield from loader()
        return
    rows = getattr(snapshot, "_rows", None)
    if rows is None:
        rows = getattr(snapshot, "_mapping", None)
    if not isinstance(rows, dict):
        return
    for key, body in rows.items():
        endpoint = key[0] if isinstance(key, tuple) and key else None
        if endpoint != "drug/drugsfda" or not isinstance(body, dict):
            continue
        for record in body.get("results") or []:
            if isinstance(record, dict):
                yield record


def _richer(new: dict, old: dict) -> bool:
    return (len(new.get("submissions") or []), len(new.get("products") or [])) > (
        len(old.get("submissions") or []),
        len(old.get("products") or []),
    )
