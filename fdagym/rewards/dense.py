"""Near-miss correctness for the training reward.

The strict scorer already gives full credit for case and whitespace. This layer only
fires when that score is 0. Punctuation, a year pulled out of a date, a 10% numeric
band on counts, and a brand/generic synonym from the snapshot each earn a fraction
of the correctness component. Years are not given the numeric band: 1999 is not
partial credit for 1998.

Weights, as a fraction of the correctness component (itself 0.60 of the total):

- punctuation-normalized match: 0.85
- year extracted from a date: 0.70
- brand or generic synonym: 0.60
- count inside a 10% relative band: 0.50 * (1 - rel / 0.10)
- composite "<count>; yes|no": 0.50 for each correct half (exact count, exact yes/no).
  Both halves right but the strict match missed (say "8,714; Yes") is capped at
  the punctuation rate. Training only: the strict eval still needs the whole string.

A citation whose id matches gold but whose field or quote does not earns 0.50 of
the citation component (0.25 of the total). Format stays weighted at 0.10.
"""

from __future__ import annotations

import re

from fdagym.queries import approvals_search, label_search
from fdagym.rewards.correctness import _as_number, _is_not_found, _match
from fdagym.snapshot import Snapshot, SnapshotMiss

PUNCT = 0.85
YEAR = 0.70
SYNONYM = 0.60
COUNT_BAND = 0.10
COUNT_CAP = 0.50
COMPOSITE_HALF = 0.50

_PUNCT = re.compile(r"[^\w\s]+")
_COMPACT_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)")
_SPLIT_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-/](\d{2})[-/](\d{2})(?!\d)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_YES_NO = re.compile(r"^(yes|no)\b", re.IGNORECASE)


def norm_punct(value: object) -> str:
    text = _PUNCT.sub(" ", str(value).strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def years_in(value: object) -> set[int]:
    """Years that are the YYYY of a real-looking date, not bare four-digit numbers."""
    text = str(value)
    found: set[int] = set()
    for pattern in (_COMPACT_DATE, _SPLIT_DATE):
        for match in pattern.finditer(text):
            year, month, day = (int(match.group(i)) for i in (1, 2, 3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                found.add(year)
    return found


def _is_year(value: float) -> bool:
    return value == int(value) and 1900 <= int(value) <= 2100


def synonym_names(snapshot: Snapshot | None, task: dict) -> set[str]:
    """Generic and single-ingredient brand names for this task's drug."""
    entity = task.get("entity") or {}
    generic = entity.get("generic_name") if isinstance(entity, dict) else None
    names: set[str] = set()
    if isinstance(entity, dict):
        for key in ("generic_name", "brand_name"):
            value = entity.get(key)
            if isinstance(value, str) and value.strip():
                names.add(norm_punct(value))
    if snapshot is None or not isinstance(generic, str) or not generic.strip():
        return {name for name in names if name}
    wanted = norm_punct(generic)
    names |= _label_names(snapshot, generic, wanted)
    names |= _approval_names(snapshot, generic, wanted)
    return {name for name in names if name}


def _label_names(snapshot: Snapshot, generic: str, wanted: str) -> set[str]:
    body = _fetch(snapshot, label_search(generic))
    names: set[str] = set()
    for record in body.get("results") or []:
        openfda = record.get("openfda") or {}
        generics = [norm_punct(item) for item in _as_list(openfda.get("generic_name"))]
        if generics != [wanted]:
            continue
        names.add(wanted)
        for item in _as_list(openfda.get("brand_name")):
            names.add(norm_punct(item))
    return names


def _approval_names(snapshot: Snapshot, generic: str, wanted: str) -> set[str]:
    body = _fetch(snapshot, approvals_search(generic))
    names: set[str] = set()
    for record in body.get("results") or []:
        for product in record.get("products") or []:
            ingredients = [
                norm_punct(item.get("name") or "")
                for item in product.get("active_ingredients") or []
                if isinstance(item, dict)
            ]
            if ingredients != [wanted]:
                continue
            names.add(wanted)
            brand = product.get("brand_name")
            if isinstance(brand, str) and brand.strip():
                names.add(norm_punct(brand))
    return names


def _fetch(snapshot: Snapshot, query) -> dict:
    try:
        body = snapshot.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
    except SnapshotMiss:
        return {}
    return body if isinstance(body, dict) else {}


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def dense_correctness(
    answer: object,
    task: dict,
    *,
    valid_citations: int,
    id_hits: int,
    format_ok: bool,
    empty_grounded: bool,
    judge,
    snapshot: Snapshot | None,
) -> tuple[float, list[str]]:
    if not format_ok:
        return 0.0, ["correctness forced to 0: invalid format"]
    if valid_citations < 1 and id_hits < 1:
        return 0.0, ["correctness gated: cited id does not match a gold citation id"]
    strict, notes = _match(answer, task, empty_grounded=empty_grounded, judge=judge)
    if strict > 0:
        return strict, notes
    composite = composite_halves(answer, task)
    if composite is not None:
        return composite
    near = _near(answer, task, snapshot)
    if near is None:
        return 0.0, notes
    score, note = near
    return score, [*notes, note]


def _split_composite(value: object) -> tuple[float, str] | None:
    """``"8714; yes"`` -> ``(8714.0, "yes")``. None when it is not that shape."""
    if not isinstance(value, str) or value.count(";") != 1:
        return None
    left, right = (part.strip() for part in value.split(";"))
    number = _as_number(left.replace(",", ""))
    match = _YES_NO.match(right)
    if number is None or match is None:
        return None
    return number, match.group(1).lower()


def composite_halves(answer: object, task: dict) -> tuple[float, list[str]] | None:
    """0.5 per correct half of a composite count/yes-no answer. None for other tasks or shapes."""
    if task.get("type") != "composite":
        return None
    gold = task.get("gold") or {}
    if gold.get("kind") != "exact":
        return None
    want = _split_composite(gold.get("value"))
    if want is None:
        return None
    got = _split_composite(answer)
    if got is None:
        return 0.0, ["dense composite: answer is not '<count>; yes|no'"]
    count_ok = got[0] == want[0]
    flag_ok = got[1] == want[1]
    score = COMPOSITE_HALF * count_ok + COMPOSITE_HALF * flag_ok
    if count_ok and flag_ok:
        score = PUNCT
    return score, [f"dense composite: count {'ok' if count_ok else 'wrong'}, yes/no {'ok' if flag_ok else 'wrong'}, score {score:.2f}"]


def _near(answer: object, task: dict, snapshot: Snapshot | None) -> tuple[float, str] | None:
    if _is_not_found(answer):
        return None
    gold = task.get("gold") or {}
    kind = gold.get("kind")
    if kind == "exact":
        return _near_exact(answer, gold, synonym_names(snapshot, task))
    if kind == "number":
        return _near_number(answer, gold)
    return None


def _near_exact(answer: object, gold: dict, names: set[str]) -> tuple[float, str] | None:
    if isinstance(answer, str) and re.search(r"\bor\b", answer, re.IGNORECASE):
        return None
    wanted = [gold.get("value"), *(gold.get("aliases") or [])]
    folded = {norm_punct(item) for item in wanted if item is not None}
    if norm_punct(answer) in folded:
        return PUNCT, "dense exact: punctuation-normalized match"
    gold_years: set[int] = set()
    for item in wanted:
        gold_years |= years_in(item)
        number = _as_number(item)
        if number is not None and _is_year(number):
            gold_years.add(int(number))
    if gold_years and years_in(answer) & gold_years:
        year = min(years_in(answer) & gold_years)
        return YEAR, f"dense exact: year {year} extracted from a date"
    pred = norm_punct(answer)
    gold_names = {norm_punct(item) for item in wanted if item is not None}
    if pred in names and gold_names & names and pred not in gold_names:
        return SYNONYM, "dense exact: brand or generic synonym from the snapshot"
    return None


def _near_number(answer: object, gold: dict) -> tuple[float, str] | None:
    target = _as_number(gold.get("value"))
    if target is None:
        return None
    if _is_year(target):
        if int(target) in years_in(answer):
            return YEAR, f"dense number: year {int(target)} extracted from a date"
        return None
    pred = _as_number(answer)
    if pred is None and isinstance(answer, str):
        match = _NUMBER.search(answer)
        if match:
            pred = _as_number(match.group(0))
    if pred is None or target == 0:
        return None
    rel = abs(pred - target) / abs(target)
    if rel <= 1e-12 or rel > COUNT_BAND:
        return None
    score = COUNT_CAP * (1.0 - rel / COUNT_BAND)
    return score, f"dense number: relative error {rel:.4f} inside {COUNT_BAND:g}, score {score:.4f}"
