"""Process hops for compare / sponsor_hop training reward.

Eval stays strict. These predicates read the episode's tool outputs and the
question string; they do not read gold or oracle_calls. A hop is an observation,
not a submitted citation. A year hop counts a salt or ester of the question's
generic as that drug (``fdagym.drugnames.product_matches``). ``earliest_boxed``
does not credit a boxed-warning observation unless that drug's original-approval
year is also in the trace, so a labels-only loop scores 0. ``sponsor_hop`` does
not credit a candidate approval page unless the application-number anchor is in
the trace, so searching the named drugs without the NDA/BLA scores 0. Tool order
is not scored: batched years then labels, or interleaved per drug, are the same
join.
"""

from __future__ import annotations

from dataclasses import dataclass

from fdagym.approvals import candidate_numbers, parse_application_number
from fdagym.drugnames import product_matches

HOP_TYPES = frozenset({"compare", "sponsor_hop"})
HOP_CAP = 0.5


@dataclass(frozen=True)
class HopScore:
    applies: bool
    rate: float
    hits: int
    total: int
    notes: list[str]
    flags: tuple[tuple[str, bool], ...] = ()
    variant: str = ""

    def proves(self) -> bool:
        """Whether a right answer on this sponsor trace may keep correctness.

        The three unnamed questions need every flag. An older sponsor question
        needs the application, the sponsor match, and FAERS when the question
        has that step. The boxed-warning keyword is optional once that match
        has already left a single drug, so a filler word in the warning is not
        a second proof.
        """
        if self.variant == "sponsor_boxed":
            return bool(self.flags) and all(ok for _name, ok in self.flags)
        if self.variant in {"sponsor_faers", "sponsor_drug", "sponsor_count"}:
            required = [ok for name, ok in self.flags if name != "label"]
            return bool(required) and all(required)
        return False


def hop_score(steps: list[dict], task: dict) -> HopScore:
    """Fraction of question-derived hops whose evidence appeared in ``steps``."""
    if task.get("type") not in HOP_TYPES:
        return HopScore(False, 0.0, 0, 0, ["hops not used: type is not compare/sponsor_hop"])
    from fdagym.tasks.hard_solver import Unsolvable, parse_question

    try:
        spec = parse_question(task["question"])
    except (Unsolvable, KeyError, TypeError, ValueError):
        return HopScore(True, 0.0, 0, 1, ["hops: question did not parse"])
    flags = _flags(_ok_steps(steps), spec)
    if not flags:
        return HopScore(True, 0.0, 0, 1, ["hops: no predicates for this variant"], (), spec.get("variant", ""))
    hits = sum(1 for _name, ok in flags if ok)
    total = len(flags)
    notes = [f"hops {hits}/{total}"]
    notes.extend(f"hop {name}: {'hit' if ok else 'miss'}" for name, ok in flags)
    return HopScore(True, hits / total, hits, total, notes, tuple(flags), spec["variant"])


def hop_target_searched(steps: list[dict] | None, task: dict) -> bool:
    """True when a tool call looked up the sponsor-hop target drug.

    JP's rule (see ``fdagym.rewards.combine.score``) withholds dense citation
    credit until this is true. The target is ``entity.generic_name``. Task
    generation stores the sponsor's other drug there for ``sponsor_drug``,
    ``sponsor_faers``, and ``sponsor_count``; gold is that drug only for
    ``sponsor_drug`` (a reaction term, or a count, for the other two), so this
    does not read ``gold`` or ``oracle_calls``.

    A lookup is ``arguments.generic_name`` on a successful tool call, matched
    the same way as a sponsor cover hop (case-fold, trimmed). The anchor
    ``application_number`` search does not match. Searching a different
    candidate does not match either. ``search_approvals``, ``search_drug_label``,
    and ``query_adverse_events`` all count.
    """
    if task.get("type") != "sponsor_hop":
        return False
    entity = task.get("entity")
    target = entity.get("generic_name") if isinstance(entity, dict) else None
    if not isinstance(target, str) or not target.strip():
        return False
    for step in _ok_steps(steps):
        if _generic(step.get("arguments") or {}, target):
            return True
    return False


def _flags(steps: list[dict], spec: dict) -> list[tuple[str, bool]]:
    variant = spec["variant"]
    if variant in {"sponsor_faers", "sponsor_drug", "sponsor_count"}:
        return _sponsor_flags(steps, spec)
    if variant == "sponsor_boxed":
        return _sponsor_boxed_flags(steps, spec["application_number"])
    if variant == "earliest_boxed":
        return _earliest_boxed_flags(steps, spec["drugs"])
    if variant == "faers_year":
        return _faers_year_flags(steps, spec["drugs"], spec["year"])
    if variant == "faers_reaction":
        return _faers_reaction_flags(steps, spec["drugs"], spec["term"])
    if variant == "boxed_none":
        return _boxed_none_flags(steps, spec["drugs"])
    if variant == "year_reaction":
        return _year_reaction_flags(steps, spec["drugs"], spec["year"])
    if variant == "earliest_reaction":
        return _earliest_reaction_flags(steps, spec["drugs"])
    return []


def _sponsor_boxed_flags(steps: list[dict], application: str) -> list[tuple[str, bool]]:
    """Unnamed sponsor question. The drugs come from a sponsor_name call for the opened application's sponsor."""
    anchor = _approval_for_app(steps, application)
    sponsors = _sponsors(anchor) if anchor else set()
    drugs = _listed_sponsor_drugs(steps, sponsors)
    flags = [("anchor", bool(sponsors)), ("sponsor_list", bool(drugs))]
    for drug in drugs:
        flags.append((f"label:{drug}", _boxed_warning_searched(steps, drug)))
    if not drugs:
        flags.append(("labels", False))
    return flags


def _listed_sponsor_drugs(steps: list[dict], sponsors: set[str]) -> list[str]:
    if not sponsors:
        return []
    for step in _ok_steps(steps):
        if step.get("tool") != "search_approvals":
            continue
        raw = (step.get("arguments") or {}).get("sponsor_name")
        if not isinstance(raw, str) or raw.strip().casefold() not in sponsors:
            continue
        names = []
        for row in _rows(step.get("output") or {}):
            name = row.get("generic_name")
            if isinstance(name, str) and name.strip() and name not in names:
                names.append(name.strip())
        if names:
            return names
    return []


def _boxed_warning_searched(steps: list[dict], drug: str) -> bool:
    for step in _ok_steps(steps):
        if step.get("tool") != "search_drug_label":
            continue
        arguments = step.get("arguments") or {}
        if _generic(arguments, drug) and arguments.get("has_section") == "boxed_warning":
            return True
    return False


def _sponsor_flags(steps: list[dict], spec: dict) -> list[tuple[str, bool]]:
    drugs: list[str] = spec["drugs"]
    anchor = _approval_for_app(steps, spec["application_number"])
    anchor_ok = bool(anchor and _sponsors(anchor))
    flags = [("anchor", anchor_ok)]
    covered = []
    for drug in drugs:
        found = bool(_approval_steps_for_drug(steps, drug))
        covered.append(found)
        flags.append((f"cover:{drug}", bool(found and anchor_ok)))
    match = _matched_drug(steps, spec["application_number"], drugs) if all(covered) and anchor_ok else None
    flags.append(("match", match is not None))
    if spec["variant"] != "sponsor_count":
        keyword = spec.get("keyword")
        flags.append(("label", bool(match and _boxed_mentions(steps, match, keyword))))
    if spec["variant"] == "sponsor_faers":
        flags.append(("faers", bool(match and _event_hit(steps, match, mode="top_reactions"))))
    if spec["variant"] == "sponsor_count":
        flags.append(
            (
                "faers",
                bool(match and _event_hit(steps, match, mode="count", year=spec["year"], serious=True)),
            )
        )
    return flags


def _earliest_boxed_flags(steps: list[dict], drugs: list[str]) -> list[tuple[str, bool]]:
    flags = []
    years: dict[str, int | None] = {}
    boxed: dict[str, bool | None] = {}
    for drug in drugs:
        years[drug] = _year_for_drug(steps, drug)
        boxed[drug] = _boxed_present(steps, drug)
        year_ok = years[drug] is not None
        flags.append((f"year:{drug}", year_ok))
        flags.append((f"boxed:{drug}", year_ok and boxed[drug] is not None))
    ready = all(years[drug] is not None and boxed[drug] is not None for drug in drugs)
    winners = [drug for drug in drugs if boxed.get(drug) is True]
    unique = None
    if ready and winners:
        unique = _unique_extreme({drug: years[drug] for drug in winners}, least=True)
    flags.append(("unique", unique is not None))
    return flags


def _faers_year_flags(steps: list[dict], drugs: list[str], year: int) -> list[tuple[str, bool]]:
    counts = {}
    flags = []
    for drug in drugs:
        total = _event_total(steps, drug, year=year, serious=True)
        counts[drug] = total
        flags.append((f"count:{drug}", total is not None))
    flags.append(("unique", _unique_extreme(counts, least=False) is not None))
    return flags


def _faers_reaction_flags(steps: list[dict], drugs: list[str], term: str) -> list[tuple[str, bool]]:
    counts = {}
    flags = []
    for drug in drugs:
        n = _reaction_count(steps, drug, term)
        counts[drug] = n
        flags.append((f"term:{drug}", n is not None))
    flags.append(("unique", _unique_extreme(counts, least=False) is not None))
    return flags


def _boxed_none_flags(steps: list[dict], drugs: list[str]) -> list[tuple[str, bool]]:
    flags = []
    resolved = []
    for drug in drugs:
        present = _boxed_present(steps, drug)
        resolved.append(present is not None)
        flags.append((f"label:{drug}", present is not None))
    flags.append(("none", all(resolved) and all(_boxed_present(steps, drug) is False for drug in drugs)))
    return flags


def _year_reaction_flags(steps: list[dict], drugs: list[str], year: int) -> list[tuple[str, bool]]:
    flags = _faers_year_flags(steps, drugs, year)
    winner = _unique_extreme(
        {drug: _event_total(steps, drug, year=year, serious=True) for drug in drugs},
        least=False,
    )
    flags.append(("faers", bool(winner and _event_hit(steps, winner, mode="top_reactions"))))
    return flags


def _earliest_reaction_flags(steps: list[dict], drugs: list[str]) -> list[tuple[str, bool]]:
    flags = []
    years: dict[str, int | None] = {}
    for drug in drugs:
        years[drug] = _year_for_drug(steps, drug)
        flags.append((f"year:{drug}", years[drug] is not None))
    winner = _unique_extreme(years, least=True)
    flags.append(("unique", winner is not None))
    flags.append(("faers", bool(winner and _event_hit(steps, winner, mode="top_reactions"))))
    return flags


def _ok_steps(steps: list[dict] | None) -> list[dict]:
    out = []
    for step in steps or []:
        if not isinstance(step, dict) or step.get("tool") == "submit_answer" or step.get("error"):
            continue
        if not isinstance(step.get("output"), dict):
            continue
        out.append(step)
    return out


def _rows(output: dict) -> list[dict]:
    return [row for row in output.get("results") or [] if isinstance(row, dict)]


def _generic(arguments: dict, drug: str) -> bool:
    got = arguments.get("generic_name")
    return isinstance(got, str) and got.strip().casefold() == drug.strip().casefold()


def _same_app(left: str, right: str) -> bool:
    parsed_left = parse_application_number(left)
    parsed_right = parse_application_number(right)
    if parsed_left is None or parsed_right is None:
        return False
    return bool(set(candidate_numbers(parsed_left)) & set(candidate_numbers(parsed_right)))


def _approval_for_app(steps: list[dict], number: str) -> dict | None:
    for step in steps:
        if step.get("tool") != "search_approvals":
            continue
        raw = (step.get("arguments") or {}).get("application_number")
        if isinstance(raw, str) and _same_app(raw, number):
            return step
    return None


def _approval_steps_for_drug(steps: list[dict], drug: str) -> list[dict]:
    found = []
    for step in steps:
        if step.get("tool") != "search_approvals":
            continue
        arguments = step.get("arguments") or {}
        if "application_number" in arguments:
            continue
        if _generic(arguments, drug):
            found.append(step)
    return found


def _sponsors(step: dict) -> set[str]:
    names: set[str] = set()
    for row in _rows(step.get("output") or {}):
        name = row.get("sponsor_name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip().casefold())
    return names


def _sponsors_for_drug(steps: list[dict], drug: str) -> set[str]:
    names: set[str] = set()
    for step in _approval_steps_for_drug(steps, drug):
        names |= _sponsors(step)
    return names


def _matched_drug(steps: list[dict], number: str, drugs: list[str]) -> str | None:
    anchor = _approval_for_app(steps, number)
    if anchor is None:
        return None
    wanted = _sponsors(anchor)
    if len(wanted) != 1:
        return None
    sponsor = next(iter(wanted))
    hits = [drug for drug in drugs if sponsor in _sponsors_for_drug(steps, drug)]
    if len(hits) != 1:
        return None
    return hits[0]


def _label_steps_for_drug(steps: list[dict], drug: str) -> list[dict]:
    found = []
    for step in steps:
        if step.get("tool") != "search_drug_label":
            continue
        if _generic(step.get("arguments") or {}, drug):
            found.append(step)
    return found


def _set_ids_for_drug(steps: list[dict], drug: str) -> set[str]:
    ids: set[str] = set()
    for step in _label_steps_for_drug(steps, drug):
        for row in _rows(step.get("output") or {}):
            for key in ("set_id", "id"):
                value = row.get(key)
                if isinstance(value, str) and value and not value.startswith("step:"):
                    ids.add(value)
    return ids


def _boxed_present(steps: list[dict], drug: str) -> bool | None:
    labels = _label_steps_for_drug(steps, drug)
    if not labels:
        return None
    if _boxed_text(steps, drug):
        return True
    for step in labels:
        for row in _rows(step.get("output") or {}):
            sections = row.get("sections") or []
            if "boxed_warning" in sections:
                return True
            text = row.get("boxed_warning")
            if isinstance(text, str) and text.strip():
                return True
    return False


def _boxed_text(steps: list[dict], drug: str) -> str:
    chunks: list[str] = []
    for step in _label_steps_for_drug(steps, drug):
        for row in _rows(step.get("output") or {}):
            text = row.get("boxed_warning")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    ids = _set_ids_for_drug(steps, drug)
    for step in steps:
        if step.get("tool") != "get_label_section":
            continue
        arguments = step.get("arguments") or {}
        if arguments.get("section") != "boxed_warning":
            continue
        if arguments.get("set_id") not in ids:
            continue
        text = (step.get("output") or {}).get("boxed_warning")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks)


def _boxed_mentions(steps: list[dict], drug: str, keyword: str | None) -> bool:
    if _boxed_present(steps, drug) is not True:
        return False
    if not keyword:
        return True
    return keyword.casefold() in _boxed_text(steps, drug).casefold()


def _event_steps(steps: list[dict], drug: str, *, mode: str, year: int | None = None, serious: bool | None = None) -> list[dict]:
    found = []
    for step in steps:
        if step.get("tool") != "query_adverse_events":
            continue
        arguments = step.get("arguments") or {}
        if not _generic(arguments, drug):
            continue
        if arguments.get("mode") != mode:
            continue
        if year is not None and _year_arg(arguments) != year:
            continue
        if serious is True and arguments.get("serious") is not True:
            continue
        found.append(step)
    return found


def _event_hit(steps: list[dict], drug: str, *, mode: str, year: int | None = None, serious: bool | None = None) -> bool:
    return bool(_event_steps(steps, drug, mode=mode, year=year, serious=serious))


def _event_total(steps: list[dict], drug: str, *, year: int, serious: bool) -> int | None:
    for step in _event_steps(steps, drug, mode="count", year=year, serious=serious):
        total = ((step.get("output") or {}).get("meta") or {}).get("results") or {}
        value = total.get("total") if isinstance(total, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _reaction_count(steps: list[dict], drug: str, term: str) -> int | None:
    wanted = term.strip().casefold()
    for step in _event_steps(steps, drug, mode="top_reactions"):
        for row in _rows(step.get("output") or {}):
            got = row.get("term")
            if isinstance(got, str) and got.strip().casefold() == wanted and isinstance(row.get("count"), int):
                return row["count"]
    return None


def _year_arg(arguments: dict) -> int | None:
    year = arguments.get("year")
    if isinstance(year, bool) or not isinstance(year, int):
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        else:
            return None
    if 1900 <= year <= 2100:
        return year
    return None


def _year_for_drug(steps: list[dict], drug: str) -> int | None:
    best = None
    for step in _approval_steps_for_drug(steps, drug):
        for row in _rows(step.get("output") or {}):
            year = _visible_year(row, drug)
            if year is not None and (best is None or year < best):
                best = year
    return best


def _visible_year(row: dict, drug: str) -> int | None:
    number = str(row.get("application_number") or "")
    if not number.upper().startswith(("NDA", "BLA")):
        return None
    products = [item for item in row.get("products") or [] if isinstance(item, dict)]
    if (
        products
        and _ingredient_names_present(products)
        and not any(product_matches(product, drug) for product in products)
    ):
        return None
    years = []
    for sub in row.get("submissions") or []:
        if not isinstance(sub, dict):
            continue
        if str(sub.get("submission_type", "")).upper() != "ORIG":
            continue
        if str(sub.get("submission_status", "")).upper() not in {"AP", "APPROVED"}:
            continue
        date = sub.get("submission_status_date")
        text = str(date) if date is not None else ""
        if len(text) >= 4 and text[:4].isdigit():
            year = int(text[:4])
            if 1900 <= year <= 2100:
                years.append(year)
    return min(years) if years else None


def _ingredient_names_present(products: list[dict]) -> bool:
    for product in products:
        for item in product.get("active_ingredients") or []:
            if isinstance(item, dict) and str(item.get("name") or "").strip():
                return True
    return False


def _unique_extreme(values: dict[str, int | None], *, least: bool) -> str | None:
    if any(value is None for value in values.values()) or not values:
        return None
    filled = {key: int(value) for key, value in values.items() if value is not None}
    target = min(filled.values()) if least else max(filled.values())
    winners = [key for key, value in filled.items() if value == target]
    if len(winners) != 1:
        return None
    return winners[0]
