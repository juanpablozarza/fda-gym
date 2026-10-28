"""Canonical openFDA queries: the only queries the snapshot records and the tools issue.

Tools take constrained arguments (a drug name, a year, a seriousness flag) instead of raw openFDA
search strings, so every query an agent can trigger is enumerable ahead of time and the frozen
snapshot can answer it. The snapshot script records `all_queries(drug)` for every drug in the
universe; tools and the task generator build queries only through these functions.
"""

from typing import NamedTuple

from fdagym.snapshot import Endpoint

FAERS_YEARS = range(2020, 2026)
# openFDA's per-request max. Metformin has 365 labels (47 MB untrimmed), so the snapshot must trim records.
LABEL_LIMIT = 1000
# Metformin has 214 applications; a page of 100 dropped Glucophage (NDA020357), the original approval.
APPROVALS_LIMIT = 1000
TOP_REACTIONS_LIMIT = 10


class Query(NamedTuple):
    endpoint: Endpoint
    search: str | None
    count: str | None
    limit: int


def _name(generic_name: str) -> str:
    return generic_name.strip().lower()


def _faers_drug(generic_name: str) -> str:
    # openFDA only harmonizes reports for drugs it has mapped: generic_name finds 0 finerenone reports while
    # activesubstancename finds 2,960. Neither field is a superset (metformin: 439k vs 402k, union 492k).
    name = _name(generic_name)
    return f'(patient.drug.openfda.generic_name:"{name}" OR patient.drug.activesubstance.activesubstancename:"{name}")'


def _faers_search(generic_name: str, serious: bool | None, year: int | None) -> str:
    parts = [_faers_drug(generic_name)]
    if serious is not None:
        parts.append(f"serious:{1 if serious else 2}")
    if year is not None:
        parts.append(f"receivedate:[{year}0101 TO {year}1231]")
    return " AND ".join(parts)


def label_search(generic_name: str) -> Query:
    return Query("drug/label", f'openfda.generic_name:"{_name(generic_name)}"', None, LABEL_LIMIT)


def approvals_search(generic_name: str) -> Query:
    # Many Drugs@FDA records have no `openfda` block: searching openfda.generic_name finds 1 of 6 semaglutide
    # applications (missing Ozempic's NDA209637) and 5 of 21 warfarin ones.
    return Query("drug/drugsfda", f'products.active_ingredients.name:"{_name(generic_name)}"', None, APPROVALS_LIMIT)


def faers_top_reactions(generic_name: str, *, serious: bool | None = None, year: int | None = None) -> Query:
    return Query(
        "drug/event",
        _faers_search(generic_name, serious, year),
        "patient.reaction.reactionmeddrapt.exact",
        TOP_REACTIONS_LIMIT,
    )


def faers_report_count(generic_name: str, *, serious: bool | None = None, year: int | None = None) -> Query:
    """The answer is meta.results.total of the response."""
    return Query("drug/event", _faers_search(generic_name, serious, year), None, 1)


def all_queries(generic_name: str) -> list[Query]:
    faers = [
        build(generic_name, serious=serious, year=year)
        for build in (faers_top_reactions, faers_report_count)
        for serious in (None, True)
        for year in (None, *FAERS_YEARS)
    ]
    return [label_search(generic_name), approvals_search(generic_name), *faers]
