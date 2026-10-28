"""Queries the task generator reads. A subset of `fdagym.queries.all_queries`.

The year is a hash of the drug name, not of the split seed, so the same drug always
probes the same FAERS window.
"""

import hashlib
from pathlib import Path

from fdagym.queries import Query, approvals_search, faers_report_count, faers_top_reactions, label_search

FAERS_YEAR_MIN = 2020
FAERS_YEAR_SPAN = 6

UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "data" / "snapshots" / "drugs.txt"

# All-severity totals of 1 in these years are possible name collisions, not clean empties.
# Do not emit NOT_FOUND for them even when the serious-only slice is empty.
UNSAFE_NOT_FOUND_YEARS = frozenset(
    {
        ("aprocitentan", 2021),
        ("resmetirom", 2020),
        ("resmetirom", 2021),
        ("resmetirom", 2022),
    }
)


def load_universe(path: Path | None = None) -> tuple[str, ...]:
    """Generic names from drugs.txt. ibuprofen is not in the list and must not be added back."""
    source = path or UNIVERSE_PATH
    drugs: list[str] = []
    for line in source.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name and name not in drugs:
            drugs.append(name)
    if "ibuprofen" in {d.lower() for d in drugs}:
        raise ValueError("ibuprofen exceeds LABEL_LIMIT; diclofenac replaced it")
    return tuple(drugs)


def allow_unanswerable(generic_name: str, year: int | None) -> bool:
    if year is None:
        return True
    return (generic_name.strip().lower(), int(year)) not in UNSAFE_NOT_FOUND_YEARS


def probe_year(generic_name: str) -> int:
    digest = hashlib.sha256(generic_name.strip().lower().encode()).hexdigest()
    return FAERS_YEAR_MIN + (int(digest, 16) % FAERS_YEAR_SPAN)


def absence_year(generic_name: str) -> int:
    """A second serious-FAERS year, used to observe genuine zero-report windows.

    Never equal to `probe_year`, so the two queries don't collapse into one cache key.
    2020 is empty for drugs approved later (mavacamten, finerenone, vericiguat).
    """
    year = probe_year(generic_name)
    return FAERS_YEAR_MIN + 1 if year == FAERS_YEAR_MIN else FAERS_YEAR_MIN


def planned_queries(generic_name: str) -> list[Query]:
    """Label, approvals, three top-reaction windows, and the serious-year count.

    The unfiltered report count is omitted so re-recording stays under the shared
    keyless request budget. The absence year is the window that can be truly empty.
    """
    year = probe_year(generic_name)
    gap = absence_year(generic_name)
    return [
        label_search(generic_name),
        approvals_search(generic_name),
        faers_top_reactions(generic_name),
        faers_top_reactions(generic_name, serious=True, year=year),
        faers_top_reactions(generic_name, serious=True, year=gap),
        faers_report_count(generic_name, serious=True, year=year),
    ]


def queries_for_oracle(calls: list[dict]) -> list[Query]:
    """Snapshot queries an oracle trajectory reads. `get_label_section` uses the label search already issued."""
    found: list[Query] = []
    for call in calls:
        query = _query_for_call(call)
        if query is not None and query not in found:
            found.append(query)
    return found


def _query_for_call(call: dict) -> Query | None:
    tool = call["tool"]
    arguments = call["arguments"]
    drug = arguments.get("generic_name")
    if tool == "search_drug_label":
        return label_search(drug)
    if tool == "search_approvals":
        return approvals_search(drug)
    if tool == "get_label_section":
        return None
    if tool == "query_adverse_events":
        serious = True if arguments.get("serious") is True else None
        year = arguments.get("year")
        if arguments["mode"] == "top_reactions":
            return faers_top_reactions(drug, serious=serious, year=year)
        return faers_report_count(drug, serious=serious, year=year)
    raise ValueError(tool)
