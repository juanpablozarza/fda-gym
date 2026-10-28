"""Reference solver for the hard eval split.

It reads the question and tool observations only. It does not read ``gold``,
``oracle_calls``, or any other field stored on the task. Every observation has
to parse inside the 3,500-character clip the rollout shows the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from fdagym.rewards.citation import _collapse, _render, _resolve
from fdagym.tasks.generators import _orig_year, _product_is_single
from fdagym.tools import EpisodeState, execute
from fdagym.train.rollout import observation_payload

OBS_LIMIT = 3500
BUDGET = 6

_OF = re.compile(r"^Of (.+), which drug ")
_APP = re.compile(r"^Application ([A-Z]{3}\d{6}) is held by a sponsor")
_DRUGS_OF = re.compile(r"one of these drugs: (.+?)\.")
_KEYWORD = re.compile(r"boxed warning mentioning ([a-z]{6,20})\.")
_YEAR = re.compile(r"receiv(?:e|ed) in (\d{4})")
_TERM = re.compile(r"most FAERS reports of (.+)\?$")
_OR_WORD = re.compile(r"\bor\b", re.IGNORECASE)


class Unsolvable(Exception):
    """The question cannot be settled from the observations this solver is allowed to read."""


@dataclass
class Solution:
    variant: str
    answer: object
    calls: list[dict]
    citations: list[dict]
    observation_chars: list[int] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def question_earliest(drugs: list[str]) -> str:
    a, b, c = drugs
    return f"Of {a}, {b}, and {c}, which drug has a boxed warning and the earliest original FDA approval (NDA or BLA)?"


def question_faers_year(drugs: list[str], year: int) -> str:
    a, b, c = drugs
    return f"Of {a}, {b}, and {c}, which drug has the most serious FAERS reports received in {year}?"


def question_reaction(drugs: list[str], term: str) -> str:
    a, b, c = drugs
    return f"Of {a}, {b}, and {c}, which drug has the most FAERS reports of {term}?"


def question_none(drugs: list[str]) -> str:
    a, b, c = drugs
    return f"Of {a}, {b}, and {c}, which drug has a boxed warning on its label? Answer no if none of them do."


def question_none_yes_no(drugs: list[str]) -> str:
    """Train boxed_none prompt. Eval keeps ``question_none`` so ``hard.jsonl`` stays reproducible."""
    a, b, c = drugs
    return (
        f"Of {a}, {b}, and {c}, which drug has a boxed warning on its label, if any? "
        "Reply with yes or no only, not a drug name. "
        "Answer yes if at least one does. Answer no if none of them do."
    )


def question_year_reaction(drugs: list[str], year: int) -> str:
    a, b, c = drugs
    return (
        f"Of {a}, {b}, and {c}, which drug has the most serious FAERS reports received in {year}? "
        "Answer with only that drug's most-reported FAERS reaction, not the drug name."
    )


def question_earliest_reaction(drugs: list[str]) -> str:
    a, b, c = drugs
    return (
        f"Of {a}, {b}, and {c}, which drug has the earliest original FDA approval (NDA or BLA)? "
        "Answer with only that drug's most-reported FAERS reaction, not the drug name."
    )


def question_sponsor_count(application: str, drugs: list[str], year: int) -> str:
    listed = _listed(drugs)
    return (
        f"Application {application} is held by a sponsor that also markets one of these drugs: {listed}. "
        f"How many serious FAERS reports did that drug receive in {year}?"
    )


def _listed(drugs: list[str]) -> str:
    if len(drugs) == 2:
        return f"{drugs[0]} or {drugs[1]}"
    if len(drugs) == 3:
        return f"{drugs[0]}, {drugs[1]}, or {drugs[2]}"
    raise Unsolvable("sponsor candidates must be 2 or 3 drugs")


def question_sponsor_faers(application: str, drugs: list[str], keyword: str) -> str:
    a, b = drugs
    return (
        f"Application {application} is held by a sponsor that also markets one of these drugs: {a} or {b}. "
        f"That other drug has a boxed warning mentioning {keyword}. "
        "What is that drug's most-reported FAERS reaction?"
    )


def question_sponsor_drug(application: str, drugs: list[str], keyword: str) -> str:
    a, b, c = drugs
    return (
        f"Application {application} is held by a sponsor that also markets one of these drugs: "
        f"{a}, {b}, or {c}. "
        f"That other drug has a boxed warning mentioning {keyword}. "
        "What is that drug?"
    )


_UNNAMED_SPONSOR = re.compile(
    r"^The sponsor of application ((?:NDA|BLA|ANDA)\d{6}) markets two drugs in the snapshot\. "
    r"Which of them has a boxed warning\?$"
)


def parse_question(question: str) -> dict:
    """Pull only the strings the question itself prints."""
    unnamed = _UNNAMED_SPONSOR.match(question)
    if unnamed:
        return {"variant": "sponsor_boxed", "application_number": unnamed.group(1), "drugs": []}
    if question.startswith("Of "):
        match = _OF.match(question)
        if match is None:
            raise Unsolvable("compare question has no drug list")
        drugs = _split_and(match.group(1))
        if "most serious FAERS reports received in" in question and "most-reported FAERS reaction" in question:
            return {"variant": "year_reaction", "drugs": drugs, "year": _year_of(question)}
        if "earliest original FDA approval" in question and "most-reported FAERS reaction" in question:
            return {"variant": "earliest_reaction", "drugs": drugs}
        if "earliest original FDA approval" in question:
            return {"variant": "earliest_boxed", "drugs": drugs}
        if "most serious FAERS reports received in" in question:
            return {"variant": "faers_year", "drugs": drugs, "year": _year_of(question)}
        if "most FAERS reports of" in question:
            term = _TERM.search(question)
            if term is None:
                raise Unsolvable("missing reaction term")
            return {"variant": "faers_reaction", "drugs": drugs, "term": term.group(1)}
        if "Answer no if none of them do." in question:
            return {"variant": "boxed_none", "drugs": drugs}
        raise Unsolvable("unknown compare question")

    app = _APP.match(question)
    drugs = _DRUGS_OF.search(question)
    if app is not None and drugs is not None and "How many serious FAERS reports did that drug receive in" in question:
        return {
            "variant": "sponsor_count",
            "application_number": app.group(1),
            "drugs": _split_or(drugs.group(1)),
            "year": _year_of(question),
        }
    keyword = _KEYWORD.search(question)
    if app is None or drugs is None or keyword is None:
        raise Unsolvable("sponsor question is missing the application, the candidates, or the keyword")
    names = _split_or(drugs.group(1))
    variant = "sponsor_faers" if "most-reported FAERS reaction" in question else "sponsor_drug"
    if variant == "sponsor_drug" and not question.endswith("What is that drug?"):
        raise Unsolvable("unknown sponsor question")
    return {
        "variant": variant,
        "application_number": app.group(1),
        "drugs": names,
        "keyword": keyword.group(1),
    }


def _year_of(question: str) -> int:
    year = _YEAR.search(question)
    if year is None:
        raise Unsolvable("missing FAERS year")
    return int(year.group(1))


def solve(snapshot, question: str, *, budget: int = BUDGET) -> Solution:
    """Play the question through the real tools. Raises ``Unsolvable`` on a tie or a blind hop.

    ``budget`` defaults to 6, the cap the hard split was generated under.
    Reachability passes the eval budget (8 for a new run, 6 to reproduce the old check).
    The reference path does not grow to fill a wider budget.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget must be an integer >= 1, got {budget!r}")
    spec = parse_question(question)
    session = _Session(snapshot, budget)
    variant = spec["variant"]
    if variant == "earliest_boxed":
        return _earliest(session, spec)
    if variant == "faers_year":
        return _faers_year(session, spec)
    if variant == "faers_reaction":
        return _faers_reaction(session, spec)
    if variant == "boxed_none":
        return _boxed_none(session, spec)
    if variant == "year_reaction":
        return _year_reaction(session, spec)
    if variant == "earliest_reaction":
        return _earliest_reaction(session, spec)
    if variant == "sponsor_count":
        return _sponsor_count(session, spec)
    if variant in {"sponsor_faers", "sponsor_drug"}:
        return _sponsor(session, spec)
    raise Unsolvable(variant)


def same_answer(got: object, gold: object) -> bool:
    """Gold booleans are submitted as yes/no. Everything else matches after case-folding."""
    if gold is False:
        return got == "no" or got is False
    if gold is True:
        return got == "yes" or got is True
    if isinstance(got, str) and isinstance(gold, str):
        return _collapse(got).casefold() == _collapse(gold).casefold()
    return got == gold


def run_call(state: EpisodeState, tool: str, arguments: dict, index: int, *, budget: int = BUDGET) -> tuple[dict, int]:
    """One tool call, as the rollout shows it: full JSON, or ``Unsolvable`` if the clip would cut it."""
    output, err = execute(state, tool, dict(arguments), step_index=index)
    obs = {
        "tool": tool,
        "output": output,
        "error": err,
        "tool_calls_remaining": max(0, budget - index),
    }
    payload = observation_payload(obs)
    if len(payload) > OBS_LIMIT:
        raise Unsolvable(f"{tool} observation is {len(payload)} characters")
    if err:
        raise Unsolvable(str(err))
    body = json.loads(payload)
    shown = body.get("output")
    if not isinstance(shown, dict):
        raise Unsolvable(f"{tool} returned {type(shown).__name__}")
    return shown, len(payload)


class _Session:
    def __init__(self, snapshot, budget: int = BUDGET):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be an integer >= 1")
        self.budget = budget
        self.state = EpisodeState(snapshot)
        self.calls: list[dict] = []
        self.lengths: list[int] = []

    def call(self, tool: str, arguments: dict) -> dict:
        if len(self.calls) >= self.budget:
            raise Unsolvable("tool budget exhausted")
        index = len(self.calls) + 1
        output, length = run_call(self.state, tool, arguments, index, budget=self.budget)
        self.calls.append({"tool": tool, "arguments": dict(arguments)})
        self.lengths.append(length)
        return output

    def finish(self, variant: str, answer: object, citations: list[dict]) -> Solution:
        if _OR_WORD.search(str(answer)):
            raise Unsolvable("answer contains 'or', which the exact scorer rejects")
        return Solution(variant, answer, self.calls, citations, list(self.lengths))


def _earliest(session: _Session, spec: dict) -> Solution:
    years: dict[str, int] = {}
    apps: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    boxed_nodes: dict[str, dict] = {}
    for drug in spec["drugs"]:
        output = session.call("search_approvals", {"generic_name": drug})
        year, chosen, node = _earliest_visible(output, drug)
        if year is None or node is None:
            raise Unsolvable(f"{drug} has no visible original approval")
        years[drug] = year
        apps[drug] = chosen
        nodes[drug] = node
        labels = session.call("search_drug_label", {"generic_name": drug, "has_section": "boxed_warning"})
        if _total(labels) > 0:
            row = _first_boxed_row(labels)
            if row is None:
                raise Unsolvable(f"{drug} boxed warning is past the observation")
            boxed_nodes[drug] = row
    contenders = [drug for drug in spec["drugs"] if drug in boxed_nodes]
    if not contenders:
        raise Unsolvable("no drug has a boxed warning")
    winner = _unique(contenders, years, least=True)
    if winner is None:
        raise Unsolvable("tied earliest boxed approval")
    citations = [
        _cite("drug/drugsfda", apps[winner], "submissions", nodes[winner]),
        _cite("drug/label", boxed_nodes[winner]["set_id"], "sections", boxed_nodes[winner]),
    ]
    return session.finish("earliest_boxed", winner, citations)


def _faers_year(session: _Session, spec: dict) -> Solution:
    counts: dict[str, int] = {}
    nodes: dict[str, dict] = {}
    for drug in spec["drugs"]:
        output = session.call(
            "query_adverse_events",
            {"generic_name": drug, "mode": "count", "serious": True, "year": spec["year"]},
        )
        counts[drug] = _total(output)
        nodes[drug] = output
    winner = _unique(spec["drugs"], counts, least=False)
    if winner is None:
        raise Unsolvable("tied serious FAERS count")
    cite = _cite("drug/event", output_id(nodes[winner]), "meta.results.total", nodes[winner])
    return session.finish("faers_year", winner, [cite])


def _faers_reaction(session: _Session, spec: dict) -> Solution:
    term = spec["term"]
    counts: dict[str, int] = {}
    nodes: dict[str, dict] = {}
    for drug in spec["drugs"]:
        output = session.call("query_adverse_events", {"generic_name": drug, "mode": "top_reactions"})
        found = _term_count(output, term)
        if found is None:
            raise Unsolvable(f"{term} is not in the visible reactions for {drug}")
        counts[drug] = found
        nodes[drug] = output
    winner = _unique(spec["drugs"], counts, least=False)
    if winner is None:
        raise Unsolvable("tied reaction count")
    cite = _cite("drug/event", output_id(nodes[winner]), "results", nodes[winner])
    return session.finish("faers_reaction", winner, [cite])


def _boxed_none(session: _Session, spec: dict) -> Solution:
    citations = []
    hits = []
    for drug in spec["drugs"]:
        output = session.call("search_drug_label", {"generic_name": drug})
        rows = _rows(output)
        if _total(output) > len(rows):
            raise Unsolvable(f"{drug} labels continue on another page")
        for row in rows:
            sections = row.get("sections") or []
            if "boxed_warning" in sections:
                hits.append(drug)
                break
        else:
            if not rows or not rows[0].get("set_id"):
                raise Unsolvable(f"{drug} has no visible label")
            citations.append(_cite("drug/label", rows[0]["set_id"], "sections", rows[0]))
    if hits:
        raise Unsolvable("a named drug has a boxed warning")
    return session.finish("boxed_none", "no", citations)


def _year_reaction(session: _Session, spec: dict) -> Solution:
    counts, nodes = _serious_counts(session, spec)
    winner = _unique(spec["drugs"], counts, least=False)
    if winner is None:
        raise Unsolvable("tied serious FAERS count")
    events = session.call("query_adverse_events", {"generic_name": winner, "mode": "top_reactions"})
    term = _top_term(events)
    citations = [
        _cite("drug/event", output_id(nodes[winner]), "meta.results.total", nodes[winner]),
        _cite("drug/event", output_id(events), "results", events),
    ]
    return session.finish("year_reaction", term, citations)


def _earliest_reaction(session: _Session, spec: dict) -> Solution:
    years: dict[str, int] = {}
    apps: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    for drug in spec["drugs"]:
        output = session.call("search_approvals", {"generic_name": drug})
        year, chosen, node = _earliest_visible(output, drug)
        if year is None or node is None or chosen is None:
            raise Unsolvable(f"{drug} has no visible original approval")
        years[drug] = year
        apps[drug] = chosen
        nodes[drug] = node
    winner = _unique(spec["drugs"], years, least=True)
    if winner is None:
        raise Unsolvable("tied earliest approval")
    events = session.call("query_adverse_events", {"generic_name": winner, "mode": "top_reactions"})
    term = _top_term(events)
    citations = [
        _cite("drug/drugsfda", apps[winner], "submissions", nodes[winner]),
        _cite("drug/event", output_id(events), "results", events),
    ]
    return session.finish("earliest_reaction", term, citations)


def _sponsor_count(session: _Session, spec: dict) -> Solution:
    drug, citations = _match_sponsor(session, spec)
    events = session.call(
        "query_adverse_events",
        {"generic_name": drug, "mode": "count", "serious": True, "year": spec["year"]},
    )
    citations.append(_cite("drug/event", output_id(events), "meta.results.total", events))
    return session.finish("sponsor_count", _total(events), citations)


def _serious_counts(session: _Session, spec: dict) -> tuple[dict[str, int], dict[str, dict]]:
    counts: dict[str, int] = {}
    nodes: dict[str, dict] = {}
    for drug in spec["drugs"]:
        output = session.call(
            "query_adverse_events",
            {"generic_name": drug, "mode": "count", "serious": True, "year": spec["year"]},
        )
        counts[drug] = _total(output)
        nodes[drug] = output
    return counts, nodes


def _sponsor(session: _Session, spec: dict) -> Solution:
    drug, citations = _match_sponsor(session, spec)
    labels = session.call("search_drug_label", {"generic_name": drug, "has_section": "boxed_warning"})
    boxed = _first_boxed_row(labels)
    if boxed is None:
        raise Unsolvable(f"{drug} has no visible boxed warning")
    section = session.call("get_label_section", {"set_id": boxed["set_id"], "section": "boxed_warning"})
    text = section.get("boxed_warning")
    if not isinstance(text, str) or spec["keyword"] not in text.casefold():
        raise Unsolvable("boxed warning does not mention the keyword")
    citations.append(_cite("drug/label", section["set_id"], "boxed_warning", section))
    if spec["variant"] == "sponsor_drug":
        return session.finish("sponsor_drug", drug, citations)
    events = session.call("query_adverse_events", {"generic_name": drug, "mode": "top_reactions"})
    term = _top_term(events)
    citations.append(_cite("drug/event", output_id(events), "results", events))
    return session.finish("sponsor_faers", term, citations)


def _match_sponsor(session: _Session, spec: dict) -> tuple[str, list[dict]]:
    anchor = session.call("search_approvals", {"application_number": spec["application_number"]})
    rows = _rows(anchor)
    if len(rows) != 1 or not str(rows[0].get("sponsor_name") or "").strip():
        raise Unsolvable("anchor application did not show one sponsor")
    sponsor = str(rows[0]["sponsor_name"]).strip()
    key = sponsor.casefold()
    citations = [_cite("drug/drugsfda", rows[0]["application_number"], "sponsor_name", rows[0])]
    matches: list[tuple[str, dict]] = []
    for drug in spec["drugs"]:
        output = session.call("search_approvals", {"generic_name": drug})
        found = [row for row in _rows(output) if str(row.get("sponsor_name") or "").strip().casefold() == key]
        if found:
            matches.append((drug, found[0]))
    if len(matches) != 1:
        raise Unsolvable(f"{len(matches)} candidates share the sponsor")
    drug, row = matches[0]
    citations.append(_cite("drug/drugsfda", row["application_number"], "sponsor_name", row))
    return drug, citations


def _split_and(blob: str) -> list[str]:
    if ", and " not in blob:
        raise Unsolvable(blob)
    head, last = blob.rsplit(", and ", 1)
    names = [part.strip() for part in head.split(", ") if part.strip()]
    names.append(last.strip())
    if len(names) < 3 or any(not name for name in names):
        raise Unsolvable(blob)
    return names


def _split_or(blob: str) -> list[str]:
    if ", or " in blob:
        head, last = blob.rsplit(", or ", 1)
        names = [part.strip() for part in head.split(", ") if part.strip()]
        names.append(last.strip())
    elif " or " in blob:
        names = [part.strip() for part in blob.split(" or ") if part.strip()]
    else:
        raise Unsolvable(blob)
    if len(names) < 2:
        raise Unsolvable(blob)
    return names


def _rows(output: dict) -> list[dict]:
    return [row for row in output.get("results") or [] if isinstance(row, dict)]


def _total(output: dict) -> int:
    total = ((output.get("meta") or {}).get("results") or {}).get("total")
    if isinstance(total, bool) or not isinstance(total, int):
        raise Unsolvable("tool result has no total")
    return total


def _pages(output: dict) -> int | None:
    pages = ((output.get("meta") or {}).get("results") or {}).get("n_pages")
    return pages if isinstance(pages, int) else None


def output_id(output: dict) -> str:
    step = output.get("id")
    if not isinstance(step, str) or not step.startswith("step:"):
        raise Unsolvable("aggregate result has no step id")
    return step


def _earliest_visible(output: dict, drug: str) -> tuple[int | None, str | None, dict | None]:
    best: int | None = None
    chosen: str | None = None
    node: dict | None = None
    for row in _rows(output):
        number = str(row.get("application_number") or "")
        if not number.upper().startswith(("NDA", "BLA")):
            continue
        products = [item for item in row.get("products") or [] if isinstance(item, dict)]
        if not any(_product_is_single(product, drug) for product in products):
            continue
        year = _orig_year(row)
        if year is None:
            continue
        if best is None or year < best:
            best, chosen, node = year, number, row
    return best, chosen, node


def _first_boxed_row(output: dict) -> dict | None:
    for row in _rows(output):
        sections = row.get("sections") or []
        if row.get("set_id") and "boxed_warning" in sections:
            return row
    return None


def _term_count(output: dict, term: str) -> int | None:
    for row in _rows(output):
        if row.get("term") == term and isinstance(row.get("count"), int):
            return row["count"]
    return None


def _top_term(output: dict) -> str:
    rows = _rows(output)
    if not rows or not str(rows[0].get("term") or "").strip():
        raise Unsolvable("no FAERS reaction")
    if len(rows) > 1 and rows[0].get("count") == rows[1].get("count"):
        raise Unsolvable("tied FAERS top reaction")
    return str(rows[0]["term"])


def _unique(names: list[str], values: dict[str, int], *, least: bool) -> str | None:
    target = min(values[name] for name in names) if least else max(values[name] for name in names)
    winners = [name for name in names if values[name] == target]
    if len(winners) != 1:
        return None
    return winners[0]


def _cite(endpoint: str, cid: str, path: str, node: dict) -> dict:
    text = _render(_resolve(node, path))
    if not isinstance(text, str) or not text.strip():
        raise Unsolvable(f"{cid} {path} has no citable text")
    collapsed = _collapse(text)
    quote = collapsed if len(collapsed) <= 12 else collapsed[:120]
    if len(quote) > 500:
        quote = quote[:500]
    return {"endpoint": endpoint, "id": cid, "field": path, "quote": quote}
