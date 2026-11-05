"""Parse FDA-Gym question text. No env, rewards, tasks, or train imports."""

from __future__ import annotations

import re

_OF = re.compile(r"^Of (.+), which drug ")
_APP = re.compile(r"^Application ([A-Z]{3}\d{6}) is held by a sponsor")
_DRUGS_OF = re.compile(r"one of these drugs: (.+?)\.")
_KEYWORD = re.compile(r"boxed warning mentioning ([a-z]{6,20})\.")
_YEAR = re.compile(r"receiv(?:e|ed) in (\d{4})")
_TERM = re.compile(r"most FAERS reports of (.+)\?$")
_UNNAMED_SPONSOR = re.compile(
    r"^The sponsor of application ((?:NDA|BLA|ANDA)\d{6}) markets two drugs in the snapshot\. "
    r"Which of them has a boxed warning\?$"
)


class Unsolvable(Exception):
    """The question cannot be settled from the observations this solver is allowed to read."""


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
    """Train boxed_none prompt. Eval keeps ``question_none`` so that wording stays stable."""
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


def _listed(drugs: list[str]) -> str:
    if len(drugs) == 2:
        return f"{drugs[0]} or {drugs[1]}"
    if len(drugs) == 3:
        return f"{drugs[0]}, {drugs[1]}, or {drugs[2]}"
    raise Unsolvable("sponsor candidates must be 2 or 3 drugs")


def _year_of(question: str) -> int:
    year = _YEAR.search(question)
    if year is None:
        raise Unsolvable("missing FAERS year")
    return int(year.group(1))


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
