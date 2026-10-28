"""Boxed-warning topics. Headings come from label text; synonyms attach common wordings.

`build_topics` scans a Snapshot so the vocabulary can be regenerated when the full
snapshot exists. Scoring uses the checked-in file, which also keeps the headings JP
named (bleeding risk, lactic acidosis, neuropsychiatric events, hepatotoxicity) and
the titles the stuffing check needs.
"""

import json
import re
from pathlib import Path

from fdagym.queries import label_search
from fdagym.snapshot import Snapshot, SnapshotMiss

MAX_TOPICS = 2
TOPICS_PATH = Path(__file__).with_name("boxed_topics.json")

# Key is matched inside a heading. Values are extra ways an answer can name that topic.
_LEXICON = (
    ("bleeding risk", ("bleeding", "hemorrhage", "haemorrhage")),
    ("lactic acidosis", ("lactic acidosis",)),
    ("neuropsychiatric", ("neuropsychiatric",)),
    ("hepatotoxicity", ("hepatotoxicity", "liver", "hepatic")),
    ("mortality", ("death", "mortality")),
    ("malignancy", ("cancer", "malignancy")),
    ("stroke", ("stroke",)),
    ("myocardial infarction", ("infarct", "infarction", "myocardial infarction")),
)

# Kept until a full snapshot scan replaces this file. Snapshot headings are added on top.
SEED_HEADINGS = (
    "BLEEDING RISK",
    "LACTIC ACIDOSIS",
    "SERIOUS NEUROPSYCHIATRIC EVENTS",
    "HEPATOTOXICITY",
    "INCREASED MORTALITY IN ELDERLY PATIENTS WITH DEMENTIA-RELATED PSYCHOSIS",
    "SERIOUS INFECTIONS AND MALIGNANCY",
    "STROKE",
    "MYOCARDIAL INFARCTION",
)

_HEADING = re.compile(r"WARNING:\s+([A-Z][A-Z0-9]+(?:\s+[A-Z][A-Z0-9]+){0,12})")


def build_topics(snapshot: Snapshot, generic_names: list[str]) -> list[dict]:
    found: list[str] = []
    for name in generic_names:
        query = label_search(name)
        try:
            body = snapshot.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
        except SnapshotMiss:
            continue
        for record in body.get("results") or []:
            if not isinstance(record, dict):
                continue
            found.extend(_headings(record.get("boxed_warning")))
    headings = list(dict.fromkeys([*found, *SEED_HEADINGS]))
    return [_topic(heading) for heading in headings]


def load_topics() -> list[dict]:
    return json.loads(TOPICS_PATH.read_text())


def topics_named(answer: str, topics: list[dict] | None = None) -> list[str]:
    text = answer.lower()
    named = []
    for topic in topics if topics is not None else load_topics():
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in topic["terms"]):
            named.append(topic["heading"])
    return named


def _headings(value: object) -> list[str]:
    texts = value if isinstance(value, list) else [value]
    found = []
    for text in texts:
        if isinstance(text, str):
            found.extend(match.group(1).strip() for match in _HEADING.finditer(text))
    return found


def _topic(heading: str) -> dict:
    low = heading.lower()
    terms = [low]
    for key, synonyms in _LEXICON:
        if key in low:
            for synonym in synonyms:
                if synonym not in terms:
                    terms.append(synonym)
    return {"heading": heading, "terms": terms}
