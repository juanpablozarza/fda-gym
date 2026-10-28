"""Generation of sponsor-hop questions over comparable drug sets.

This module owns candidate rotation and private-sponsor selection. The caller
injects task acceptance, keeping snapshot solving and task formatting in
``hard.py``.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Protocol

from fdagym.tasks.hard_solver import question_sponsor_count, question_sponsor_drug, question_sponsor_faers
from fdagym.tools import SECTION_LIMIT

FAERS_YEARS = (2022, 2023, 2024)
LABEL_RATIO = 3.0


class SponsorProfile(Protocol):
    name: str
    label_n: int
    sponsors: dict[str, list[str]]
    all_sponsors: dict[str, list[str]]
    boxed: bool
    boxed_text: str | None
    top: list[tuple[str, int]]
    counts: dict[int, int]

_KEYWORD_STOP = {
    "warning",
    "warnings",
    "patients",
    "patient",
    "including",
    "increased",
    "increase",
    "serious",
    "reported",
    "reports",
    "report",
    "reaction",
    "reactions",
    "adverse",
    "treatment",
    "treated",
    "prescribing",
    "information",
    "healthcare",
    "provider",
    "should",
    "which",
    "their",
    "there",
    "these",
    "those",
    "about",
    "after",
    "before",
    "during",
    "within",
    "without",
    "other",
    "using",
    "clinical",
    "studies",
    "boxed",
    "black",
    "product",
    "products",
    "therapy",
    "history",
    "known",
    "complete",
    "discontinue",
    "monitor",
    "associated",
    "because",
    "however",
    "following",
    "recommended",
    "contraindicated",
    "hypersensitivity",
    "administration",
    "sponsor",
    "application",
    "markets",
    "mentioning",
    "original",
    "approval",
    "earliest",
    "received",
    "label",
    "labels",
    "drug",
    "drugs",
    "faers",
    "none",
    "answer",
    "fatal",
    "death",
    "deaths",
    "cause",
    "causes",
    "risk",
    "risks",
    "effects",
    "effect",
    "dependent",
    "duration",
    "secondary",
    "indicated",
    "administered",
    "concomitant",
    "requiring",
    "premature",
    "interval",
    "diminished",
    "receiving",
    "carefully",
    "potential",
    "possible",
    "physician",
    "physicians",
    "symptoms",
    "symptom",
    "disease",
    "diseases",
    "condition",
    "conditions",
    "medication",
    "medications",
    "observed",
    "frequency",
    "frequent",
    "results",
    "another",
    "monitoring",
    "threatening",
    "hydrochloride",
    "sulfate",
    "citrate",
    "tablets",
    "tablet",
    "capsules",
    "capsule",
    "transdermal",
    "injection",
    "solution",
    "sodium",
    "potassium",
    "section",
    "system",
}

_SPONSOR_WIDTH = {"sponsor_faers": 2, "sponsor_drug": 3, "sponsor_count": 3}
_OR_TERM = re.compile(r"\bor\b", re.IGNORECASE)


def _profile_top_ok(profile: SponsorProfile) -> bool:
    if not profile.top:
        return False
    if len(profile.top) > 1 and profile.top[0][1] == profile.top[1][1]:
        return False
    return _OR_TERM.search(profile.top[0][0]) is None


def _labels_within(left: SponsorProfile, right: SponsorProfile, ratio: float = LABEL_RATIO) -> bool:
    """True when two label counts are within ``ratio`` of each other."""
    a = max(left.label_n, 1)
    b = max(right.label_n, 1)
    return max(a, b) / min(a, b) <= ratio


def _sponsor_pool(profiles: list[SponsorProfile], variant: str) -> list[SponsorProfile]:
    picked = []
    for profile in profiles:
        if variant == "sponsor_count":
            if any(year not in profile.counts for year in FAERS_YEARS):
                continue
        else:
            if not profile.boxed or not profile.boxed_text:
                continue
            if variant == "sponsor_faers" and not _profile_top_ok(profile):
                continue
        if not profile.sponsors:
            continue
        picked.append(profile)
    return picked


def _sponsor_index(profiles: list[SponsorProfile], blocked_sponsors: set[str], blocked_apps: set[str]) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for profile in profiles:
        for key, apps in profile.sponsors.items():
            if key in blocked_sponsors:
                continue
            for app in apps:
                if app in blocked_apps or not app.startswith(("NDA", "BLA")):
                    continue
                index[key].append((profile.name, app))
    return index


def _anchor_options(
    profile: SponsorProfile,
    others: list[SponsorProfile],
    index: dict[str, list[tuple[str, str]]],
    used_anchor: Counter,
    anchor_cap: int,
) -> list[str]:
    """Applications whose sponsor is on ``profile`` and on none of the other candidates."""
    blocked: set[str] = set()
    for other in others:
        blocked |= _sponsor_keys(other)
    names = {profile.name, *(other.name for other in others)}
    found: list[str] = []
    seen: set[str] = set()
    for key in profile.sponsors:
        if key in blocked:
            continue
        for drug, app in index.get(key, []):
            if drug in names or used_anchor[app] >= anchor_cap or app in seen:
                continue
            seen.add(app)
            found.append(app)
    return found


def _keyword_ready(members: list[SponsorProfile]) -> list[SponsorProfile]:
    """Members whose boxed warning shares a word with another member of the set."""
    names = [item.name for item in members]
    ready = []
    for profile in members:
        others = [other for other in members if other.name != profile.name]
        texts = [other.boxed_text or "" for other in others]
        if _shared_keyword(profile.boxed_text or "", texts, names) is not None:
            ready.append(profile)
    return ready


def _set_is_symmetric(
    members: list[SponsorProfile],
    index: dict[str, list[tuple[str, str]]],
    used_anchor: Counter,
    anchor_cap: int,
    *,
    need_keyword: bool,
    keyword_mode: str = "all",
) -> bool:
    """Every member has a private sponsor. ``all`` also requires a shared boxed word on each.

    ``pair`` is the training relaxation: at least two members share a boxed word, and
    the answer rotates only among those. The printed keyword still has to hit a distractor.
    """
    for profile in members:
        others = [other for other in members if other.name != profile.name]
        if not _anchor_options(profile, others, index, used_anchor, anchor_cap):
            return False
    if not need_keyword:
        return True
    ready = _keyword_ready(members)
    if keyword_mode == "pair":
        return len(ready) >= min(2, len(members))
    return len(ready) == len(members)


def _sample_comparable(
    pool: list[SponsorProfile], k: int, rng: random.Random, *, ratio: float = LABEL_RATIO
) -> list[SponsorProfile] | None:
    if len(pool) < k:
        return None
    for _ in range(12):
        seed = rng.choice(pool)
        band = [profile for profile in pool if _labels_within(seed, profile, ratio)]
        if len(band) < k:
            continue
        picked = rng.sample(band, k)
        counts = [max(profile.label_n, 1) for profile in picked]
        if max(counts) / min(counts) <= ratio:
            return picked
    return None


def _discover_sets(
    pool: list[SponsorProfile],
    k: int,
    rng: random.Random,
    index: dict[str, list[tuple[str, str]]],
    *,
    want: int,
    need_keyword: bool,
    anchor_cap: int,
    prefer: set[str],
    dropped: Counter,
    skip: set[frozenset[str]] | None = None,
    keyword_mode: str = "all",
) -> list[list[SponsorProfile]]:
    found: list[list[SponsorProfile]] = []
    seen: set[frozenset[str]] = set(skip or ())

    def search(source: list[SponsorProfile], tries: int, limit: int) -> None:
        if len(source) < k:
            return
        for _ in range(tries):
            if len(found) >= limit:
                return
            members = _sample_comparable(source, k, rng)
            if members is None:
                continue
            key = frozenset(profile.name for profile in members)
            if key in seen:
                continue
            seen.add(key)
            if not _set_is_symmetric(
                members, index, Counter(), anchor_cap, need_keyword=need_keyword, keyword_mode=keyword_mode
            ):
                dropped["not_symmetric"] += 1
                continue
            found.append(members)

    preferred = [profile for profile in pool if profile.name in prefer] if prefer else []
    if len(preferred) >= k:
        search(preferred, max(200, want * 20), max(4, want // 3))
    if len(found) < want:
        search(pool, max(600, want * 50), want)
    return found


def _next_slot(counts: list[int], rng: random.Random) -> int:
    least = min(counts)
    options = [index for index, count in enumerate(counts) if count == least]
    return rng.choice(options)


def _insert_slot(members: list[SponsorProfile], answer: str, slot: int, rng: random.Random) -> list[str]:
    others = [profile.name for profile in members if profile.name != answer]
    rng.shuffle(others)
    names = list(others)
    names.insert(slot, answer)
    return names


def fill_sponsor_tasks(
    profiles: list[SponsorProfile],
    rng: random.Random,
    *,
    quotas: dict[str, int],
    dropped: Counter,
    accept,
    named_cap: int,
    answer_cap: int,
    anchor_cap: int,
    blocked_sponsors: set[str] | None = None,
    blocked_apps: set[str] | None = None,
    prefer: set[str] | None = None,
    seen: set[str] | None = None,
    keyword_mode: str = "all",
) -> dict[str, list[dict]]:
    """Sponsor questions whose answer is a rotation over a comparable drug set.

    Distractors are a random draw from drugs with a similar label count, not one
    rarer name and one more famous name. Eval keeps a set only when every member
    could have been the answer. Training (``keyword_mode="pair"``) still requires
    a private sponsor on every member, and a shared boxed word on at least two,
    then rotates the answer among those two or three. The printed slot rotates
    across the list. Label rank, tie structure, and alphabetical order are not
    how the answer is chosen.
    """
    blocked_sponsors = blocked_sponsors or set()
    blocked_apps = blocked_apps or set()
    prefer = prefer or set()
    seen = seen if seen is not None else set()
    index = _sponsor_index(profiles, blocked_sponsors, blocked_apps)
    keyword_used: Counter[str] = Counter()
    used_named: Counter[str] = Counter()
    used_answer: Counter[str] = Counter()
    used_anchor: Counter[str] = Counter()
    used_sig: set[tuple] = set()
    groups: dict[str, list[dict]] = {}
    for variant, quota in quotas.items():
        if quota <= 0:
            continue
        width = _SPONSOR_WIDTH[variant]
        pool = _sponsor_pool(profiles, variant)
        need_keyword = variant != "sponsor_count"
        kept: list[dict] = []
        slots = [0] * width
        known: set[frozenset[str]] = set()
        good: list[list[SponsorProfile]] = []
        while len(kept) < quota:
            fresh = _discover_sets(
                [profile for profile in pool if used_named[profile.name] < named_cap and used_answer[profile.name] < answer_cap],
                width,
                rng,
                index,
                want=max(12, (quota - len(kept)) * 2),
                need_keyword=need_keyword,
                anchor_cap=anchor_cap,
                prefer=prefer if not kept else set(),
                dropped=dropped,
                skip=known,
                keyword_mode=keyword_mode,
            )
            added = [members for members in fresh if frozenset(profile.name for profile in members) not in known]
            if not added:
                break
            for members in added:
                known.add(frozenset(profile.name for profile in members))
            good = added
            rng.shuffle(good)
            stall = 0
            cursor = 0
            guard = 0
            produced = len(kept)
            while good and len(kept) < quota and stall < len(good) * 3 and guard < max(40, quota * 20):
                guard += 1
                members = good[cursor % len(good)]
                cursor += 1
                if any(
                    used_named[profile.name] >= named_cap or used_answer[profile.name] >= answer_cap
                    for profile in members
                ):
                    stall += 1
                    continue
                eligible = _keyword_ready(members) if need_keyword and keyword_mode == "pair" else members
                if need_keyword and keyword_mode == "pair" and len(eligible) < 2:
                    stall += 1
                    continue
                low = min(used_answer[profile.name] for profile in eligible)
                choices = [profile for profile in eligible if used_answer[profile.name] == low]
                answer = rng.choice(choices)
                others = [profile for profile in members if profile.name != answer.name]
                anchors = _anchor_options(answer, others, index, used_anchor, anchor_cap)
                rng.shuffle(anchors)
                task = None
                chosen_app = None
                for app in anchors[:4]:
                    year = None
                    keyword = None
                    slot = _next_slot(slots, rng)
                    names = _insert_slot(members, answer.name, slot, rng)
                    if variant == "sponsor_count":
                        year = FAERS_YEARS[len(kept) % len(FAERS_YEARS)]
                        question = question_sponsor_count(app, names, year)
                    else:
                        texts = [profile.boxed_text or "" for profile in others]
                        keyword = _shared_keyword(answer.boxed_text or "", texts, names, keyword_used)
                        if keyword is None:
                            continue
                        if variant == "sponsor_faers":
                            question = question_sponsor_faers(app, names, keyword)
                        else:
                            question = question_sponsor_drug(app, names, keyword)
                    sig = (variant, app, frozenset(names), year, keyword)
                    if question in seen or sig in used_sig:
                        continue
                    built = accept(question, variant)
                    if built is None:
                        continue
                    if built["entity"]["generic_name"] != answer.name:
                        dropped["entity"] += 1
                        continue
                    task = built
                    chosen_app = app
                    slots[slot] += 1
                    seen.add(question)
                    used_sig.add(sig)
                    break
                if task is None or chosen_app is None:
                    stall += 1
                    continue
                stall = 0
                used_anchor[chosen_app] += 1
                used_answer[answer.name] += 1
                for profile in members:
                    used_named[profile.name] += 1
                kept.append(task)
            if len(kept) == produced:
                break
        groups[variant] = kept
        if len(kept) < quota:
            dropped[f"short:{variant}"] += quota - len(kept)
    return groups

def _sponsor_keys(profile: SponsorProfile) -> set[str]:
    """Every sponsor on the snapshot, not only page 1. Hidden later-page matches are ties."""
    return set(profile.all_sponsors or profile.sponsors)

def _keyword_ranked(text: str, drugs: list[str]) -> list[str]:
    """Whole words from the warning headline, not hyphen fragments or boilerplate."""
    banned = set(_KEYWORD_STOP)
    for drug in drugs:
        for token in re.split(r"[^a-z0-9]+", drug.casefold()):
            if token:
                banned.add(token)
    folded = text.casefold()
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"[a-z]{6,20}", folded):
        start, end = match.span()
        if start and folded[start - 1] == "-":
            continue
        if end < len(folded) and folded[end] == "-":
            continue
        if len(folded) >= SECTION_LIMIT - 5 and end >= len(folded) - 1:
            continue
        word = match.group(0)
        if word in banned:
            continue
        found.append((start, word))
    if not found:
        return []
    headline = [word for start, word in found if start < 280]
    pool = headline or [word for _start, word in found]
    order: dict[str, int] = {}
    for start, word in found:
        if word in pool and word not in order:
            order[word] = start
    return sorted(order, key=lambda word: (order[word], -len(word)))


def _keyword(text: str, drugs: list[str], used: Counter | None = None) -> str | None:
    """A whole word from the warning headline, not a hyphen fragment or boilerplate."""
    ranked = _keyword_ranked(text, drugs)
    if not ranked:
        return None
    counts = used or Counter()
    ranked = sorted(ranked, key=lambda word: (counts[word], ranked.index(word)))
    choice = ranked[0]
    counts[choice] += 1
    return choice


def _shared_keyword(
    answer_text: str, other_texts: list[str], names: list[str], used: Counter | None = None
) -> str | None:
    """A headline word from the answer that also appears in another named drug's warning."""
    ranked = _keyword_ranked(answer_text, names)
    if not ranked:
        return None
    blobs = [(text or "")[:SECTION_LIMIT].casefold() for text in other_texts]
    counts = used or Counter()
    ranked = sorted(ranked, key=lambda word: (counts[word], ranked.index(word)))
    for word in ranked:
        if any(word in blob for blob in blobs if blob):
            counts[word] += 1
            return word
    return None
