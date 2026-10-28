"""Hard-split tasks. Eval only: callers must not fold these into train, val, or test.

Two templates, both answered from the frozen snapshot:

* ``compare`` — three named drugs, one fact that is unique across them.
* ``sponsor_hop`` — an application number, then a sponsor match among named candidates,
  then that drug's label and (for most items) its top FAERS reaction.

``search_approvals`` cannot filter by sponsor, so the sponsor question names the
candidate drugs. The reference solver in ``hard_solver`` is what decides that a
question is reachable; this module only proposes candidates and keeps the ones
the solver settles inside 6 calls.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field

from fdagym.approvals import find_applications
from fdagym.env import FDAEnv
from fdagym.queries import (
    approvals_search,
    faers_report_count,
    faers_top_reactions,
    label_search,
)
from fdagym.tasks.disjoint import holdout_entities, task_entities
from fdagym.tasks.generators import _make, _orig_year, _product_is_single
from fdagym.tasks.hard_solver import (
    BUDGET,
    Unsolvable,
    _first_boxed_row,
    _pages,
    _rows,
    _total,
    parse_question,
    question_earliest,
    question_faers_year,
    question_none,
    question_reaction,
    run_call,
    same_answer,
    solve,
)
from fdagym.tasks.hard_sponsor import FAERS_YEARS, fill_sponsor_tasks
from fdagym.tasks.meddra import aliases
from fdagym.tasks.plan import load_universe
from fdagym.tools import SECTION_LIMIT, EpisodeState
from fdagym.tools import _section_text as tool_section_text

HARD_SEED = 0
ID_START = 8001
PUBLIC_SEED = 1
PUBLIC_ID_START = 9101
_CANON = re.compile(r"^(NDA|BLA|ANDA)\d{6}$")
_QUOTAS = {
    "earliest_boxed": 16,
    "faers_year": 12,
    "faers_reaction": 18,
    "boxed_none": 4,
    "sponsor_faers": 25,
    "sponsor_drug": 25,
}
# Max times a name is listed in an eval sponsor question. The answer cap stops
# one drug from being the solution of every question it appears in.
NAMED_CAP = 5
ANSWER_CAP = 5
# Candidate drugs stay in the same order of magnitude. The answer is then
# rotated across that set, so label rank is not a selection rule.


@dataclass
class Profile:
    name: str
    year: int | None = None
    pages: int = 0
    one_page: bool = False
    label_n: int = 0
    sponsors: dict[str, list[str]] = field(default_factory=dict)
    all_sponsors: dict[str, list[str]] = field(default_factory=dict)
    boxed: bool = False
    boxed_text: str | None = None
    top: list[tuple[str, int]] = field(default_factory=list)
    counts: dict[int, int] = field(default_factory=dict)
    faers_all: int = 0
    label_ok: bool = False


@dataclass
class _Cache:
    snapshot: object
    bodies: dict = field(default_factory=dict)

    def query(self, query):
        key = (query.endpoint, query.search, query.count, query.limit)
        if key not in self.bodies:
            self.bodies[key] = self.snapshot.get(
                query.endpoint, search=query.search, count=query.count, limit=query.limit
            )
        return self.bodies[key]


def sponsor_surface_caches(snapshot, tasks: list[dict]) -> tuple[dict[str, int], dict[str, str]]:
    """Build the label-count and boxed-text lookups used by sponsor leak audits."""
    from fdagym.tasks.leaks import iter_sponsor

    cache = _Cache(snapshot)
    wanted: dict[str, str] = {}
    for _task, _variant, names, _answer, spec in iter_sponsor(tasks):
        for drug in spec["drugs"]:
            wanted[drug.casefold()] = drug
        for name in names:
            wanted.setdefault(name, name)

    label_counts: dict[str, int] = {}
    boxed_texts: dict[str, str] = {}
    for key, drug in wanted.items():
        profile = _profile(snapshot, cache, drug)
        label_counts[key] = profile.label_n
        boxed_texts[key] = (profile.boxed_text or "")[:SECTION_LIMIT].casefold()
    return label_counts, boxed_texts


def _eval_profiles(profiles: list[Profile], rng, *, hold_back: int = 40) -> list[Profile]:
    """Keep some names out of the eval sheet so training still has a drug pool."""
    order = list(profiles)
    rng.shuffle(order)
    if hold_back <= 0 or hold_back >= len(order):
        return list(profiles)
    return order[hold_back:]


def generate_hard(
    snapshot,
    *,
    seed: int = HARD_SEED,
    id_start: int = ID_START,
    split: str = "hard",
    holdout_tasks: list[dict] | None = None,
    hold_back: int = 40,
) -> tuple[list[dict], dict]:
    """Build one hard-style sheet. Every kept question was re-solved from the question text alone.

    ``holdout_tasks`` drops drugs, applications, and sponsors that already appear on another
    sheet, so a public sheet can be checked against the private hard eval.
    """
    cache = _Cache(snapshot)
    rng = random.Random(seed)
    held = holdout_entities(holdout_tasks, snapshot) if holdout_tasks else None
    profiles = []
    for drug in load_universe():
        if held is not None and drug.casefold() in held.drugs:
            continue
        profiles.append(_profile(snapshot, cache, drug))
    profiles = _eval_profiles(profiles, rng, hold_back=hold_back)
    dropped: Counter[str] = Counter()
    raw: list[tuple[str, dict]] = []
    compare = _compare_tasks(snapshot, cache, profiles, rng, dropped, split=split, holdout=held)
    raw.extend(compare)
    preferred = set()
    for _variant, task in compare:
        preferred.update(parse_question(task["question"])["drugs"])
    built = fill_sponsor_tasks(
        profiles,
        rng,
        quotas={"sponsor_faers": _QUOTAS["sponsor_faers"], "sponsor_drug": _QUOTAS["sponsor_drug"]},
        dropped=dropped,
        accept=lambda question, variant: _accept(snapshot, cache, question, variant, dropped, split=split, holdout=held),
        named_cap=NAMED_CAP,
        answer_cap=ANSWER_CAP,
        anchor_cap=2,
        prefer=preferred,
    )
    for variant, tasks in built.items():
        raw.extend((variant, task) for task in tasks)
    tasks = _assign_ids([task for _variant, task in raw], id_start)
    report = {
        "seed": seed,
        "split": split,
        "id_start": id_start,
        "n": len(tasks),
        "by_variant": dict(Counter(variant for variant, _task in raw)),
        "by_type": dict(Counter(task["type"] for task in tasks)),
        "dropped": dict(dropped),
        "pool": len(profiles),
    }
    return tasks, report


def snapshot_answer(snapshot, question: str):
    """Gold recomputed from snapshot bodies, not from the solver's tool trace."""
    return _snapshot_answer(_Cache(snapshot), question)


def _assign_ids(tasks: list[dict], id_start: int = ID_START) -> list[dict]:
    ordered = sorted(tasks, key=lambda task: (task["type"], task["question"]))
    tagged = []
    for offset, task in enumerate(ordered):
        tagged.append({**task, "id": f"FG-{id_start + offset:04d}"})
    return tagged


def _compare_tasks(snapshot, cache, profiles, rng, dropped, *, split: str = "hard", holdout=None) -> list[tuple[str, dict]]:
    kept: list[tuple[str, dict]] = []
    used: Counter[str] = Counter()
    seen: set[frozenset[str]] = set()
    earliest = [profile for profile in profiles if profile.year is not None]
    counted = [profile for profile in profiles if all(year in profile.counts for year in FAERS_YEARS)]
    reacted = [profile for profile in profiles if profile.top]
    quiet = [profile for profile in profiles if profile.label_ok and not profile.boxed]
    kept.extend(
        _sample_trios(
            snapshot,
            cache,
            earliest,
            rng,
            used,
            seen,
            dropped,
            "earliest_boxed",
            _QUOTAS["earliest_boxed"],
            lambda trio: question_earliest([profile.name for profile in trio]),
            ok=_needs_year_compare,
            split=split,
            holdout=holdout,
        )
    )
    for year in FAERS_YEARS:
        quota = _QUOTAS["faers_year"] // len(FAERS_YEARS)
        kept.extend(
            _sample_trios(
                snapshot,
                cache,
                counted,
                rng,
                used,
                seen,
                dropped,
                "faers_year",
                quota,
                lambda trio, year=year: question_faers_year([profile.name for profile in trio], year),
                rank_of=lambda trio, year=year: _faers_year_rank(trio, year),
                split=split,
                holdout=holdout,
            )
        )
    kept.extend(
        _sample_reaction(
            snapshot, cache, reacted, rng, used, seen, dropped, _QUOTAS["faers_reaction"], split=split, holdout=holdout
        )
    )
    kept.extend(
        _sample_trios(
            snapshot,
            cache,
            quiet,
            rng,
            Counter(),
            seen,
            dropped,
            "boxed_none",
            _QUOTAS["boxed_none"],
            lambda trio: question_none([profile.name for profile in trio]),
            cap=2,
            split=split,
            holdout=holdout,
        )
    )
    return kept


def _reuse_trio(pool: list[Profile], rng, used: Counter, cap: int) -> list[Profile] | None:
    """Keep two already-named drugs and add one new one, so holdout stays small."""
    open_pool = [profile for profile in pool if used[profile.name] < cap]
    if len(open_pool) < 3:
        return None
    reused = [profile for profile in open_pool if used[profile.name] > 0]
    fresh = [profile for profile in open_pool if used[profile.name] == 0]
    if len(reused) >= 2 and fresh:
        return [*rng.sample(reused, 2), rng.choice(fresh)]
    if reused and len(fresh) >= 3 - len(reused):
        return [*reused, *rng.sample(fresh, 3 - len(reused))]
    if len(reused) >= 3:
        return rng.sample(reused, 3)
    return rng.sample(open_pool, 3)


def _needs_year_compare(trio: list[Profile]) -> bool:
    """Reject trios where one boxed drug is already the answer without years."""
    return sum(1 for profile in trio if profile.boxed) >= 2


def _faers_year_rank(trio: list[Profile], year: int) -> str:
    """All-years FAERS rank of the drug that wins this year's serious count."""
    values = [profile.counts[year] for profile in trio]
    winner = trio[values.index(max(values))]
    return _unfiltered_faers_rank(winner, [profile for profile in trio if profile.name != winner.name])


def _sample_trios(
    snapshot,
    cache,
    pool,
    rng,
    used,
    seen,
    dropped,
    variant,
    quota,
    question_of,
    cap=3,
    ok=None,
    rank_of=None,
    split="hard",
    holdout=None,
):
    kept = []
    rank_used: Counter[str] = Counter()
    rank_cap = (quota + 1) // 2
    if len(pool) < 3 or quota <= 0:
        return kept
    attempts = 0
    limit = max(8000, quota * 800)
    while len(kept) < quota and attempts < limit:
        attempts += 1
        trio = _reuse_trio(pool, rng, used, cap)
        if trio is None:
            continue
        names = [profile.name for profile in trio]
        if any(used[name] >= cap for name in names):
            continue
        key = frozenset(names)
        if key in seen:
            continue
        if ok is not None and not ok(trio):
            dropped[f"skip:{variant}"] += 1
            continue
        rank = rank_of(trio) if rank_of is not None else None
        if rank is not None and rank_used[rank] >= rank_cap:
            dropped[f"faers_rank:{variant}:{rank}"] += 1
            continue
        order = list(trio)
        rng.shuffle(order)
        question = question_of(order)
        task = _accept(snapshot, cache, question, variant, dropped, split=split, holdout=holdout)
        if task is None:
            continue
        seen.add(key)
        if rank is not None:
            rank_used[rank] += 1
        for name in names:
            used[name] += 1
        kept.append((variant, task))
    if len(kept) < quota:
        dropped[f"short:{variant}"] += quota - len(kept)
    return kept


def _sample_reaction(snapshot, cache, pool, rng, used, seen, dropped, quota, *, split="hard", holdout=None):
    kept = []
    term_used: Counter[str] = Counter()
    rank_used: Counter[str] = Counter()
    rank_cap = (quota + 1) // 2
    attempts = 0
    while len(kept) < quota and attempts < quota * 2000:
        attempts += 1
        trio = _reuse_trio(pool, rng, used, 3)
        if trio is None:
            continue
        names = [profile.name for profile in trio]
        if any(used[name] >= 3 for name in names) or frozenset(names) in seen:
            continue
        term = _shared_term(trio, term_used)
        if term is None:
            dropped["no_shared_reaction"] += 1
            continue
        winner = max(trio, key=lambda profile: dict(profile.top).get(term, -1))
        rank = _unfiltered_faers_rank(winner, [profile for profile in trio if profile.name != winner.name])
        if rank_used[rank] >= rank_cap:
            dropped[f"faers_rank:faers_reaction:{rank}"] += 1
            continue
        order = list(trio)
        rng.shuffle(order)
        question = question_reaction([profile.name for profile in order], term)
        task = _accept(snapshot, cache, question, "faers_reaction", dropped, split=split, holdout=holdout)
        if task is None:
            continue
        seen.add(frozenset(names))
        term_used[term] += 1
        rank_used[rank] += 1
        for name in names:
            used[name] += 1
        kept.append(("faers_reaction", task))
    if len(kept) < quota:
        dropped["short:faers_reaction"] += quota - len(kept)
    return kept


def _shared_term(trio: list[Profile], term_used: Counter[str]) -> str | None:
    bags = [{term: count for term, count in profile.top} for profile in trio]
    common = set(bags[0]) & set(bags[1]) & set(bags[2])
    ranked = []
    for term in common:
        counts = [bag[term] for bag in bags]
        if counts.count(max(counts)) != 1:
            continue
        if re.search(r"\bor\b", term, re.IGNORECASE):
            continue
        gap = max(counts) - sorted(counts)[-2]
        ranked.append((term_used[term], -gap, term))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]




def _reuses_holdout(task: dict, holdout, snapshot) -> bool:
    found = task_entities(task, snapshot)
    return bool(
        set(found.drugs) & set(holdout.drugs)
        or set(found.applications) & set(holdout.applications)
        or set(found.sponsors) & set(holdout.sponsors)
    )


def _accept(
    snapshot, cache, question, variant, dropped, *, budget: int = BUDGET, split: str = "hard", holdout=None
) -> dict | None:
    try:
        gold = _snapshot_answer(cache, question)
        solution = solve(snapshot, question, budget=budget)
    except (Unsolvable, KeyError, ValueError) as exc:
        dropped[f"reject:{variant}:{type(exc).__name__}"] += 1
        return None
    if not same_answer(solution.answer, gold):
        dropped[f"mismatch:{variant}"] += 1
        return None
    if solution.n_calls > budget or any(length > 3500 for length in solution.observation_chars):
        dropped[f"budget:{variant}"] += 1
        return None
    try:
        task = _task(cache, question, variant, gold, solution)
    except (Unsolvable, ValueError) as exc:
        dropped[f"task:{variant}:{type(exc).__name__}"] += 1
        return None
    task["id"] = "FG-8000"
    task["split"] = split
    if holdout is not None and _reuses_holdout(task, holdout, snapshot):
        dropped[f"holdout:{variant}"] += 1
        return None
    reward = _reward(snapshot, task, solution)
    if reward["total"] < 1.0 - 1e-9:
        dropped[f"score:{variant}"] += 1
        if dropped[f"score:{variant}"] <= 3:
            dropped[f"score-note:{variant}"] = 1
            print(variant, question[:140], reward["notes"][:6])
        return None
    task.pop("id")
    return task


def _task(cache, question, variant, gold, solution) -> dict:
    spec = parse_question(question)
    if variant == "boxed_none":
        entity = spec["drugs"][0]
        gold_body = {"kind": "boolean", "value": False}
        sources = ["drug/label"]
    elif variant == "sponsor_faers":
        entity = _sponsor_entity(cache, spec)
        gold_body = {"kind": "exact", "value": gold}
        also = aliases(str(gold))
        if also:
            gold_body["aliases"] = also
        sources = ["drug/drugsfda", "drug/label", "drug/event"]
    elif variant == "sponsor_drug":
        entity = str(gold)
        gold_body = {"kind": "exact", "value": gold}
        sources = ["drug/drugsfda", "drug/label"]
    elif variant in {"year_reaction", "earliest_reaction"}:
        entity = _event_drug(solution)
        gold_body = {"kind": "exact", "value": gold}
        also = aliases(str(gold))
        if also:
            gold_body["aliases"] = also
        sources = ["drug/event"] if variant == "year_reaction" else ["drug/drugsfda", "drug/event"]
    elif variant == "sponsor_count":
        entity = _sponsor_entity(cache, spec)
        gold_body = {"kind": "number", "value": int(gold)}
        sources = ["drug/drugsfda", "drug/event"]
    else:
        entity = str(gold)
        gold_body = {"kind": "exact", "value": gold}
        sources = {
            "earliest_boxed": ["drug/drugsfda", "drug/label"],
            "faers_year": ["drug/event"],
            "faers_reaction": ["drug/event"],
        }[variant]
    citations = _gold_citations(cache, spec, solution)
    return _make(
        type="sponsor_hop" if variant.startswith("sponsor") else "compare",
        question=question,
        sources=sources,
        drug=entity,
        brand=None,
        version=cache.snapshot.version,
        answerable=True,
        gold=gold_body,
        citations=citations,
        oracle_calls=solution.calls,
        source_count=len(sources),
        needs_disambiguation=True,
        needs_aggregation=variant in {
            "earliest_boxed",
            "faers_year",
            "faers_reaction",
            "sponsor_faers",
            "year_reaction",
            "earliest_reaction",
            "sponsor_count",
        },
        needs_normalization=variant in {"sponsor_faers", "year_reaction", "earliest_reaction"},
        has_constraint=True,
    )


def _gold_citations(cache, spec, solution) -> list[dict]:
    extras: dict[tuple[str, str], list[str]] = {}
    variant = spec["variant"]
    if variant == "earliest_boxed":
        winner = solution.answer
        extras[("drug/drugsfda", "submissions")] = _year_apps(cache, winner)
        extras[("drug/label", "sections")] = _boxed_ids(cache, winner)
    elif variant == "earliest_reaction":
        extras[("drug/drugsfda", "submissions")] = _year_apps(cache, _event_drug(solution))
    elif variant == "boxed_none":
        ids: list[str] = []
        for drug in spec["drugs"]:
            ids.extend(_all_set_ids(cache, drug))
        extras[("drug/label", "sections")] = ids
    elif variant in {"sponsor_faers", "sponsor_drug", "sponsor_count"}:
        entity = _sponsor_entity(cache, spec)
        extras[("drug/drugsfda", "sponsor_name")] = _sponsor_apps(cache, spec, entity)
        if variant != "sponsor_count":
            extras[("drug/label", "boxed_warning")] = _keyword_ids(cache, entity, spec["keyword"])
    grouped: dict[tuple[str, str], list[str] | None] = {}
    for cite in solution.citations:
        key = (cite["endpoint"], cite["field"])
        if str(cite["id"]).startswith("step:"):
            grouped[key] = None
            continue
        bucket = grouped.get(key)
        if bucket is None and key in grouped:
            continue
        grouped.setdefault(key, [])
        if cite["id"] not in grouped[key]:
            grouped[key].append(cite["id"])
    spec_rows = []
    for (endpoint, path), ids in grouped.items():
        if ids is None:
            spec_rows.append({"endpoint": endpoint, "ids": [], "fields": [path]})
            continue
        for extra in extras.get((endpoint, path), []):
            if extra not in ids:
                ids.append(extra)
        spec_rows.append({"endpoint": endpoint, "ids": ids, "fields": [path]})
    if not spec_rows:
        raise Unsolvable("no citations")
    return spec_rows


def _reward(snapshot, task, solution) -> dict:
    env = FDAEnv(snapshot, [task], run_id="hard-ref", seed=0)
    env.reset(task["id"])
    for call in solution.calls:
        _obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            return env.trajectory["reward"]
    env.step(
        {
            "tool": "submit_answer",
            "arguments": {"answer": solution.answer, "citations": solution.citations},
        }
    )
    return env.trajectory["reward"]


def _snapshot_answer(cache: _Cache, question: str):
    spec = parse_question(question)
    variant = spec["variant"]
    if variant == "earliest_boxed":
        boxed = []
        for drug in spec["drugs"]:
            if not _boxed_ids(cache, drug):
                continue
            years = _year_apps(cache, drug, with_years=True)
            if not years:
                raise Unsolvable(f"{drug} has no original approval")
            boxed.append((drug, min(year for year, _app in years)))
        return _one(boxed, least=True, what="earliest boxed approval")
    if variant == "faers_year":
        counts = []
        for drug in spec["drugs"]:
            body = cache.query(faers_report_count(drug, serious=True, year=spec["year"]))
            counts.append((drug, int(body["meta"]["results"]["total"])))
        return _one(counts, least=False, what="serious FAERS count")
    if variant == "faers_reaction":
        counts = []
        for drug in spec["drugs"]:
            found = dict(_top_rows(cache, drug)).get(spec["term"])
            if found is None:
                raise Unsolvable(f"{spec['term']} missing for {drug}")
            counts.append((drug, found))
        return _one(counts, least=False, what="reaction count")
    if variant == "boxed_none":
        for drug in spec["drugs"]:
            if _boxed_ids(cache, drug):
                raise Unsolvable(f"{drug} has a boxed warning")
            if not _all_set_ids(cache, drug):
                raise Unsolvable(f"{drug} has no label")
        return False
    if variant == "year_reaction":
        counts = []
        for drug in spec["drugs"]:
            body = cache.query(faers_report_count(drug, serious=True, year=spec["year"]))
            counts.append((drug, int(body["meta"]["results"]["total"])))
        return _top_term_of(cache, _one(counts, least=False, what="serious FAERS count"))
    if variant == "earliest_reaction":
        years = []
        for drug in spec["drugs"]:
            found = _year_apps(cache, drug, with_years=True)
            if not found:
                raise Unsolvable(f"{drug} has no original approval")
            years.append((drug, min(year for year, _app in found)))
        return _top_term_of(cache, _one(years, least=True, what="earliest approval"))
    if variant == "sponsor_count":
        entity = _sponsor_entity(cache, spec)
        body = cache.query(faers_report_count(entity, serious=True, year=spec["year"]))
        return int(body["meta"]["results"]["total"])
    entity = _sponsor_entity(cache, spec)
    _keyword_ids(cache, entity, spec["keyword"], required=True)
    if variant == "sponsor_drug":
        return entity
    rows = _top_rows(cache, entity)
    if not rows:
        raise Unsolvable("no FAERS reaction")
    if len(rows) > 1 and rows[0][1] == rows[1][1]:
        raise Unsolvable("tied FAERS top reaction")
    if re.search(r"\bor\b", rows[0][0], re.IGNORECASE):
        raise Unsolvable("top term contains or")
    return rows[0][0]


def _event_drug(solution) -> str:
    found = None
    for call in solution.calls:
        if call["tool"] == "query_adverse_events":
            name = (call.get("arguments") or {}).get("generic_name")
            if isinstance(name, str) and name.strip():
                found = name
    if not found:
        raise Unsolvable("reaction drug missing")
    return found


def _top_term_of(cache: _Cache, drug: str) -> str:
    rows = _top_rows(cache, drug)
    if not rows:
        raise Unsolvable("no FAERS reaction")
    if len(rows) > 1 and rows[0][1] == rows[1][1]:
        raise Unsolvable("tied FAERS top reaction")
    if re.search(r"\bor\b", rows[0][0], re.IGNORECASE):
        raise Unsolvable("top term contains or")
    return rows[0][0]


def _sponsor_entity(cache: _Cache, spec: dict) -> str:
    records = find_applications(cache.snapshot, spec["application_number"])
    if not records or not str(records[0].get("sponsor_name") or "").strip():
        raise Unsolvable("anchor sponsor missing")
    key = str(records[0]["sponsor_name"]).strip().casefold()
    matches = []
    for drug in spec["drugs"]:
        body = cache.query(approvals_search(drug))
        if any(str(row.get("sponsor_name") or "").strip().casefold() == key for row in body.get("results") or []):
            matches.append(drug)
    if len(matches) != 1:
        raise Unsolvable("sponsor match is not unique in the snapshot")
    return matches[0]


def _one(pairs: list[tuple[str, int]], *, least: bool, what: str) -> str:
    if not pairs:
        raise Unsolvable(f"empty {what}")
    target = min(value for _name, value in pairs) if least else max(value for _name, value in pairs)
    winners = [name for name, value in pairs if value == target]
    if len(winners) != 1:
        raise Unsolvable(f"tied {what}")
    return winners[0]


def _year_apps(cache: _Cache, drug: str, *, with_years: bool = False):
    body = cache.query(approvals_search(drug))
    found = []
    for record in body.get("results") or []:
        number = str(record.get("application_number") or "")
        if not number.upper().startswith(("NDA", "BLA")):
            continue
        products = [item for item in record.get("products") or [] if isinstance(item, dict)]
        if not any(_product_is_single(product, drug) for product in products):
            continue
        year = _orig_year(record)
        if year is not None:
            found.append((year, number))
    if not found:
        return []
    best = min(year for year, _number in found)
    apps = []
    for year, number in found:
        if year == best and number not in apps:
            apps.append(number)
    if with_years:
        return [(year, number) for year, number in found]
    return apps


def _boxed_ids(cache: _Cache, drug: str) -> list[str]:
    body = cache.query(label_search(drug))
    ids = []
    for record in body.get("results") or []:
        if tool_section_text(record, "boxed_warning") and record.get("set_id") not in ids:
            set_id = record.get("set_id")
            if isinstance(set_id, str):
                ids.append(set_id)
    return ids


def _all_set_ids(cache: _Cache, drug: str) -> list[str]:
    body = cache.query(label_search(drug))
    ids = []
    for record in body.get("results") or []:
        set_id = record.get("set_id")
        if isinstance(set_id, str) and set_id not in ids:
            ids.append(set_id)
    return ids


def _sponsor_apps(cache: _Cache, spec: dict, entity: str) -> list[str]:
    records = find_applications(cache.snapshot, spec["application_number"])
    if not records:
        return [spec["application_number"]]
    key = str(records[0].get("sponsor_name") or "").strip().casefold()
    ids = [spec["application_number"]]
    body = cache.query(approvals_search(entity))
    for record in body.get("results") or []:
        number = str(record.get("application_number") or "").upper()
        sponsor = str(record.get("sponsor_name") or "").strip().casefold()
        if sponsor == key and _CANON.match(number) and number not in ids:
            ids.append(number)
    return ids


def _keyword_ids(cache: _Cache, drug: str, keyword: str, *, required: bool = False) -> list[str]:
    body = cache.query(label_search(drug))
    ids = []
    first = None
    for record in body.get("results") or []:
        text = tool_section_text(record, "boxed_warning")
        if not text:
            continue
        if first is None:
            first = text[:SECTION_LIMIT]
        set_id = record.get("set_id")
        visible = keyword in text[:SECTION_LIMIT].casefold()
        if visible and isinstance(set_id, str) and set_id not in ids:
            ids.append(set_id)
    if required and (first is None or keyword not in first.casefold()):
        raise Unsolvable("keyword is not in the first boxed warning")
    return ids


def _top_rows(cache: _Cache, drug: str) -> list[tuple[str, int]]:
    body = cache.query(faers_top_reactions(drug))
    rows = []
    for row in body.get("results") or []:
        if isinstance(row, dict) and isinstance(row.get("term"), str) and isinstance(row.get("count"), int):
            rows.append((row["term"], row["count"]))
    return rows


def _profile(snapshot, cache: _Cache, drug: str) -> Profile:
    profile = Profile(name=drug)
    state = EpisodeState(snapshot)
    try:
        approvals, _length = run_call(state, "search_approvals", {"generic_name": drug}, 1)
    except Unsolvable:
        return profile
    pages = _pages(approvals) or 0
    profile.pages = pages
    profile.one_page = pages == 1
    for row in _rows(approvals):
        display = str(row.get("sponsor_name") or "").strip()
        number = str(row.get("application_number") or "").upper()
        if display and _CANON.match(number):
            profile.sponsors.setdefault(display.casefold(), []).append(number)
    _fill_catalog(cache, profile)
    visible_year, _app, _node = _earliest_visible_year(approvals, drug)
    full_years = _year_apps(cache, drug, with_years=True)
    full_year = min((year for year, _number in full_years), default=None)
    if visible_year is not None and visible_year == full_year:
        profile.year = visible_year
    boxed, text = _read_boxed(snapshot, drug)
    profile.boxed = boxed
    profile.boxed_text = text
    if not boxed:
        profile.label_ok = _labels_fit(snapshot, drug)
    try:
        state = EpisodeState(snapshot)
        top, _length = run_call(
            state, "query_adverse_events", {"generic_name": drug, "mode": "top_reactions"}, 1
        )
        profile.top = [
            (row["term"], row["count"])
            for row in _rows(top)
            if isinstance(row.get("term"), str) and isinstance(row.get("count"), int)
        ]
    except Unsolvable:
        profile.top = []
    for year in FAERS_YEARS:
        try:
            state = EpisodeState(snapshot)
            counted, _length = run_call(
                state,
                "query_adverse_events",
                {"generic_name": drug, "mode": "count", "serious": True, "year": year},
                1,
            )
            profile.counts[year] = _total(counted)
        except Unsolvable:
            continue
    try:
        state = EpisodeState(snapshot)
        counted, _length = run_call(
            state, "query_adverse_events", {"generic_name": drug, "mode": "count"}, 1
        )
        profile.faers_all = _total(counted)
    except Unsolvable:
        profile.faers_all = 0
    return profile


def _earliest_visible_year(output: dict, drug: str) -> tuple[int | None, str | None, dict | None]:
    from fdagym.tasks.hard_solver import _earliest_visible

    return _earliest_visible(output, drug)


def _read_boxed(snapshot, drug: str) -> tuple[bool, str | None]:
    state = EpisodeState(snapshot)
    try:
        output, _length = run_call(
            state,
            "search_drug_label",
            {"generic_name": drug, "has_section": "boxed_warning"},
            1,
        )
    except Unsolvable:
        return False, None
    if _total(output) <= 0:
        return False, None
    row = _first_boxed_row(output)
    if row is None:
        return True, None
    try:
        section, _length = run_call(
            state,
            "get_label_section",
            {"set_id": row["set_id"], "section": "boxed_warning"},
            2,
        )
    except Unsolvable:
        return True, None
    text = section.get("boxed_warning")
    if not isinstance(text, str) or not text.strip():
        return True, None
    return True, text


def _labels_fit(snapshot, drug: str) -> bool:
    state = EpisodeState(snapshot)
    try:
        output, _length = run_call(state, "search_drug_label", {"generic_name": drug}, 1)
    except Unsolvable:
        return False
    rows = _rows(output)
    if _total(output) > len(rows) or not rows:
        return False
    for row in rows:
        if "boxed_warning" in (row.get("sections") or []):
            return False
        if not row.get("set_id"):
            return False
    return True




def _fill_catalog(cache: _Cache, profile: Profile) -> None:
    body = cache.query(label_search(profile.name))
    profile.label_n = int(((body.get("meta") or {}).get("results") or {}).get("total") or 0)
    full = cache.query(approvals_search(profile.name))
    for row in full.get("results") or []:
        if not isinstance(row, dict):
            continue
        display = str(row.get("sponsor_name") or "").strip()
        number = str(row.get("application_number") or "").upper()
        if display and _CANON.match(number):
            profile.all_sponsors.setdefault(display.casefold(), []).append(number)


def _unique_max_value(answer_n: int, values: list[int]) -> bool:
    """True when answer_n is the only maximum. Three distinct totals always rank."""
    return bool(values) and answer_n == max(values) and values.count(max(values)) == 1


def _unfiltered_faers_max(answer: Profile, companions: list[Profile]) -> bool:
    """True when the answer is the unique max of all-years FAERS (all or serious)."""
    return _unfiltered_faers_rank(answer, companions) == "max"


def _unfiltered_faers_rank(answer: Profile, companions: list[Profile]) -> str:
    """Rank of the answer among named all-years FAERS totals. The cheap 3-call ranking."""
    named = [answer, *companions]
    values = [profile.faers_all for profile in named]
    answer_n = answer.faers_all
    if _unique_max_value(answer_n, values):
        return "max"
    if answer_n == min(values) and values.count(min(values)) == 1:
        return "min"
    if values.count(answer_n) == 1:
        return "middle"
    return "tie"


def _faers_join_winner(cache: _Cache, spec: dict, variant: str, gold) -> str | None:
    """The named drug the year or term join picks. year_reaction gold is the term."""
    if variant in {"faers_year", "faers_reaction"} and isinstance(gold, str):
        return gold
    if variant == "year_reaction":
        counts = []
        for drug in spec["drugs"]:
            body = cache.query(faers_report_count(drug, serious=True, year=spec["year"]))
            counts.append((drug, int(body["meta"]["results"]["total"])))
        return _one(counts, least=False, what="serious FAERS count")
    return None
