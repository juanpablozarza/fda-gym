"""Hard-training tasks. Same templates as the hard eval, plus three longer hops.

Reaction-answer compares and boxed_none use the train wording in ``hard_solver``
so the gold stays a single value. Eval boxed_none keeps ``question_none``.

The pool is whatever is left after ``disjoint.holdout_entities`` removes every drug,
application number, and sponsor that appears in ``hard.jsonl`` or ``test.jsonl``.
Gold is recomputed from the snapshot. A task is kept only when the question-only
solver scores 1.0 inside ``budget`` calls (default 6, today's env cap).
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from fdagym.tasks.disjoint import holdout_entities, task_entities
from fdagym.tasks.hard import (
    FAERS_YEARS,
    _accept,
    _Cache,
    _profile,
    _shared_term,
    _unfiltered_faers_rank,
    fill_sponsor_tasks,
)
from fdagym.tasks.hard_solver import (
    BUDGET,
    parse_question,
    question_earliest,
    question_earliest_reaction,
    question_faers_year,
    question_none_yes_no,
    question_reaction,
    question_year_reaction,
)
from fdagym.tasks.plan import load_universe

HARD_TRAIN_SEED = 0
ID_START = 8101
MAX_TASKS = 1500
_OR = re.compile(r"\bor\b", re.IGNORECASE)

# Targets, not a promise. The snapshot's leftover drugs decide how many survive.
# No single variant is most of the file. sponsor_faers stays large enough that the
# training rows, after the known-drug holdout, still clear about 15.
_QUOTAS = {
    "earliest_boxed": 44,
    "sponsor_faers": 40,
    "sponsor_drug": 44,
    "sponsor_count": 72,
    "year_reaction": 44,
    "earliest_reaction": 44,
    "faers_year": 44,
    "faers_reaction": 44,
    "boxed_none": 4,
}
NAMED_CAP = 16
ANSWER_CAP = 8


def generate_hard_train(
    snapshot,
    *,
    seed: int = HARD_TRAIN_SEED,
    budget: int = BUDGET,
    quotas: dict[str, int] | None = None,
    max_tasks: int = MAX_TASKS,
    holdout_tasks: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Build ``data/tasks/hard_train.jsonl``. ``holdout_tasks`` defaults to hard + test."""
    if holdout_tasks is None:
        holdout_tasks = _load_holdout()
    quotas = dict(_QUOTAS if quotas is None else quotas)
    holdout = holdout_entities(holdout_tasks, snapshot)
    cache = _Cache(snapshot)
    rng = random.Random(seed)
    profiles, pool_notes = _pools(snapshot, cache, holdout)
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    groups: dict[str, list[dict]] = defaultdict(list)
    _fill_compare(snapshot, cache, profiles, rng, quotas, budget, holdout, seen, dropped, groups)
    _fill_sponsor(snapshot, cache, profiles, rng, quotas, budget, holdout, seen, dropped, groups)
    _trim(groups, max_tasks)
    tasks = _assign_ids([task for rows in groups.values() for task in rows])
    by_variant = Counter(parse_question(task["question"])["variant"] for task in tasks)
    by_calls = Counter(len(task["oracle_calls"]) for task in tasks)
    report = {
        "seed": seed,
        "budget": budget,
        "n": len(tasks),
        "by_variant": dict(sorted(by_variant.items())),
        "by_type": dict(Counter(task["type"] for task in tasks)),
        "min_calls": {str(n): by_calls[n] for n in sorted(by_calls)},
        "pool": pool_notes,
        "holdout": {
            "drugs": len(holdout.drugs),
            "applications": len(holdout.applications),
            "sponsors": len(holdout.sponsors),
            "tasks": len(holdout_tasks),
        },
        "dropped": dict(dropped),
        "quotas": quotas,
    }
    return tasks, report


def _load_holdout() -> list[dict]:
    from fdagym.agents.baseline import TASKS_DIR, load_split

    rows = []
    for name in ("hard", "test"):
        if (TASKS_DIR / f"{name}.jsonl").exists():
            rows.extend(load_split(name))
    return rows


def _pools(snapshot, cache, holdout) -> tuple[list, dict]:
    profiles = []
    named = 0
    for drug in load_universe():
        if "," in drug or " and " in drug.casefold():
            continue
        if drug.casefold() in holdout.drugs:
            named += 1
            continue
        profiles.append(_profile(snapshot, cache, drug))
    one_page = sum(1 for profile in profiles if profile.one_page)
    boxed = sum(1 for profile in profiles if profile.boxed and profile.year is not None)
    shared = _shared_sponsors(profiles, holdout)
    notes = {
        "universe": len(load_universe()),
        "held_out_by_name": named,
        "remaining": len(profiles),
        "with_visible_year": sum(1 for profile in profiles if profile.year is not None),
        "boxed_with_year": boxed,
        "one_page": one_page,
        "label_ok_unboxed": sum(1 for profile in profiles if profile.label_ok and not profile.boxed),
        "page1_sponsors_shared": len(shared),
        "page1_sponsor_drugs": len({name for names in shared.values() for name in names}),
    }
    return profiles, notes


def _shared_sponsors(profiles, holdout) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for profile in profiles:
        for key in profile.sponsors:
            if key in holdout.sponsors:
                continue
            index[key].append(profile.name)
    return {key: names for key, names in index.items() if len(set(names)) >= 2}


def _fill_compare(snapshot, cache, profiles, rng, quotas, budget, holdout, seen, dropped, groups) -> None:
    counted = [profile for profile in profiles if all(year in profile.counts for year in FAERS_YEARS)]
    dated = [profile for profile in profiles if profile.year is not None]
    boxed = [profile for profile in profiles if profile.boxed and profile.year is not None]
    reacted = [profile for profile in profiles if profile.top]
    quiet = [profile for profile in profiles if profile.label_ok and not profile.boxed]
    per_year = max(1, quotas.get("faers_year", 0) // len(FAERS_YEARS))
    for year in FAERS_YEARS:
        groups["faers_year"].extend(
            _sample_compare(
                snapshot,
                cache,
                counted,
                rng,
                "faers_year",
                per_year,
                budget,
                holdout,
                seen,
                dropped,
                lambda trio, year=year: _faers_year_token(trio, year),
                lambda order, token: question_faers_year([profile.name for profile in order], token),
                rank_of=lambda trio, year: _faers_year_rank(trio, year),
            )
        )
    groups["year_reaction"].extend(
        _sample_years(
            snapshot,
            cache,
            counted,
            rng,
            "year_reaction",
            quotas.get("year_reaction", 0),
            budget,
            holdout,
            seen,
            dropped,
            require_top=True,
            question_of=lambda order, year: question_year_reaction([profile.name for profile in order], year),
            rank_of=lambda trio, year: _faers_year_rank(trio, year),
        )
    )
    groups["earliest_reaction"].extend(
        _sample_compare(
            snapshot,
            cache,
            dated,
            rng,
            "earliest_reaction",
            quotas.get("earliest_reaction", 0),
            budget,
            holdout,
            seen,
            dropped,
            _earliest_reaction_ok,
            lambda order, _token: question_earliest_reaction([profile.name for profile in order]),
        )
    )
    groups["earliest_boxed"].extend(
        _sample_compare(
            snapshot,
            cache,
            boxed,
            rng,
            "earliest_boxed",
            quotas.get("earliest_boxed", 0),
            budget,
            holdout,
            seen,
            dropped,
            _earliest_boxed_ok,
            lambda order, _token: question_earliest([profile.name for profile in order]),
        )
    )
    groups["faers_reaction"].extend(
        _sample_compare(
            snapshot,
            cache,
            reacted,
            rng,
            "faers_reaction",
            quotas.get("faers_reaction", 0),
            budget,
            holdout,
            seen,
            dropped,
            _reaction_token,
            lambda order, token: question_reaction([profile.name for profile in order], token),
            rank_of=lambda trio, term: _reaction_rank(trio, term),
        )
    )
    groups["boxed_none"].extend(
        _sample_compare(
            snapshot,
            cache,
            quiet,
            rng,
            "boxed_none",
            quotas.get("boxed_none", 0),
            budget,
            holdout,
            seen,
            dropped,
            lambda _trio: True,
            lambda order, _token: question_none_yes_no([profile.name for profile in order]),
            cap=4,
        )
    )


def _sample_years(
    snapshot, cache, pool, rng, variant, quota, budget, holdout, seen, dropped, require_top, question_of, rank_of=None
):
    kept = []
    if quota <= 0:
        return kept
    share = max(1, quota // len(FAERS_YEARS))
    for year in FAERS_YEARS:
        kept.extend(
            _sample_compare(
                snapshot,
                cache,
                pool,
                rng,
                variant,
                share,
                budget,
                holdout,
                seen,
                dropped,
                lambda trio, year=year: _year_reaction_token(trio, year, require_top),
                lambda order, token, question_of=question_of: question_of(order, token),
                rank_of=None if rank_of is None else (lambda trio, token, year=year: rank_of(trio, year)),
            )
        )
    return kept


def _sample_compare(
    snapshot,
    cache,
    pool,
    rng,
    variant,
    quota,
    budget,
    holdout,
    seen,
    dropped,
    qualify,
    question_of,
    cap=60,
    rank_of=None,
):
    kept = []
    seen_keys: set[tuple] = set()
    used: Counter[str] = Counter()
    rank_used: Counter[str] = Counter()
    rank_cap = (quota + 1) // 2
    if len(pool) < 3 or quota <= 0:
        if quota > 0:
            dropped[f"short:{variant}"] += quota
        return kept
    attempts = 0
    limit = max(4000, quota * 400)
    while len(kept) < quota and attempts < limit:
        attempts += 1
        trio = rng.sample(pool, 3)
        names = [profile.name for profile in trio]
        if any(used[name] >= cap for name in names):
            continue
        token = qualify(trio)
        if not token:
            continue
        rank = rank_of(trio, token) if rank_of is not None else None
        if rank is not None and rank_used[rank] >= rank_cap:
            dropped[f"faers_rank:{variant}:{rank}"] += 1
            continue
        key = (frozenset(name.casefold() for name in names), token if token is not True else variant)
        if key in seen_keys:
            continue
        order = list(trio)
        rng.shuffle(order)
        question = question_of(order, token)
        if question in seen:
            continue
        task = _keep(snapshot, cache, question, variant, budget, holdout, dropped)
        if task is None:
            continue
        seen_keys.add(key)
        seen.add(question)
        if rank is not None:
            rank_used[rank] += 1
        for name in names:
            used[name] += 1
        kept.append(task)
    if len(kept) < quota:
        dropped[f"short:{variant}"] += quota - len(kept)
    return kept


def _fill_sponsor(snapshot, cache, profiles, rng, quotas, budget, holdout, seen, dropped, groups) -> None:
    wanted = {
        "sponsor_faers": quotas.get("sponsor_faers", 0),
        "sponsor_drug": quotas.get("sponsor_drug", 0),
        "sponsor_count": quotas.get("sponsor_count", 0),
    }

    def accept(question, variant):
        return _keep(snapshot, cache, question, variant, budget, holdout, dropped)

    built = fill_sponsor_tasks(
        profiles,
        rng,
        quotas=wanted,
        dropped=dropped,
        accept=accept,
        named_cap=NAMED_CAP,
        answer_cap=ANSWER_CAP,
        anchor_cap=8,
        blocked_sponsors=set(holdout.sponsors),
        blocked_apps=set(holdout.applications),
        seen=seen,
        keyword_mode="pair",
    )
    for variant, tasks in built.items():
        groups[variant].extend(tasks)


def _trim(groups: dict[str, list[dict]], max_tasks: int) -> None:
    order = ["faers_year", "year_reaction", "faers_reaction", "earliest_reaction", "sponsor_count"]
    total = sum(len(rows) for rows in groups.values())
    for variant in order:
        while total > max_tasks and len(groups.get(variant, [])) > 40:
            groups[variant].pop()
            total -= 1


def _keep(snapshot, cache, question, variant, budget, holdout, dropped):
    task = _accept(snapshot, cache, question, variant, dropped, budget=budget, split="hard_train")
    if task is None:
        return None
    found = task_entities(task, snapshot)
    if set(found.drugs) & set(holdout.drugs):
        dropped[f"heldout-drug:{variant}"] += 1
        return None
    if set(found.applications) & set(holdout.applications):
        dropped[f"heldout-app:{variant}"] += 1
        return None
    if set(found.sponsors) & set(holdout.sponsors):
        dropped[f"heldout-sponsor:{variant}"] += 1
        return None
    return task


def _assign_ids(tasks: list[dict]) -> list[dict]:
    ordered = sorted(tasks, key=lambda task: (task["type"], task["question"]))
    tagged = []
    for offset, task in enumerate(ordered):
        number = ID_START + offset
        if number > 9999:
            break
        tagged.append({**task, "id": f"FG-{number:04d}"})
    return tagged


def _companions(trio, winner):
    return [profile for profile in trio if winner is not None and profile.name != winner.name]


def _faers_year_token(trio, year):
    winner = _winner(trio, lambda profile: profile.counts.get(year), least=False)
    if winner is None:
        return None
    return year


def _faers_year_rank(trio, year):
    winner = _winner(trio, lambda profile: profile.counts.get(year), least=False)
    if winner is None:
        return "tie"
    return _unfiltered_faers_rank(winner, _companions(trio, winner))


def _year_reaction_token(trio, year, require_top):
    winner = _winner(trio, lambda profile: profile.counts.get(year), least=False)
    if winner is None:
        return None
    if require_top and not _top_ok(winner):
        return None
    return year


def _reaction_token(trio):
    return _shared_term(trio, Counter())


def _reaction_rank(trio, term):
    winner = max(trio, key=lambda profile: dict(profile.top).get(term, -1))
    return _unfiltered_faers_rank(winner, _companions(trio, winner))


def _earliest_reaction_ok(trio):
    winner = _winner(trio, lambda profile: profile.year, least=True)
    if winner is None or not _top_ok(winner):
        return None
    return True


def _earliest_boxed_ok(trio):
    return True if _winner(trio, lambda profile: profile.year, least=True) else None


def _unique_value(trio, value_of, *, least: bool = False):
    pairs = []
    for profile in trio:
        value = value_of(profile)
        if value is None:
            return None
        pairs.append((profile, value))
    target = min(value for _profile, value in pairs) if least else max(value for _profile, value in pairs)
    winners = [profile for profile, value in pairs if value == target]
    if len(winners) != 1:
        return None
    return winners[0]


def _winner(trio, value_of, *, least: bool):
    found = _unique_value(trio, value_of, least=least)
    return found


def _top_ok(profile) -> bool:
    if profile is None or not profile.top:
        return False
    if len(profile.top) > 1 and profile.top[0][1] == profile.top[1][1]:
        return False
    return _OR.search(profile.top[0][0]) is None
