"""Entity split: 80/20 by generic name, seeded, no drug in both sides.

Unanswerable tasks are placed so both splits contain some, then each split is
subsampled to the configured shares. Gold still comes only from the snapshot.
"""

import random
from collections import Counter, defaultdict

from fdagym.tasks.config import GenConfig


def assign_splits(by_drug: dict[str, list[dict]], cfg: GenConfig) -> dict[str, str]:
    drugs = sorted(by_drug)
    if len(drugs) < 2:
        return {d: "train" for d in drugs}
    n_test = round(len(drugs) * cfg.test_fraction)
    n_test = min(max(n_test, 1), len(drugs) - 1)
    rng = random.Random(cfg.seed)
    unans = [d for d in drugs if any(not t["answerable"] for t in by_drug[d])]
    no_boxed = [d for d in drugs if _has_easy_boxed_no(by_drug[d])]
    test: list[str] = []
    reserved_train: set[str] = set()
    if len(unans) >= 2:
        test.extend(_unans_for_test(unans, by_drug, cfg))
        reserved_train = set(unans) - set(test)
    rest = [d for d in drugs if d not in set(test) and d not in reserved_train]
    rng.shuffle(rest)
    _place_easy_no(test, rest, no_boxed, n_test, unans)
    while len(test) < n_test and rest:
        best_at = min(range(len(rest)), key=lambda i: _imbalance(by_drug, test + [rest[i]], drugs))
        test.append(rest.pop(best_at))
    chosen = set(test)
    return {d: "test" if d in chosen else "train" for d in drugs}


def balance_boxed_yes_no(tasks: list[dict]) -> list[dict]:
    """Within each split, drop surplus Yes boxed-warning tasks until Yes count equals No count.

    Only yes/no tasks (no topic keywords). A split with no No tasks keeps its Yes tasks,
    because inventing a No would not be in the snapshot.
    """
    drop: set[int] = set()
    by_split: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, task in enumerate(tasks):
        if task["type"] == "boxed_warning" and "keywords" not in task["gold"]:
            by_split[task["split"]].append((index, task))
    for group in by_split.values():
        yes = [(index, task) for index, task in group if task["gold"]["value"] is True]
        no = [task for _index, task in group if task["gold"]["value"] is False]
        if not no or len(yes) <= len(no):
            continue
        yes.sort(key=lambda item: (item[1]["entity"]["generic_name"], item[1]["question"]))
        drop.update(index for index, _task in yes[len(no) :])
    return [task for index, task in enumerate(tasks) if index not in drop]


def select_for_shares(tasks: list[dict], cfg: GenConfig) -> list[dict]:
    """Cap label-open and unanswerable shares. Never drop answerable tasks to mint a 5%."""
    by_split: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_split[task["split"]].append(task)
    chosen: list[dict] = []
    for split in ("train", "test"):
        chosen.extend(_select_split(by_split.get(split, []), cfg, split))
    return chosen


def _select_split(tasks: list[dict], cfg: GenConfig, split: str) -> list[dict]:
    if not tasks:
        return []
    rng = random.Random(cfg.seed + (1 if split == "test" else 0))
    unans = [t for t in tasks if not t["answerable"]]
    opens = [t for t in tasks if t["answerable"] and t["type"] == "label_open"]
    core = [t for t in tasks if t["answerable"] and t["type"] != "label_open"]
    open_n = min(len(opens), _open_quota(len(core), len(opens), cfg))
    kept_opens = _sample(opens, open_n, rng)
    base = len(core) + len(kept_opens)
    cap = _unanswerable_cap(base, cfg)
    kept_unans = unans if len(unans) <= cap else _sample(unans, cap, rng)
    return kept_unans + core + kept_opens


def _open_quota(n_core: int, n_open: int, cfg: GenConfig) -> int:
    if n_core + n_open == 0:
        return 0
    quota = min(n_open, round(cfg.label_open_share * (n_core + n_open)))
    return min(n_open, round(cfg.label_open_share * (n_core + quota)))


def _unanswerable_cap(n_answerable: int, cfg: GenConfig) -> int:
    """Largest unanswerable count that stays at or under the share, given the answerable rows."""
    if n_answerable == 0 or cfg.unanswerable_share <= 0:
        return 0
    if cfg.unanswerable_share >= 1:
        return 10**9
    return int(cfg.unanswerable_share / (1 - cfg.unanswerable_share) * n_answerable + 1e-9)


def _n_unans(tasks: list[dict]) -> int:
    return sum(1 for task in tasks if not task["answerable"])


def _has_easy_boxed_no(tasks: list[dict]) -> bool:
    return any(
        task["type"] == "boxed_warning"
        and "keywords" not in task.get("gold", {})
        and task.get("gold", {}).get("value") is False
        for task in tasks
    )


def _unans_for_test(unans: list[str], by_drug: dict[str, list[dict]], cfg: GenConfig) -> list[str]:
    """Put ~test_fraction of unanswerable *tasks* in test, leaving at least one unans drug in train."""
    ranked = sorted(unans, key=lambda d: (-_n_unans(by_drug[d]), d))
    target = round(sum(_n_unans(by_drug[d]) for d in ranked) * cfg.test_fraction)
    chosen: list[str] = []
    acc = 0
    for i, drug in enumerate(ranked):
        last = i == len(ranked) - 1
        if last and chosen:
            break
        if chosen and acc >= target:
            break
        chosen.append(drug)
        acc += _n_unans(by_drug[drug])
        if last:
            break
    return chosen


def _place_easy_no(test: list[str], rest: list[str], no_boxed: list[str], n_test: int, unans: list[str]) -> None:
    """If a unanimous-No boxed drug exists, put one in test so that split can host the easy No."""
    if not no_boxed or any(drug in no_boxed for drug in test):
        return
    pick = next((drug for drug in rest if drug in no_boxed), None)
    if pick is None:
        return
    if len(test) < n_test:
        rest.remove(pick)
        test.append(pick)
        return
    unans_in_test = [drug for drug in test if drug in unans]
    for i, drug in enumerate(test):
        if drug in unans and len(unans_in_test) <= 1:
            continue
        rest.append(drug)
        rest.remove(pick)
        test[i] = pick
        return


def _sample(tasks: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n >= len(tasks):
        return list(tasks)
    if n <= 0:
        return []
    return rng.sample(tasks, n)


def _imbalance(by_drug: dict[str, list[dict]], test: list[str], drugs: list[str]) -> float:
    test_set = set(test)
    test_counts: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    for drug in drugs:
        bucket = test_counts if drug in test_set else train_counts
        for task in by_drug[drug]:
            bucket[task["type"]] += 1
            bucket[task["difficulty"]] += 1
    score = 0.0
    for key in set(test_counts) | set(train_counts):
        total = test_counts[key] + train_counts[key]
        if total:
            score += abs(test_counts[key] / total - 0.2)
    return score
