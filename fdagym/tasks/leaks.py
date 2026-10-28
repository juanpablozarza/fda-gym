"""Surface features that would let a sponsor question be answered without the sponsor lookup.

Each rule names a subset of the candidates. Ties inside that subset count as an even
split, so the score is ``1/|subset|`` when the answer is in it and ``0`` otherwise.
Under the null that the answer is uniform on the named list and independent of the
rule, the expected score is ``1/k``. A rule is a clear leak when that score sits
above chance by a margin and by a one-sided z-test, on a question type with enough
rows to call the pattern real.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from fdagym.tasks.hard_solver import parse_question

SPONSOR_VARIANTS = ("sponsor_drug", "sponsor_faers", "sponsor_count")
MIN_N = 8
Z_LIMIT = 3.0
MARGIN = 0.12


def answer_drug(task: dict, names: list[str]) -> str | None:
    gold = str(task["gold"].get("value")).casefold()
    if gold in names:
        return gold
    entity = ((task.get("entity") or {}).get("generic_name") or "").casefold()
    return entity if entity in names else None


def iter_sponsor(tasks: list[dict]):
    for task in tasks:
        try:
            spec = parse_question(task["question"])
        except (ValueError, KeyError):
            continue
        variant = spec.get("variant")
        if variant not in SPONSOR_VARIANTS:
            continue
        names = [drug.casefold() for drug in spec["drugs"]]
        answer = answer_drug(task, names)
        if answer is None or answer not in names:
            continue
        yield task, variant, names, answer, spec


def _null(k: int, size: int) -> tuple[float, float]:
    mean = 1 / k
    var = 1 / (size * k) - 1 / k**2
    return mean, max(0.0, var)


def _record(bucket: list[tuple[float, float, float]], names: list[str], answer: str, chosen: list[str]) -> None:
    k = len(names)
    size = len(chosen)
    if size <= 0 or size >= k:
        return
    score = 1 / size if answer in chosen else 0.0
    mean, var = _null(k, size)
    bucket.append((score, mean, var))


def _summarize(rows: list[tuple[float, float, float]]) -> dict | None:
    n = len(rows)
    if n == 0:
        return None
    accuracy = sum(score for score, _mean, _var in rows) / n
    chance = sum(mean for _score, mean, _var in rows) / n
    variance = sum(var for _score, _mean, var in rows)
    se = math.sqrt(variance) / n if variance > 0 else 0.0
    z = (accuracy - chance) / se if se > 0 else 0.0
    return {
        "n": n,
        "accuracy": accuracy,
        "chance": chance,
        "margin": accuracy - chance,
        "z": z,
    }


def is_clear(summary: dict | None) -> bool:
    if summary is None or summary["n"] < MIN_N:
        return False
    return summary["z"] > Z_LIMIT and summary["margin"] > MARGIN


def _answer_counts(tasks: list[dict]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {variant: Counter() for variant in SPONSOR_VARIANTS}
    for _task, variant, _names, answer, _spec in iter_sponsor(tasks):
        counts[variant][answer] += 1
    return counts


def rule_rows(
    tasks: list[dict],
    *,
    label_counts: dict[str, int],
    boxed_texts: dict[str, str],
    train_tasks: list[dict] | None,
) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    """Per question type, the per-question scores of each surface rule.

    ``train_tasks`` is the training split. For rows inside that split the answer
    frequency is leave-one-out. For every other split it is the full training count.
    """
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for task, variant, names, answer, spec in iter_sponsor(tasks):
        grouped[variant].append((names, answer, spec.get("keyword")))
    train_counts = _answer_counts(train_tasks or [])
    # The training split is leave-one-out. Eval and the holdout use the full training counts.
    use_loo = train_tasks is None or tasks is train_tasks
    out: dict[str, dict[str, list]] = {}
    for variant, rows in grouped.items():
        rules: dict[str, list] = defaultdict(list)
        totals = Counter(answer for _names, answer, _keyword in rows)
        for names, answer, keyword in rows:
            k = len(names)
            labels = {name: label_counts.get(name, -1) for name in names}
            _record(rules["labels_max"], names, answer, [name for name in names if labels[name] == max(labels.values())])
            _record(rules["labels_min"], names, answer, [name for name in names if labels[name] == min(labels.values())])
            distinct = sorted(set(labels.values()))
            if len(distinct) >= 3:
                mid = distinct[1]
                _record(rules["labels_mid"], names, answer, [name for name in names if labels[name] == mid])
            counts = Counter(labels.values())
            if k == 3 and len(counts) == 2 and 1 in counts.values():
                odd_value = next(value for value, count in counts.items() if count == 1)
                odd = [name for name in names if labels[name] == odd_value]
                tied = [name for name in names if labels[name] != odd_value]
                _record(rules["odd_one"], names, answer, odd)
                _record(rules["tied_group"], names, answer, tied)
            ordered = sorted(names)
            for rank, name in enumerate(ordered, start=1):
                _record(rules[f"alpha_rank{rank}"], names, answer, [name])
            for rank, name in enumerate(names, start=1):
                _record(rules[f"position{rank}"], names, answer, [name])
            if use_loo:
                freq = Counter(totals)
                freq[answer] -= 1
            else:
                freq = train_counts[variant]
            best = max(freq[name] for name in names)
            _record(rules["loo_answer_freq"], names, answer, [name for name in names if freq[name] == best])
            if keyword:
                hits = [name for name in names if keyword.casefold() in boxed_texts.get(name, "")]
                if len(hits) == 1:
                    _record(rules["unique_keyword"], names, answer, hits)
        out[variant] = rules
    return out


def leak_report(
    tasks: list[dict],
    *,
    label_counts: dict[str, int],
    boxed_texts: dict[str, str],
    train_tasks: list[dict] | None,
) -> dict[str, dict[str, dict]]:
    rows = rule_rows(tasks, label_counts=label_counts, boxed_texts=boxed_texts, train_tasks=train_tasks)
    report: dict[str, dict[str, dict]] = {}
    for variant, rules in rows.items():
        report[variant] = {}
        for name, bucket in rules.items():
            summary = _summarize(bucket)
            if summary is None:
                continue
            keyword_hit = name == "unique_keyword" and summary["n"] > 0 and summary["accuracy"] > summary["chance"]
            summary["clear"] = is_clear(summary) or keyword_hit
            report[variant][name] = summary
    return report


def clear_leaks(report: dict[str, dict[str, dict]]) -> list[str]:
    found = []
    for variant, rules in report.items():
        for name, summary in rules.items():
            if summary.get("clear"):
                found.append(
                    f"{variant} {name}: accuracy {summary['accuracy']:.3f} "
                    f"vs chance {summary['chance']:.3f} (n={summary['n']}, z={summary['z']:.2f})"
                )
    return found


def format_leak_report(title: str, report: dict[str, dict[str, dict]]) -> str:
    lines = [title]
    for variant in SPONSOR_VARIANTS:
        rules = report.get(variant) or {}
        if not rules:
            lines.append(f"  {variant}: no rows")
            continue
        ranked = sorted(rules.items(), key=lambda item: -item[1]["accuracy"])
        chance = ranked[0][1]["chance"]
        lines.append(f"  {variant}: chance {chance:.3f}")
        for name, summary in ranked:
            mark = " LEAK" if summary.get("clear") else ""
            lines.append(
                f"    {name:20} {summary['accuracy']:.3f}  n={summary['n']:3}  "
                f"margin {summary['margin']:+.3f}  z={summary['z']:5.2f}{mark}"
            )
    return "\n".join(lines)
