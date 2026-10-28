"""Re-solve hard questions from the text alone and score them on the real reward.

The solver never reads gold or oracle_calls. ``budget`` is an argument so a 6-call
check and an 8-call check share one path. Each budget is passed to ``solve`` and
to ``FDAEnv(..., tool_budget=budget)``. The module constant ``TOOL_BUDGET`` is only
the default for callers that do not pass one.
"""

from __future__ import annotations

from collections import Counter

from fdagym.env import TOOL_BUDGET, FDAEnv
from fdagym.tasks.hard_solver import OBS_LIMIT, Unsolvable, parse_question, same_answer, solve


def check_reachability(snapshot, tasks: list[dict], budgets: list[int], *, score=None) -> dict:
    """Solve every task at each budget. A task is solved only when the answer, the clip, and the reward agree."""
    if not budgets:
        raise ValueError("at least one budget is required")
    ordered = sorted({int(budget) for budget in budgets})
    if any(budget < 1 for budget in ordered):
        raise ValueError(f"budgets must be >= 1, got {ordered}")
    per_budget = []
    call_counts: list[int] = []
    for budget in ordered:
        per_budget.append(_one_budget(snapshot, tasks, budget, score or _score))
    widest = per_budget[-1]
    call_counts = list(widest["solved_calls"])
    distribution = {str(calls): count for calls, count in sorted(Counter(call_counts).items())}
    solved_ids = {row["id"] for row in widest["solved_ids"]}
    return {
        "n": len(tasks),
        "budgets": ordered,
        "env_tool_budget": ordered[-1],
        "module_tool_budget": TOOL_BUDGET,
        "by_budget": [{key: value for key, value in row.items() if key not in {"solved_calls", "solved_ids"}} for row in per_budget],
        "min_calls": distribution,
        "solved_at_max_budget": len(solved_ids),
        "fraction_at_max_budget": (len(solved_ids) / len(tasks)) if tasks else 0.0,
    }


def _one_budget(snapshot, tasks: list[dict], budget: int, score) -> dict:
    solved = 0
    failures = []
    calls: list[int] = []
    widths: list[int] = []
    by_variant: Counter[str] = Counter()
    solved_variant: Counter[str] = Counter()
    solved_ids: list[dict] = []
    for task in tasks:
        try:
            variant = parse_question(task["question"])["variant"]
        except Unsolvable as exc:
            failures.append({"id": task["id"], "reason": f"parse: {exc}"})
            continue
        by_variant[variant] += 1
        try:
            solution = solve(snapshot, task["question"], budget=budget)
        except Unsolvable as exc:
            failures.append({"id": task["id"], "reason": str(exc), "variant": variant})
            continue
        if not same_answer(solution.answer, task["gold"]["value"]):
            failures.append(
                {
                    "id": task["id"],
                    "reason": "answer",
                    "variant": variant,
                    "got": solution.answer,
                    "gold": task["gold"]["value"],
                }
            )
            continue
        if solution.n_calls > budget or any(length > OBS_LIMIT for length in solution.observation_chars):
            failures.append({"id": task["id"], "reason": "budget", "variant": variant, "calls": solution.n_calls})
            continue
        reward = score(snapshot, task, solution, budget)
        if reward["total"] < 1.0 - 1e-9:
            failures.append({"id": task["id"], "reason": "score", "variant": variant, "notes": reward["notes"][:8]})
            continue
        solved += 1
        solved_variant[variant] += 1
        calls.append(solution.n_calls)
        widths.extend(solution.observation_chars)
        solved_ids.append({"id": task["id"]})
    return {
        "budget": budget,
        "n": len(tasks),
        "solved": solved,
        "fraction": (solved / len(tasks)) if tasks else 0.0,
        "by_variant": {
            name: {"n": by_variant[name], "solved": solved_variant[name]} for name in sorted(by_variant)
        },
        "min_calls": {str(n): count for n, count in sorted(Counter(calls).items())},
        "max_calls": max(calls) if calls else 0,
        "max_observation_chars": max(widths) if widths else 0,
        "failures": failures[:30],
        "n_failures": len(failures),
        "solved_calls": calls,
        "solved_ids": solved_ids,
    }


def _score(snapshot, task, solution, budget: int) -> dict:
    env = FDAEnv(snapshot, [task], run_id="hard-reach", seed=0, tool_budget=budget)
    env.reset(task["id"])
    for call in solution.calls:
        _obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            return env.trajectory["reward"]
    env.step({"tool": "submit_answer", "arguments": {"answer": solution.answer, "citations": solution.citations}})
    return env.trajectory["reward"]
