"""Eval-only fact swaps. Training rollouts construct ``FDAEnv`` and never this wrapper.

A swapped episode rewrites one tool output so the answer read off that output
is not the snapshot gold. The task's stored gold stays the original. The
follow rate is how often the submission matches the rewritten value.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from fdagym.env import FDAEnv
from fdagym.memorization import memorization_report
from fdagym.tasks.hard_solver import parse_question, solve
from fdagym.train.rollout import open_provided_env, run_episodes, training_rollout_active

COUNT_OFFSET = 17
REACTION_SWAPPED = "HEADACHE"
REACTION_FALLBACK = "NAUSEA"
SWAP_QUOTAS = (
    ("sponsor_count", 4),
    ("year_reaction", 2),
    ("earliest_reaction", 2),
    ("sponsor_faers", 1),
)
SWAP_N = 9
_COUNT_VARIANTS = frozenset({"sponsor_count"})
_REACTION_VARIANTS = frozenset({"year_reaction", "earliest_reaction", "sponsor_faers"})


class SwappedFactsError(RuntimeError):
    """The eval-only wrapper was asked to run inside a training rollout."""


@dataclass(frozen=True)
class FactSwap:
    task_id: str
    variant: str
    kind: str
    original: object
    swapped: object

    def apply(self, tool: str, arguments: dict, output: dict) -> dict:
        if tool != "query_adverse_events":
            return output
        mode = arguments.get("mode")
        if self.kind == "count" and mode == "count" and _total(output) == self.original:
            cloned = copy.deepcopy(output)
            cloned["meta"]["results"]["total"] = self.swapped
            return cloned
        if self.kind == "reaction" and mode == "top_reactions":
            rows = output.get("results") or []
            if not rows or not isinstance(rows[0], dict):
                return output
            term = rows[0].get("term")
            if not isinstance(term, str) or term.casefold() != str(self.original).casefold():
                return output
            cloned = copy.deepcopy(output)
            cloned["results"][0]["term"] = self.swapped
            return cloned
        return output


class SwappedFactsEnv(FDAEnv):
    """``FDAEnv`` that rewrites planned tool outputs before the caller sees them.

    Construction and ``step`` both raise when a training rollout is open.
    """

    def __init__(self, snapshot, tasks, plans: dict[str, FactSwap], **kwargs):
        _refuse_training()
        self.plans = plans
        self._plan: FactSwap | None = None
        super().__init__(snapshot, tasks, **kwargs)

    def reset(self, task_id: str) -> dict:
        _refuse_training()
        obs = super().reset(task_id)
        self._plan = self.plans.get(task_id)
        return obs

    def step(self, action):
        _refuse_training()
        import fdagym.env as env_module

        original = env_module.execute
        plan = self._plan

        def wrapped(state, name, arguments, step_index=None):
            output, err = original(state, name, arguments, step_index=step_index)
            if err is None and isinstance(output, dict) and plan is not None:
                output = plan.apply(name, arguments, output)
            return output, err

        env_module.execute = wrapped
        try:
            return super().step(action)
        finally:
            env_module.execute = original


def select_fact_swaps(tasks: list[dict], *, n: int = SWAP_N) -> list[FactSwap]:
    """Known-drug holdout questions whose last fact can be rewritten."""
    grouped: dict[str, list[dict]] = {variant: [] for variant, _quota in SWAP_QUOTAS}
    for task in tasks:
        variant = parse_question(task["question"])["variant"]
        if variant in grouped:
            grouped[variant].append(task)
    for rows in grouped.values():
        rows.sort(key=lambda task: task["id"])
    picked: list[dict] = []
    used: set[str] = set()
    for variant, quota in SWAP_QUOTAS:
        for task in grouped[variant][:quota]:
            picked.append(task)
            used.add(task["id"])
    if len(picked) < n:
        rest = [
            task
            for variant, _quota in SWAP_QUOTAS
            for task in grouped[variant]
            if task["id"] not in used
        ]
        rest.sort(key=lambda task: task["id"])
        picked.extend(rest[: n - len(picked)])
    return [fact_swap(task) for task in picked[:n]]


def fact_swap(task: dict) -> FactSwap:
    variant = parse_question(task["question"])["variant"]
    original = task["gold"]["value"]
    if variant in _COUNT_VARIANTS:
        if isinstance(original, bool) or not isinstance(original, int):
            raise ValueError(f"{task['id']} count gold is {original!r}")
        return FactSwap(task["id"], variant, "count", original, original + COUNT_OFFSET)
    if variant not in _REACTION_VARIANTS:
        raise ValueError(f"{task['id']} variant {variant} is not swappable")
    if not isinstance(original, str) or not original.strip():
        raise ValueError(f"{task['id']} reaction gold is {original!r}")
    swapped = REACTION_FALLBACK if original.casefold() == REACTION_SWAPPED.casefold() else REACTION_SWAPPED
    return FactSwap(task["id"], variant, "reaction", original, swapped)


def scripted_oracle_report(snapshot, known: list[dict], unseen: list[dict], *, tool_budget: int = 8) -> dict:
    """Play the question solver on known and unseen, and the swapped outputs on a held-out slice.

    The solver reads tool output, so the swapped slice should follow the rewrite.
    """
    known_rows = [_play_solver(snapshot, task, tool_budget) for task in known]
    unseen_rows = [_play_solver(snapshot, task, tool_budget) for task in unseen]
    plans = select_fact_swaps(known)
    by_id = {task["id"]: task for task in known}
    swapped_rows = [
        _play_swapped(snapshot, by_id[plan.task_id], plan, tool_budget) for plan in plans
    ]
    return memorization_report(known=known_rows, unseen=unseen_rows, swapped=swapped_rows)


def rollout_swapped(snapshot, tasks, plans: list[FactSwap], tokenizer, generator, codec, limits, *, run_id, seed, model):
    """Model rollouts on the wrapper. Raises inside a training rollout."""
    _refuse_training()
    by_id = {task["id"]: task for task in tasks}
    plan_map = {plan.task_id: plan for plan in plans}
    selected = [by_id[plan.task_id] for plan in plans]
    episodes = []
    for index, task in enumerate(selected):
        env = SwappedFactsEnv(
            snapshot,
            selected,
            plan_map,
            run_id=run_id,
            seed=seed + index,
            model=dict(model),
            tool_budget=limits.tool_budget,
        )
        episodes.append(open_provided_env(env, task["id"], tokenizer, codec, limits))
    rollouts = run_episodes(episodes, tokenizer, generator, codec, limits)
    rows = []
    for rollout, plan in zip(rollouts, plans, strict=True):
        rows.append((by_id[plan.task_id], rollout.trajectory, plan.swapped))
    return rows


def answer_from_output(steps: list[dict], plan: FactSwap):
    """The answer a reader of the (possibly rewritten) tool output would submit."""
    mode = "count" if plan.kind == "count" else "top_reactions"
    for step in reversed(steps):
        if step.get("tool") != "query_adverse_events":
            continue
        if (step.get("arguments") or {}).get("mode") != mode:
            continue
        output = step.get("output") or {}
        if plan.kind == "count":
            return _total(output)
        rows = output.get("results") or []
        if rows and isinstance(rows[0], dict) and isinstance(rows[0].get("term"), str):
            return rows[0]["term"]
    raise RuntimeError(f"{plan.task_id} did not show a {plan.kind} output")


def _play_solver(snapshot, task: dict, tool_budget: int) -> tuple[dict, dict]:
    solution = solve(snapshot, task["question"], budget=tool_budget)
    env = FDAEnv(snapshot, [task], run_id="memo-oracle", seed=0, tool_budget=tool_budget)
    env.reset(task["id"])
    for call in solution.calls:
        _obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            raise RuntimeError(f"{task['id']} ended before submit")
    env.step({"tool": "submit_answer", "arguments": {"answer": solution.answer, "citations": solution.citations}})
    return task, env.trajectory


def _play_swapped(snapshot, task: dict, plan: FactSwap, tool_budget: int) -> tuple[dict, dict, object]:
    env = SwappedFactsEnv(
        snapshot,
        [task],
        {plan.task_id: plan},
        run_id="memo-swap",
        seed=0,
        tool_budget=tool_budget,
    )
    env.reset(task["id"])
    for call in task["oracle_calls"]:
        _obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            raise RuntimeError(f"{task['id']} ended before the swapped submit")
    answer = answer_from_output(env.steps, plan)
    env.step({"tool": "submit_answer", "arguments": {"answer": answer, "citations": []}})
    return task, env.trajectory, plan.swapped


def _total(output: dict):
    return ((output.get("meta") or {}).get("results") or {}).get("total")


def _refuse_training() -> None:
    if training_rollout_active():
        raise SwappedFactsError("SwappedFactsEnv is eval-only and cannot run inside a training rollout")
