"""CPU dry run: a scripted oracle plays two tasks through the rollout runner.

The tokenizer is one character per token, so a 256-token budget is 256 characters.
Qwen BPE is denser than that. A dry run that fits is a lower bound, not a GPU measurement.
"""

from __future__ import annotations

import re

from fdagym.env import FDAEnv
from fdagym.snapshot import Snapshot
from fdagym.train.boxed_oracle import ScriptError, evidence_oracle, tool_actions
from fdagym.train.context import play_oracle
from fdagym.train.gold import gold_submit
from fdagym.train.rollout import (
    TOOL_MARK,
    QwenToolCodec,
    RolloutLimits,
    TurnGeneration,
    rollout_batch,
)
from fdagym.train.tool_score import episode_scores


class CharTokenizer:
    """Reversible stand-in. Token 0 is EOS, token 1 is pad, everything else is ``ord(ch) + 2``."""

    eos_token_id = 0
    pad_token_id = 1

    def encode(self, text: str) -> list[int]:
        return [ord(char) + 2 for char in text]

    def decode(self, token_ids) -> str:
        chars = []
        for token_id in token_ids:
            if token_id in (self.eos_token_id, self.pad_token_id):
                continue
            chars.append(chr(token_id - 2))
        return "".join(chars)


def scripted_actions(snapshot: Snapshot, tasks: list[dict], task: dict) -> list[dict]:
    """Tool calls plus ``submit_answer`` for one scripted episode.

    Boxed-warning and composite label steps come from the search result. Every
    other type replays ``oracle_calls``. ``ScriptError`` means the episode cannot
    be cited; the caller skips that row.
    """
    if evidence_oracle(task):
        actions, env = tool_actions(snapshot, tasks, task)
        try:
            submit = gold_submit(task, env.steps)
        except RuntimeError as exc:
            raise ScriptError(str(exc)) from exc
        return [*actions, {"tool": "submit_answer", "arguments": submit}]
    task = by_application_number(task)
    env = FDAEnv(snapshot, tasks, run_id="oracle-script")
    play_oracle(env, task)
    actions = [{"tool": call["tool"], "arguments": dict(call["arguments"])} for call in task["oracle_calls"]]
    actions.append({"tool": "submit_answer", "arguments": gold_submit(task, env.steps)})
    return actions


_QUESTION_APP_NUMBER = re.compile(r"\b(NDA|ANDA|BLA)\s*-?\s*(\d{4,6})\b", re.IGNORECASE)


def by_application_number(task: dict) -> dict:
    """Approval questions that name only an application number search by that number.

    The stored ``oracle_calls`` search ``entity.generic_name``, which such a question
    never shows the model. Questions that name the drug keep the stored call.
    """
    if task.get("type") != "approval":
        return task
    question = task.get("question") or ""
    match = _QUESTION_APP_NUMBER.search(question)
    generic = ((task.get("entity") or {}).get("generic_name") or "").casefold()
    if match is None or (generic and generic in question.casefold()):
        return task
    number = f"{match.group(1).upper()}{match.group(2).zfill(6)}"
    calls = []
    for call in task.get("oracle_calls") or []:
        if call.get("tool") == "search_approvals":
            call = {"tool": "search_approvals", "arguments": {"application_number": number}}
        calls.append(call)
    return {**task, "oracle_calls": calls}


class OracleGenerator:
    """Emit a scripted tool trajectory and then the gold ``submit_answer``.

    The turn index is how many ``<tool_response>`` blocks are already in the prompt,
    which both codecs write. Temperature is ignored; the script is deterministic.
    Tasks that raise ``ScriptError`` are recorded on ``failures`` and omitted from
    ``plans``.
    """

    def __init__(self, tokenizer: CharTokenizer, codec, snapshot: Snapshot, tasks: list[dict]):
        self.tokenizer = tokenizer
        self.codec = codec
        self.tasks = list(tasks)
        self.plans: dict[str, list[dict]] = {}
        self.failures: dict[str, str] = {}
        self.batch_sizes: list[int] = []
        self.temperatures: list[float] = []
        for task in tasks:
            try:
                self.plans[task["id"]] = scripted_actions(snapshot, tasks, task)
            except ScriptError as exc:
                self.failures[task["id"]] = str(exc)

    def generate(self, prompt_token_ids, *, max_tokens: int, temperature: float):
        self.batch_sizes.append(len(prompt_token_ids))
        self.temperatures.append(temperature)
        turns = []
        for token_ids in prompt_token_ids:
            text = self.tokenizer.decode(token_ids)
            task = match_task(text, self.tasks)
            seen = text.count(TOOL_MARK)
            action = self.plans[task["id"]][min(seen, len(self.plans[task["id"]]) - 1)]
            rendered = self.codec.action_text(action)
            body = self.tokenizer.encode(rendered)[: max(max_tokens - 1, 0)]
            tokens = tuple(body + [self.tokenizer.eos_token_id])
            turns.append(TurnGeneration(tokens, tuple(-0.2 for _ in tokens)))
        return turns


def match_task(text: str, tasks: list[dict]) -> dict:
    found = [task for task in tasks if task["question"] in text]
    if not found:
        raise RuntimeError("prompt does not contain a known question")
    return max(found, key=lambda task: len(task["question"]))


def resolve_task_ids(tasks: list[dict], requested: list[str] | None, *, n: int = 2) -> list[str]:
    """Pick dry-run ids from the tasks that were actually loaded.

    Fixture FG-0001 / FG-0002 live on TEST in the 150-drug sheet. Defaulting to
    those ids against ``load_split("train")`` KeyErrors. With no ``--task-id``,
    take the first ``n`` non-``label_open`` rows in ``tasks`` order (train is
    concatenated first by the snapshot CLI).
    """
    by_id = {task["id"]: task for task in tasks}
    if requested:
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown task id(s): {', '.join(missing)}")
        return list(requested)
    chosen: list[str] = []
    for task in tasks:
        if task.get("type") == "label_open":
            continue
        chosen.append(task["id"])
        if len(chosen) == n:
            break
    if not chosen:
        raise ValueError("no gradable tasks to dry-run")
    return chosen


def run_rollout_dry(
    snapshot: Snapshot,
    tasks: list[dict],
    task_ids: list[str],
    *,
    limits: RolloutLimits | None = None,
    codec=None,
    dense: bool = True,
    bonus: float = 0.05,
) -> dict:
    limits = limits or RolloutLimits()
    codec = codec or QwenToolCodec()
    by_id = {task["id"]: task for task in tasks}
    chosen = [by_id[task_id] for task_id in task_ids]
    tokenizer = CharTokenizer()
    generator = OracleGenerator(tokenizer, codec, snapshot, chosen)
    rollouts = rollout_batch(
        snapshot,
        tasks,
        task_ids,
        tokenizer,
        generator,
        codec,
        limits,
        run_id="rollout-dry",
        seed=0,
    )
    episodes = []
    for rollout in rollouts:
        model_tokens = sum(len(span.token_ids) for span in rollout.spans if span.role == "model")
        tool_tokens = sum(len(span.token_ids) for span in rollout.spans if span.role == "tool")
        _strict, base, shaped, _hops = episode_scores(
            rollout, by_id[rollout.task_id], snapshot, dense=dense, bonus=bonus
        )
        episodes.append(
            {
                "task_id": rollout.task_id,
                "n_turns": rollout.n_turns,
                "terminated_by": rollout.terminated_by,
                "reward": rollout.reward,
                "model_tokens": model_tokens,
                "tool_tokens": tool_tokens,
                "loss_on": sum(rollout.loss_mask),
                "parsed_tool_calls": rollout.parsed_tool_calls,
                "dense_total": base,
                "training_reward": shaped,
            }
        )
    return {
        "codec": codec.name,
        "dense_reward": dense,
        "batch_sizes": generator.batch_sizes,
        "episodes": episodes,
        "rollouts": rollouts,
    }


def format_report(report: dict) -> str:
    lines = [f"codec={report['codec']} generate_batches={report['batch_sizes']}"]
    for episode in report["episodes"]:
        reward = episode["reward"]
        lines.append(
            f"{episode['task_id']} turns={episode['n_turns']} terminated={episode['terminated_by']} "
            f"total={reward['total']:.4f} correctness={reward['correctness']:.4f} "
            f"citation={reward['citation']:.4f} format={reward['format']:.4f} "
            f"efficiency={reward['efficiency']:.4f} "
            f"model_tokens={episode['model_tokens']} tool_tokens={episode['tool_tokens']} "
            f"loss_on={episode['loss_on']} parsed_tools={episode['parsed_tool_calls']} "
            f"dense={episode['dense_total']:.4f} training_reward={episode['training_reward']:.4f}"
        )
    return "\n".join(lines)
