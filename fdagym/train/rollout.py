"""Batched multi-turn rollouts against FDAEnv.

``ToolGRPOTrainer`` in ``fdagym/train/grpo_tools.py`` calls this runner. TRL stays
at 0.21. This module returns token ids plus two masks: attention over every
completion token, loss on model-written tokens only.

Nothing in this module imports torch, TRL, or vLLM. ``generate_with_vllm``
adapts a vLLM 0.8 ``LLM`` object the caller already holds (``trainer.llm``).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass

from fdagym import queries
from fdagym.env import FDAEnv
from fdagym.snapshot import Snapshot
from fdagym.tools import TOOLS, faers_modes

TOOL_MARK = "<tool_response>"
ASSISTANT_TURN = "<|im_start|>assistant\n"
_TRUNCATED = "...[truncated]"
_FORCED_SUBMIT = {"tool": "submit_answer", "arguments": {"answer": "NOT_FOUND", "citations": []}}
_DEFAULT_MODEL = {"name": "rollout", "kind": "trained", "checkpoint_step": None}

# New tool-mode runs and evals. v3 at budget 6 stays available for a past run.
DEFAULT_TOOL_BUDGET = 8
DEFAULT_PROMPT_VERSION = "v4"
PROMPT_VERSIONS = ("v3", "v4")

# JP's system prompt for the v3 retrain (same text as the paid baseline, without the worked example).
# The budget sentence is the only part ``system_prompt`` rewrites, and only when the enforced budget is not 6.
TOOL_SYSTEM_PROMPT = """\
You are an FDA-Gym agent. Answer only from tool outputs returned in THIS episode.
Do not use memorized FDA facts. If a tool result does not contain the answer, call another tool or submit NOT_FOUND.

Call submit_answer when you are done. Its arguments must match schemas/answer.schema.json:
  {"answer": <string|number|boolean>, "citations": [{"endpoint": "...", "id": "...", "field": "...", "quote": "..."}]}
- endpoint is one of drug/label, drug/event, drug/drugsfda.
- id is a set_id, application_number, or "step:<n>" for FAERS aggregates and empty searches (n is the tool-call index that returned them).
- field is a path the tool actually returned: boxed_warning, sections, submissions, products, sponsor_name, results, meta.results.total, ...
- search_approvals accepts application_number (NDA020977, NDA 020977, 020977, or a BLA/ANDA number) as well as generic_name. Use the application number when the question names one.
- search_approvals returns one page (page starts at 1). NDA and BLA rows come before ANDA, and submissions list only the original approval.
- quote is copied from that field in THIS episode's tool output (verbatim span, not a paraphrase). Quotes shorter than 4 characters never count.
- MedDRA reaction terms are uppercase with British spelling; copy the term from results, do not translate it.
- A missing boxed warning is cited from search_drug_label: id = the label's set_id, field = sections.
- NOT_FOUND is only correct when the snapshot has no record, and only after you searched.
- You get at most 6 tool calls, then you must submit_answer."""

_V3_BUDGET_LINE = "- You get at most 6 tool calls, then you must submit_answer."


def _budget_line(tool_budget: int) -> str:
    return f"- You get at most {tool_budget} tool calls, then you must submit_answer."


def _faers_bullet() -> str:
    """One system-prompt line. Return shapes are what ``_adverse_events`` actually writes."""
    modes = faers_modes()
    if modes != ("top_reactions", "count"):
        raise RuntimeError(f"query_adverse_events modes changed to {modes}; update the v4 prompt")
    year_from = min(queries.FAERS_YEARS)
    year_to = max(queries.FAERS_YEARS)
    return (
        "- query_adverse_events requires generic_name and mode. "
        f"mode is {modes[0]} or {modes[1]}. "
        f"top_reactions returns at most {queries.TOP_REACTIONS_LIMIT} MedDRA terms "
        "(each term and count) in results, under id step:<n>. "
        "count returns how many reports match in meta.results.total and leaves results empty, "
        "under the same id. "
        "serious may only be true; omit it for all reports. "
        f"year, when set, must be from {year_from} through {year_to}. "
        "serious and year filter both modes.\n"
    )


def system_prompt(version: str, tool_budget: int) -> str:
    """v3 is JP's prompt with the budget number filled in. v4 adds the FAERS argument line.

    ``system_prompt("v3", 6)`` is the historical string, byte for byte.
    """
    if isinstance(tool_budget, bool) or not isinstance(tool_budget, int) or tool_budget < 1:
        raise ValueError("tool_budget must be an integer >= 1")
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"prompt_version must be one of {', '.join(PROMPT_VERSIONS)}")
    if not TOOL_SYSTEM_PROMPT.endswith(_V3_BUDGET_LINE):
        raise RuntimeError("v3 prompt no longer ends with the budget sentence")
    body = TOOL_SYSTEM_PROMPT[: -len(_V3_BUDGET_LINE)]
    if version == "v4":
        body += _faers_bullet()
    return body + _budget_line(tool_budget)


@dataclass(frozen=True)
class RolloutLimits:
    """Approved tool-rollout budgets.

    ``tool_budget`` is what ``FDAEnv`` enforces. ``max_turns`` defaults to one
    past that, so a submit can still be written after the last tool call.
    ``prompt_version`` selects the system prompt, whose budget sentence is
    rendered from ``tool_budget``. New tool-mode runs use budget 8 and v4.
    ``tool_budget=6`` and ``prompt_version="v3"`` reproduce a v3 run.
    """

    max_turns: int | None = None
    max_new_tokens: int = 256
    max_tool_chars: int = 3500
    max_context_tokens: int = 8192
    temperature: float = 0.7
    force_submit: bool = False
    tool_budget: int = DEFAULT_TOOL_BUDGET
    prompt_version: str = DEFAULT_PROMPT_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.tool_budget, bool) or not isinstance(self.tool_budget, int) or self.tool_budget < 1:
            raise ValueError("tool_budget must be an integer >= 1")
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"prompt_version must be one of {', '.join(PROMPT_VERSIONS)}")
        if self.max_turns is None:
            object.__setattr__(self, "max_turns", self.tool_budget + 1)
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.max_tool_chars < 1:
            raise ValueError("max_tool_chars must be >= 1")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")


@dataclass(frozen=True)
class TurnGeneration:
    """One completion from the engine. ``logprobs`` aligns with ``token_ids`` when present."""

    token_ids: tuple[int, ...]
    logprobs: tuple[float | None, ...] | None = None


@dataclass(frozen=True)
class Span:
    """A slice of the completion. ``role`` is ``model``, ``tool``, or ``injected``."""

    role: str
    token_ids: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class Rollout:
    task_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    attention_mask: tuple[int, ...]
    logprobs: tuple[float | None, ...]
    spans: tuple[Span, ...]
    reward: dict
    trajectory: dict
    n_turns: int
    terminated_by: str
    parsed_tool_calls: int
    # Turns whose text parsed into an action (submit included). parse failures end the episode.
    parsed_turns: int = 0
    # Set when the runner closed the episode: "turn_limit" or "context". None when the model ended it.
    closed_by: str | None = None


class _Episode:
    def __init__(self, task_id: str, env: FDAEnv, prompt_ids: list[int]):
        self.task_id = task_id
        self.env = env
        self.prompt_ids = prompt_ids
        self.spans: list[Span] = []
        self.logprobs: list[float | None] = []
        self.n_turns = 0
        self.parsed_tool_calls = 0
        self.parsed_turns = 0
        self.closed_by: str | None = None
        self.done = False
        self.reward: dict | None = None
        self.trajectory: dict | None = None
        self.terminated_by: str | None = None

    def completion_ids(self) -> list[int]:
        ids: list[int] = []
        for span in self.spans:
            ids.extend(span.token_ids)
        return ids

    def room(self, limits: RolloutLimits) -> int:
        return limits.max_context_tokens - len(self.prompt_ids) - len(self.completion_ids())


_ROLLOUT_DEPTH = 0


@contextmanager
def training_rollout():
    """Mark the FDAEnv episodes a GRPO step is playing. Eval wrappers must stay outside."""
    global _ROLLOUT_DEPTH
    _ROLLOUT_DEPTH += 1
    try:
        yield
    finally:
        _ROLLOUT_DEPTH -= 1


def training_rollout_active() -> bool:
    return _ROLLOUT_DEPTH > 0


def rollout_batch(
    snapshot: Snapshot,
    tasks: list[dict],
    task_ids: list[str],
    tokenizer,
    generator,
    codec,
    limits: RolloutLimits | None = None,
    *,
    run_id: str = "rollout",
    seed: int = 0,
    model: dict | None = None,
) -> list[Rollout]:
    """Run one episode per ``task_ids`` entry. Duplicates are independent episodes.

    ``generator.generate(prompt_token_ids, *, max_tokens, temperature)`` returns
    one ``TurnGeneration`` per prompt, in order. Prompts are token ids, not text,
    so a model span is never retokenized. Episodes are plain ``FDAEnv``.
    """
    limits = limits or RolloutLimits()
    with training_rollout():
        episodes = [
            _open_episode(snapshot, tasks, task_id, tokenizer, codec, index, run_id, seed, model, limits)
            for index, task_id in enumerate(task_ids)
        ]
        return run_episodes(episodes, tokenizer, generator, codec, limits)


def run_episodes(episodes, tokenizer, generator, codec, limits: RolloutLimits) -> list[Rollout]:
    """Play episodes that are already open. Training and the eval wrapper share this loop."""
    for _turn in range(limits.max_turns):
        _close_if_full(episodes, tokenizer, codec, limits)
        active = [episode for episode in episodes if not episode.done and episode.room(limits) > 0]
        if not active:
            break
        _generate_turn(active, tokenizer, generator, codec, limits)
    for episode in episodes:
        if not episode.done:
            _end_unfinished(episode, tokenizer, codec, limits, reason="turn_limit")
    return [_finish(episode) for episode in episodes]


def open_provided_env(env, task_id: str, tokenizer, codec, limits: RolloutLimits):
    """Bind a prompt to an env the caller already built. Refuses a training rollout."""
    if training_rollout_active():
        raise RuntimeError("eval env wrappers cannot open inside a training rollout")
    return _bind_prompt(env, task_id, tokenizer, codec, limits)


def generate_with_vllm(llm, prompt_token_ids, *, max_tokens: int, temperature: float, sampling_params_cls):
    """Call a vLLM 0.8 ``LLM.generate`` with token prompts.

    ``sampling_params_cls`` is ``vllm.SamplingParams``. Passing it in keeps this
    module importable when vLLM is not installed. Colocate mode scores one
    sample per prompt (``n=1``); the caller repeats prompts for the GRPO group.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    params = sampling_params_cls(n=1, temperature=temperature, max_tokens=max_tokens, logprobs=1)
    prompts = [{"prompt_token_ids": list(ids)} for ids in prompt_token_ids]
    results = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    if len(results) != len(prompt_token_ids):
        raise RuntimeError(f"vLLM returned {len(results)} outputs for {len(prompt_token_ids)} prompts")
    turns = []
    for result in results:
        output = result.outputs[0]
        token_ids = tuple(output.token_ids)
        turns.append(TurnGeneration(token_ids, tuple(chosen_logprobs(token_ids, output.logprobs))))
    return turns


class VLLMGenerator:
    """``trainer.llm`` adapter. Imports vLLM only when ``generate`` runs."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt_token_ids, *, max_tokens: int, temperature: float):
        from vllm import SamplingParams

        return generate_with_vllm(
            self.llm,
            prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            sampling_params_cls=SamplingParams,
        )


def chosen_logprobs(token_ids, logprobs) -> list[float | None]:
    """Pick the sampled token's logprob out of vLLM's per-position dicts."""
    if not token_ids:
        return []
    if logprobs is None:
        return [None] * len(token_ids)
    chosen: list[float | None] = []
    for index, token_id in enumerate(token_ids):
        if index >= len(logprobs) or not isinstance(logprobs[index], dict):
            chosen.append(None)
            continue
        item = logprobs[index].get(token_id)
        if item is None:
            chosen.append(None)
            continue
        value = item.logprob if hasattr(item, "logprob") else item
        chosen.append(float(value))
    return chosen


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATED):
        return _TRUNCATED[:max_chars]
    return text[: max_chars - len(_TRUNCATED)] + _TRUNCATED


def tool_catalog() -> str:
    return "\n".join(f"- {tool['function']['name']}: {tool['function']['description']}" for tool in TOOLS)


class JsonActionCodec:
    """One JSON object per turn: ``{"tool", "arguments"}``. The CPU dry run uses this."""

    name = "json"

    def prompt(self, question: str, *, tool_budget: int | None = None, prompt_version: str | None = None) -> str:
        del tool_budget, prompt_version
        return (
            "You answer FDA-Gym questions using only tool results from this episode.\n"
            "Reply with one JSON object and nothing else:\n"
            '{"tool": "<name>", "arguments": {}}\n'
            "Call submit_answer when you are done. Its arguments are answer and citations.\n\n"
            f"Tools:\n{tool_catalog()}\n\n"
            f"Question:\n{question}\n"
        )

    def parse(self, text: str) -> dict | None:
        for obj in json_objects(text):
            action = action_from_obj(obj)
            if action is not None:
                return action
        return None

    def tool_message(self, observation: dict, *, max_chars: int) -> str:
        payload = clip_text(observation_payload(observation), max_chars)
        return f"{TOOL_MARK}\n{payload}\n</tool_response>"

    def action_text(self, action: dict) -> str:
        body = {"tool": action["tool"], "arguments": action.get("arguments") or {}}
        return json.dumps(body, ensure_ascii=False)


class QwenToolCodec:
    """Qwen2.5 Hermes-style ``<tool_call>`` blocks. Prompt text is a stand-in, not ``apply_chat_template``.

    Defaults are v4 and budget 8. A rollout passes ``tool_budget`` and
    ``prompt_version`` into ``prompt`` so the text matches ``RolloutLimits``.
    """

    name = "qwen"

    def __init__(
        self,
        *,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        tool_budget: int = DEFAULT_TOOL_BUDGET,
    ):
        # Validate once, up front. ``prompt`` calls ``system_prompt`` again with the rollout's values.
        system_prompt(prompt_version, tool_budget)
        self.prompt_version = prompt_version
        self.tool_budget = tool_budget

    def prompt(self, question: str, *, tool_budget: int | None = None, prompt_version: str | None = None) -> str:
        version = self.prompt_version if prompt_version is None else prompt_version
        budget = self.tool_budget if tool_budget is None else tool_budget
        return (
            "<|im_start|>system\n"
            f"{system_prompt(version, budget)}\n\n"
            "# Tools\n"
            "For each function call, return a json object inside <tool_call></tool_call> tags.\n"
            f"{tool_catalog()}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def parse(self, text: str) -> dict | None:
        for body in tool_call_bodies(text):
            for obj in json_objects(body):
                action = action_from_obj(obj)
                if action is not None:
                    return action
        for obj in json_objects(text):
            action = action_from_obj(obj)
            if action is not None:
                return action
        return None

    def tool_message(self, observation: dict, *, max_chars: int) -> str:
        payload = clip_text(observation_payload(observation), max_chars)
        # The assistant header is canonicalized by us, so it stays inside this masked span.
        return f"<|im_start|>user\n{TOOL_MARK}\n{payload}\n</tool_response><|im_end|>\n{ASSISTANT_TURN}"

    def action_text(self, action: dict) -> str:
        body = {"name": action["tool"], "arguments": action.get("arguments") or {}}
        return "<tool_call>\n" + json.dumps(body, ensure_ascii=False) + "\n</tool_call>"


def observation_payload(observation: dict) -> str:
    body = {
        "tool": observation.get("tool"),
        "output": observation.get("output"),
        "error": observation.get("error"),
        "tool_calls_remaining": observation.get("tool_calls_remaining"),
    }
    return json.dumps(body, ensure_ascii=False, default=str)


def action_from_obj(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    if "answer" in obj and "citations" in obj and "tool" not in obj and "name" not in obj:
        return {"tool": "submit_answer", "arguments": {"answer": obj["answer"], "citations": obj["citations"]}}
    name = obj.get("tool", obj.get("name"))
    return coerce_action(name, obj.get("arguments", {}))


def coerce_action(name, arguments) -> dict | None:
    if not isinstance(name, str) or not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return {"tool": name, "arguments": arguments}


def json_objects(text: str):
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield obj
        index = start + end


def tool_call_bodies(text: str):
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    position = 0
    while True:
        start = text.find(start_tag, position)
        if start < 0:
            return
        body_start = start + len(start_tag)
        end = text.find(end_tag, body_start)
        if end < 0:
            return
        yield text[body_start:end]
        position = end + len(end_tag)


def _open_episode(
    snapshot, tasks, task_id, tokenizer, codec, index, run_id, seed, model, limits: RolloutLimits
) -> _Episode:
    env = FDAEnv(
        snapshot,
        tasks,
        run_id=run_id,
        seed=seed + index,
        model=dict(model or _DEFAULT_MODEL),
        tool_budget=limits.tool_budget,
    )
    return _bind_prompt(env, task_id, tokenizer, codec, limits)


def _bind_prompt(env, task_id: str, tokenizer, codec, limits: RolloutLimits):
    obs = env.reset(task_id)
    prompt = codec.prompt(obs["question"], tool_budget=limits.tool_budget, prompt_version=limits.prompt_version)
    if isinstance(codec, QwenToolCodec):
        expected = system_prompt(limits.prompt_version, limits.tool_budget)
        if expected not in prompt:
            raise RuntimeError(
                f"system prompt does not match prompt_version={limits.prompt_version} tool_budget={limits.tool_budget}"
            )
    if env.tool_budget != limits.tool_budget:
        raise RuntimeError("env tool budget does not match rollout limits")
    return _Episode(task_id, env, list(tokenizer.encode(prompt)))


def _close_if_full(episodes, tokenizer, codec, limits: RolloutLimits) -> None:
    for episode in episodes:
        if not episode.done and episode.room(limits) <= 0:
            _end_unfinished(episode, tokenizer, codec, limits, reason="context")


def _generate_turn(active, tokenizer, generator, codec, limits: RolloutLimits) -> None:
    # One SamplingParams applies to a whole vLLM batch. Group by the room each
    # episode still has so a long episode cannot ask for more tokens than a
    # short one can legally append.
    groups: dict[int, list[_Episode]] = {}
    for episode in active:
        cap = min(limits.max_new_tokens, episode.room(limits))
        groups.setdefault(cap, []).append(episode)
    for cap, group in groups.items():
        prompts = [episode.prompt_ids + episode.completion_ids() for episode in group]
        generations = generator.generate(prompts, max_tokens=cap, temperature=limits.temperature)
        if len(generations) != len(group):
            raise RuntimeError(f"generator returned {len(generations)} turns for {len(group)} episodes")
        for episode, generation in zip(group, generations, strict=True):
            _consume_turn(episode, generation, cap, tokenizer, codec, limits)


def _consume_turn(episode, generation, limit, tokenizer, codec, limits: RolloutLimits) -> None:
    token_ids, logprobs = _take_tokens(generation, limit, tokenizer.eos_token_id)
    episode.n_turns += 1
    if token_ids:
        text = tokenizer.decode(_visible(token_ids, tokenizer))
        episode.spans.append(Span("model", tuple(token_ids), text))
        episode.logprobs.extend(logprobs)
    else:
        text = ""
    action = codec.parse(text)
    if action is not None:
        episode.parsed_turns += 1
    if action is not None and action.get("tool") != "submit_answer":
        episode.parsed_tool_calls += 1
    observation, done = _apply_step(episode, action if action is not None else text)
    if done or observation is None:
        return
    if not _append_tool(episode, observation, tokenizer, codec, limits):
        _end_unfinished(episode, tokenizer, codec, limits, reason="context")


def _take_tokens(generation: TurnGeneration, limit: int, eos_token_id: int) -> tuple[list[int], list[float | None]]:
    token_ids = list(generation.token_ids)[:limit]
    if generation.logprobs is None:
        logprobs = [None] * len(generation.token_ids)
    else:
        logprobs = list(generation.logprobs)
    logprobs = logprobs[:limit]
    if eos_token_id in token_ids:
        cut = token_ids.index(eos_token_id) + 1
        token_ids = token_ids[:cut]
        logprobs = logprobs[:cut]
    if len(logprobs) < len(token_ids):
        logprobs.extend([None] * (len(token_ids) - len(logprobs)))
    return token_ids, logprobs[: len(token_ids)]


def _visible(token_ids: list[int], tokenizer) -> list[int]:
    skip = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    return [token_id for token_id in token_ids if token_id not in skip]


def _append_tool(episode, observation, tokenizer, codec, limits: RolloutLimits) -> bool:
    text = codec.tool_message(observation, max_chars=limits.max_tool_chars)
    token_ids = list(tokenizer.encode(text))
    if not token_ids:
        return True
    if len(token_ids) > episode.room(limits):
        return False
    episode.spans.append(Span("tool", tuple(token_ids), text))
    episode.logprobs.extend([None] * len(token_ids))
    return True


def _apply_step(episode, action) -> tuple[dict | None, bool]:
    _observation, _reward, done, info = episode.env.step(action)
    if done:
        episode.done = True
        episode.reward = info["reward"]
        episode.trajectory = info["trajectory"]
        episode.terminated_by = episode.env.terminated_by
        return None, True
    return _observation, False


def _end_unfinished(episode, tokenizer, codec, limits: RolloutLimits, *, reason: str) -> None:
    if episode.done:
        return
    episode.closed_by = reason
    if limits.force_submit:
        text = codec.action_text(_FORCED_SUBMIT)
        token_ids = list(tokenizer.encode(text))[: max(episode.room(limits), 0)]
        if token_ids:
            episode.spans.append(Span("injected", tuple(token_ids), tokenizer.decode(token_ids)))
            episode.logprobs.extend([None] * len(token_ids))
        _apply_step(episode, _FORCED_SUBMIT)
        return
    _apply_step(episode, {"tool": "submit_answer", "arguments": {}})


def _finish(episode: _Episode) -> Rollout:
    if episode.reward is None or episode.trajectory is None or episode.terminated_by is None:
        raise RuntimeError(f"{episode.task_id} ended without a reward")
    completion_ids = tuple(episode.completion_ids())
    loss_bits: list[int] = []
    for span in episode.spans:
        bit = 1 if span.role == "model" else 0
        loss_bits.extend([bit] * len(span.token_ids))
    loss_mask = tuple(loss_bits)
    if len(loss_mask) != len(completion_ids) or len(episode.logprobs) != len(completion_ids):
        raise RuntimeError(f"{episode.task_id} mask length does not match completion")
    trajectory = episode.trajectory
    usage = trajectory["usage"]
    usage["prompt_tokens"] = len(episode.prompt_ids)
    usage["completion_tokens"] = len(completion_ids)
    return Rollout(
        task_id=episode.task_id,
        prompt_ids=tuple(episode.prompt_ids),
        completion_ids=completion_ids,
        loss_mask=loss_mask,
        attention_mask=tuple(1 for _ in completion_ids),
        logprobs=tuple(episode.logprobs),
        spans=tuple(episode.spans),
        reward=episode.reward,
        trajectory=trajectory,
        n_turns=episode.n_turns,
        terminated_by=episode.terminated_by,
        parsed_tool_calls=episode.parsed_tool_calls,
        parsed_turns=episode.parsed_turns,
        closed_by=episode.closed_by,
    )


class RolloutTokenizer:
    """Encode the codec's text without adding a second pair of special tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = tokenizer.eos_token_id if pad is None else pad

    def encode(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer.encode(text))

    def decode(self, token_ids) -> str:
        try:
            return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)
        except TypeError:
            return self.tokenizer.decode(list(token_ids))
