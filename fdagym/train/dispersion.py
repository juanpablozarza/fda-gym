"""Reward dispersion of one GRPO group per question, before a hard-mix run.

GRPO's advantage is zero when all ``num_generations`` completions of a question
share a reward. ``eval_tools`` cannot measure that: it is greedy, one episode
per task, and it samples with Hugging Face ``generate`` (Qwen's generation
config), not with the vLLM ``SamplingParams`` inside ``VLLMGenerator``.

This module rebuilds the prompt pool ``train_tools`` builds at
``hard_train_mix=0.6`` and scores rollouts with ``episode_scores``, the same
function ``ToolGRPOTrainer._rollout_scored`` calls. It does not implement a
reward. Sampling stays in ``VLLMGenerator`` / ``generate_with_vllm``; nothing
here constructs a sampler.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

from fdagym.train.curriculum import GRPO_POOL_TYPES_FLAG
from fdagym.train.group_sample import tied_group_flags
from fdagym.train.grpo_tools import build_tool_rows
from fdagym.train.hard_mix import hard_mix_weights, rollout_tasks
from fdagym.train.prompt_filter import blocked_eval_ids
from fdagym.train.rollout import QwenToolCodec, rollout_batch
from fdagym.train.tool_config import TOOL_TEMPERATURE, ToolTrainConfig, configure_tool_run
from fdagym.train.tool_score import episode_scores, frac_zero_std, score_dense

# The next hard GRPO run. ``train_tools`` turns the fraction into weights.
PROBE_HARD_TRAIN_MIX = 0.6
PROBE_TOOL_BUDGET = 8
PROBE_PROMPT_VERSION = "v4"
PROBE_GROUP_SIZE = 8
# Same stride ``ToolGRPOTrainer._sample_groups`` adds between resample rounds.
_SEED_STRIDE = 10007


def ids_sha256(task_ids: list[str]) -> str:
    """Order-sensitive digest of ``task_ids``. No trailing newline, ids are not sorted."""
    return hashlib.sha256("\n".join(task_ids).encode()).hexdigest()


def sampling_param_kwargs(temperature: float, max_tokens: int) -> dict:
    """The arguments ``generate_with_vllm`` passes to ``vllm.SamplingParams``.

    ``top_p``, ``top_k``, and ``repetition_penalty`` are absent on purpose.
    vLLM then uses its own defaults (1.0, -1, 1.0), which is what GRPO samples
    with. Do not fill them in here. ``max_tokens`` on a live turn is
    ``min(max_new_tokens, room)``; the recorded object uses the 256 cap.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    return {"n": 1, "temperature": float(temperature), "max_tokens": max_tokens, "logprobs": 1}


def probe_task_ids(
    task_ids: list[str],
    task_ids_sha256: str,
    *,
    hard_train_mix: float = PROBE_HARD_TRAIN_MIX,
    tool_budget: int = PROBE_TOOL_BUDGET,
    prompt_version: str = PROBE_PROMPT_VERSION,
    temperature: float = TOOL_TEMPERATURE,
    pool_types: str | tuple[str, ...] | list[str] | None = GRPO_POOL_TYPES_FLAG,
) -> tuple[ToolTrainConfig, list[dict], list[str]]:
    """Rebuild the pool ``train_tools(hard_train_mix=0.6)`` trains on and keep ``task_ids``.

    The sha is ``sha256("\\n".join(task_ids))`` in the caller's order. Every id
    has to be a row of that pool. Eval, holdout, and test ids are not rows, so
    they fail here, before any model load.
    """
    _require_probe_request(task_ids, hard_train_mix, tool_budget, prompt_version)
    digest = ids_sha256(task_ids)
    if digest != task_ids_sha256:
        raise ValueError(f"task_ids sha256 mismatch: got {digest}, expected {task_ids_sha256}")

    from fdagym.agents.baseline import load_split

    weights = hard_mix_weights(hard_train_mix)
    if weights is None:
        raise ValueError("probe requires hard_train_mix 0.6")
    tool = configure_tool_run(
        hard=True,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        hard_train_mix=weights,
        temperature=temperature,
        pool_types=pool_types,
    )
    _require_training_reward(tool)
    tasks = load_split("train")
    hard_tasks = load_split("hard_train")
    codec = QwenToolCodec(prompt_version=tool.limits.prompt_version, tool_budget=tool.limits.tool_budget)
    rows, _mix = build_tool_rows(
        tasks,
        tool.train,
        codec,
        type_mix=tool.type_mix,
        mix_rows=tool.mix_rows,
        mix_seed=tool.mix_seed,
        hard_train_mix=weights,
        hard_tasks=hard_tasks,
        prompt_filter=tool.prompt_filter,
        blocked_ids=blocked_eval_ids(),
        pool_types=tool.pool_types,
    )
    if len(rows) != tool.mix_rows:
        raise RuntimeError(f"GRPO pool has {len(rows)} rows, train_tools builds {tool.mix_rows}")
    pool_ids = {row["task_id"] for row in rows}
    blocked = blocked_eval_ids()
    leaked = [task_id for task_id in task_ids if task_id in blocked]
    if leaked:
        shown = ", ".join(leaked[:8])
        raise ValueError(f"probe ids include an eval, holdout, or test id ({shown})")
    outside = [task_id for task_id in task_ids if task_id not in pool_ids]
    if outside:
        shown = ", ".join(outside[:8])
        raise ValueError(f"probe ids are not in the GRPO training pool at hard_train_mix={hard_train_mix} ({shown})")
    env_tasks = rollout_tasks(tasks, hard_tasks)
    env_ids = {task["id"] for task in env_tasks}
    missing = [task_id for task_id in task_ids if task_id not in env_ids]
    if missing:
        shown = ", ".join(missing[:8])
        raise RuntimeError(f"pool ids missing from rollout_tasks ({shown})")
    return tool, env_tasks, list(task_ids)


def run_probe(
    snapshot,
    env_tasks: list[dict],
    task_ids: list[str],
    tokenizer,
    generator,
    tool: ToolTrainConfig,
    *,
    k: int | None = None,
    seed: int = 0,
    out_path: str | Path,
    chunk_groups: int = 10,
    on_chunk=None,
    run_id: str = "probe",
) -> dict:
    """Sample ``k`` episodes per id and append one JSON object per episode.

    Ids are repeated consecutively, the way TRL's ``RepeatSampler`` feeds GRPO.
    Each chunk of ``chunk_groups`` questions is one ``rollout_batch``. The chunk
    is written and ``on_chunk`` runs before the next chunk starts, so a timeout
    leaves whole groups. ``k is None`` uses the training group size, 8. A caller
    that passes ``k`` (the CPU tests) may use a smaller group; the Modal entry
    passes 8.

    ``rollout_batch`` already opens ``training_rollout``. This function does not
    open another one. Rewards come from ``episode_scores`` with the tool config's
    ``dense_reward`` and ``tool_parse_bonus``.
    """
    _require_training_reward(tool)
    group_size = _group_size(tool, k)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(chunk_groups, bool) or not isinstance(chunk_groups, int) or chunk_groups < 1:
        raise ValueError("chunk_groups must be an integer >= 1")
    if not task_ids:
        raise ValueError("probe task_ids is empty")
    by_id = {task["id"]: task for task in env_tasks}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        shown = ", ".join(missing[:8])
        raise KeyError(f"env tasks have no row for {shown}")

    destination = Path(out_path)
    if destination.exists() and destination.stat().st_size > 0:
        raise FileExistsError(f"refusing to append to existing rollouts at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    codec = QwenToolCodec(prompt_version=tool.limits.prompt_version, tool_budget=tool.limits.tool_budget)
    shaped: list[float] = []
    chunks = [task_ids[start : start + chunk_groups] for start in range(0, len(task_ids), chunk_groups)]
    written = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_seed = seed + chunk_index * _SEED_STRIDE
        repeated = [task_id for task_id in chunk for _ in range(group_size)]
        rollouts = rollout_batch(
            snapshot,
            env_tasks,
            repeated,
            tokenizer,
            generator,
            codec,
            tool.limits,
            run_id=run_id,
            seed=chunk_seed,
        )
        if len(rollouts) != len(repeated):
            raise RuntimeError(f"rollout_batch returned {len(rollouts)} episodes for {len(repeated)} ids")
        records = []
        for offset, task_id in enumerate(chunk):
            group_index = written + offset
            for sample_index in range(group_size):
                rollout = rollouts[offset * group_size + sample_index]
                if rollout.task_id != task_id:
                    raise RuntimeError(
                        f"group {group_index} sample {sample_index} is {rollout.task_id}, expected {task_id}"
                    )
                records.append(
                    _episode_record(
                        rollout,
                        by_id[task_id],
                        snapshot,
                        tool,
                        group_index=group_index,
                        sample_index=sample_index,
                        seed=chunk_seed,
                    )
                )
        _append_chunk(destination, records)
        shaped.extend(float(record["shaped"]) for record in records)
        written += len(chunk)
        print(
            f"probe chunk {chunk_index + 1}/{len(chunks)} groups={len(chunk)} "
            f"episodes={len(records)} seed={chunk_seed}",
            flush=True,
        )
        if on_chunk is not None:
            on_chunk()
    return _dispersion_summary(shaped, group_size)


def _require_probe_request(task_ids: list[str], hard_train_mix: float, tool_budget: int, prompt_version: str) -> None:
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty list of strings")
    if not all(isinstance(task_id, str) and task_id for task_id in task_ids):
        raise ValueError("task_ids must be a non-empty list of strings")
    if isinstance(hard_train_mix, bool) or hard_train_mix != PROBE_HARD_TRAIN_MIX:
        raise ValueError(f"probe hard_train_mix must be {PROBE_HARD_TRAIN_MIX}, the train_tools pool")
    if tool_budget != PROBE_TOOL_BUDGET or prompt_version != PROBE_PROMPT_VERSION:
        raise ValueError(
            f"probe uses tool budget {PROBE_TOOL_BUDGET} and prompt {PROBE_PROMPT_VERSION}, matching train_tools"
        )


def _require_training_reward(tool: ToolTrainConfig) -> None:
    """The numbers GRPO trains with. A drift here would measure a different run."""
    limits = tool.limits
    if limits is None:
        raise RuntimeError("tool config has no rollout limits")
    expected = {
        "num_generations": (tool.train.num_generations, PROBE_GROUP_SIZE),
        "dense_reward": (tool.train.dense_reward, True),
        "tool_parse_bonus": (tool.tool_parse_bonus, 0.05),
        "max_new_tokens": (limits.max_new_tokens, 256),
        "max_turns": (limits.max_turns, PROBE_TOOL_BUDGET + 1),
        "max_tool_chars": (limits.max_tool_chars, 3500),
        "max_context_tokens": (limits.max_context_tokens, 8192),
        "tool_budget": (limits.tool_budget, PROBE_TOOL_BUDGET),
        "prompt_version": (limits.prompt_version, PROBE_PROMPT_VERSION),
    }
    drifted = [f"{name}={value}" for name, (value, want) in expected.items() if value != want]
    if drifted or limits.force_submit:
        if limits.force_submit:
            drifted.append("force_submit=True")
        raise ValueError("probe reward and limits must match tool GRPO (" + ", ".join(drifted) + ")")
    if not tool.prompt_filter.enabled:
        raise ValueError("probe pool must use the hard-run prompt filter train_tools enables at mix 0.6")
    if tool.train.temperature != limits.temperature:
        raise ValueError("rollout temperature and GRPO temperature differ")


def _group_size(tool: ToolTrainConfig, k: int | None) -> int:
    if tool.train.num_generations != PROBE_GROUP_SIZE:
        raise ValueError(f"GRPO num_generations is {tool.train.num_generations}, the probe expects {PROBE_GROUP_SIZE}")
    if k is None:
        return PROBE_GROUP_SIZE
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer >= 2")
    return k


def _episode_record(
    rollout,
    task: dict,
    snapshot,
    tool: ToolTrainConfig,
    *,
    group_index: int,
    sample_index: int,
    seed: int,
) -> dict:
    strict, dense, shaped, hop_rate = episode_scores(
        rollout,
        task,
        snapshot,
        dense=tool.train.dense_reward,
        bonus=tool.tool_parse_bonus,
    )
    components = score_dense(rollout.trajectory, task, snapshot)
    if abs(float(components["total"]) - dense) > 1e-9:
        raise RuntimeError(f"{rollout.task_id} dense total from episode_scores does not match score_dense")
    trajectory = rollout.trajectory
    return {
        "task_id": rollout.task_id,
        "group_index": group_index,
        "sample_index": sample_index,
        "seed": seed,
        "strict": strict,
        "dense": dense,
        "shaped": shaped,
        "hop_rate": hop_rate,
        "parsed_tool_calls": rollout.parsed_tool_calls,
        "parsed_turns": rollout.parsed_turns,
        "n_turns": rollout.n_turns,
        "terminated_by": rollout.terminated_by,
        "closed_by": rollout.closed_by,
        "components": {key: value for key, value in components.items() if key != "notes"},
        "trajectory": {
            "steps": trajectory.get("steps") or [],
            "final_answer": trajectory.get("final_answer"),
        },
    }


def _append_chunk(path: Path, records: list[dict]) -> None:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _dispersion_summary(shaped: list[float], group_size: int) -> dict:
    """Mixed-group rate from the two helpers GRPO already uses for tied groups."""
    flags = tied_group_flags(shaped, group_size)
    frac_tied = frac_zero_std(shaped, group_size)
    if abs(frac_tied - (sum(flags) / len(flags))) > 1e-12:
        raise RuntimeError("frac_zero_std and tied_group_flags disagree")
    stds = [statistics.pstdev(shaped[start : start + group_size]) for start in range(0, len(shaped), group_size)]
    return {
        "n_groups": len(flags),
        "n_episodes": len(shaped),
        "frac_mixed_shaped": 1.0 - frac_tied,
        "mean_within_std_shaped": sum(stds) / len(stds),
    }
