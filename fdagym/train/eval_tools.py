"""Greedy multi-turn eval. The generator is injected so tests skip torch."""

from __future__ import annotations

from pathlib import Path

from fdagym.agents.baseline import (
    SKIP_TYPES,
    append_trajectory,
    jsonl_path,
    load_split,
    read_trajectories,
    summarize,
    write_summaries,
    write_task_sidecar,
)
from fdagym.openfda import DiskSnapshot
from fdagym.snapshot import Snapshot
from fdagym.train.config import QWEN_3B
from fdagym.train.eval_local import _dense_summary, reset_eval_outputs
from fdagym.train.rollout import QwenToolCodec, RolloutLimits, rollout_batch
from fdagym.train.tool_score import score_dense

EVAL_CHUNK = 4
PROTOCOL = "tool_rollout"
TOOL_BASELINE_ID = "baseline-qwen3b-tools"
TOOL_EVAL_ID = "grpo-qwen3b-tools-eval"


def tool_eval_model(adapter: str | None, *, base_name: str = QWEN_3B) -> dict:
    """Raw base model when ``adapter`` is empty. Same rollout, no LoRA."""
    if adapter:
        return {"name": f"{base_name}+tools", "kind": "trained", "checkpoint_step": None}
    return {"name": base_name, "kind": "baseline", "checkpoint_step": None}


def eval_tool_tasks(
    *,
    snapshot: Snapshot,
    tasks: list[dict],
    tokenizer,
    generator,
    model: dict,
    run_id: str,
    results_dir: Path,
    codec=None,
    limits: RolloutLimits | None = None,
    split: str = "test",
    skip_types: frozenset[str] = SKIP_TYPES,
    limit: int | None = None,
    chunk_size: int = EVAL_CHUNK,
    memorization: dict | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    adapter: str | None = None,
) -> dict:
    """Score a multi-turn generator on the split. Temperature lives on ``limits``."""
    limits = limits or RolloutLimits(temperature=0.0)
    codec = codec or QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    runnable = [task for task in tasks if task.get("type") not in skip_types]
    if limit is not None:
        runnable = runnable[:limit]
    skipped = {name: sum(1 for task in tasks if task.get("type") == name) for name in skip_types}
    reset_eval_outputs(results_dir, run_id)
    out = jsonl_path(results_dir, run_id)
    by_id = {task["id"]: task for task in tasks}
    task_ids = [task["id"] for task in runnable]
    dense_rows: list[dict] = []
    for start in range(0, len(task_ids), chunk_size):
        batch = rollout_batch(
            snapshot,
            tasks,
            task_ids[start : start + chunk_size],
            tokenizer,
            generator,
            codec,
            limits,
            run_id=run_id,
            seed=0,
            model=model,
        )
        for rollout in batch:
            append_trajectory(out, rollout.trajectory)
            dense = score_dense(rollout.trajectory, by_id[rollout.task_id], snapshot)
            dense_rows.append(dense)
            print(
                f"{model['name']} {rollout.task_id} R={rollout.reward['total']:.3f} "
                f"dense={dense['total']:.3f} turns={rollout.n_turns} parsed={rollout.parsed_tool_calls}",
                flush=True,
            )
    summary = summarize(read_trajectories(out), skipped=skipped)
    summary["dense_reward"] = _dense_summary(dense_rows)
    summary["run_id"] = run_id
    summary["split"] = split
    summary["seeds"] = [0]
    summary["eval_temperature"] = limits.temperature
    summary["prompt_version"] = limits.prompt_version
    summary["tool_budget"] = limits.tool_budget
    summary["cost_usd_total"] = 0.0
    summary["protocol"] = PROTOCOL
    if memorization is not None:
        summary["memorization"] = memorization
    run_dir = results_dir / run_id
    write_summaries(run_dir, summary)
    write_task_sidecar(results_dir, run_id, tasks)
    from fdagym.train.wandb_log import log_eval_summary

    log_eval_summary(
        run_id=run_id,
        summary=summary,
        tasks=tasks,
        trajectories=read_trajectories(out),
        prompt_version=limits.prompt_version,
        tool_budget=limits.tool_budget,
        adapter=adapter,
        model=model.get("name"),
        project=wandb_project,
        entity=wandb_entity,
    )
    return summary


def eval_tools_from_disk(
    *,
    tokenizer,
    generator,
    model: dict,
    run_id: str,
    split: str = "test",
    results_dir: Path | None = None,
    snapshot_version: str = "2026-10-15",
    **kwargs,
) -> dict:
    root = Path(__file__).resolve().parents[2]
    results_dir = results_dir or (root / "results")
    return eval_tool_tasks(
        snapshot=DiskSnapshot(snapshot_version),
        tasks=load_split(split),
        tokenizer=tokenizer,
        generator=generator,
        model=model,
        run_id=run_id,
        results_dir=results_dir,
        split=split,
        **kwargs,
    )


def live_memorization_measure(trainer, *, limits: RolloutLimits, codec, run_id: str):
    """Measure known-drug holdout vs hard eval on the live engine.

    The GRPO log callback calls this outside a training rollout, so the
    eval-only swapped-fact episodes stay legal. Temperature is greedy.
    """
    from dataclasses import replace

    from fdagym.train.rollout import VLLMGenerator

    eval_limits = replace(limits, temperature=0.0)

    def measure(step: int) -> dict:
        return eval_memorization(
            snapshot=trainer.snapshot,
            tokenizer=trainer.rollout_tokenizer,
            generator=VLLMGenerator(trainer.llm),
            model={"name": f"{run_id}-step{step}", "kind": "trained", "checkpoint_step": step},
            limits=eval_limits,
            codec=codec,
            run_id=f"{run_id}-mem-{step}",
            seed=0,
        )

    return measure


def eval_memorization(
    *,
    snapshot: Snapshot,
    tokenizer,
    generator,
    model: dict,
    limits: RolloutLimits,
    codec=None,
    run_id: str = "memorization",
    seed: int = 0,
) -> dict:
    """Roll the known-drug holdout, the hard eval, and the swapped-fact slice. No training pool."""
    from fdagym.agents.baseline import TASKS_DIR
    from fdagym.memorization import memorization_report
    from fdagym.swapped import rollout_swapped, select_fact_swaps

    codec = codec or QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)

    def _split(name: str) -> list[dict]:
        if not (TASKS_DIR / f"{name}.jsonl").exists():
            return []
        return load_split(name)

    known = _split("hard_train_holdout")
    unseen = _split("hard") or _split("public")
    plans = select_fact_swaps(known)
    known_rows = _episode_pairs(
        snapshot, known, tokenizer, generator, model, limits, codec, run_id=f"{run_id}-known", seed=seed
    )
    unseen_rows = _episode_pairs(
        snapshot, unseen, tokenizer, generator, model, limits, codec, run_id=f"{run_id}-unseen", seed=seed
    )
    swapped_rows = rollout_swapped(
        snapshot,
        known,
        plans,
        tokenizer,
        generator,
        codec,
        limits,
        run_id=f"{run_id}-swapped",
        seed=seed,
        model=model,
    )
    return memorization_report(known=known_rows, unseen=unseen_rows, swapped=swapped_rows)


def _episode_pairs(snapshot, tasks, tokenizer, generator, model, limits, codec, *, run_id: str, seed: int):
    by_id = {task["id"]: task for task in tasks}
    rows = []
    ids = [task["id"] for task in tasks]
    for start in range(0, len(ids), EVAL_CHUNK):
        batch = rollout_batch(
            snapshot,
            tasks,
            ids[start : start + EVAL_CHUNK],
            tokenizer,
            generator,
            codec,
            limits,
            run_id=run_id,
            seed=seed,
            model=model,
        )
        for rollout in batch:
            rows.append((by_id[rollout.task_id], rollout.trajectory))
    return rows
