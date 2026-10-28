"""Plan A eval of a local HF model (base or LoRA) on a task split.

This is the number GRPO is allowed to beat: same prefetched docs, same ``score()``.
It is not the multi-turn OpenRouter baseline.
"""

from __future__ import annotations

from collections import defaultdict
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
from fdagym.rewards import score as score_reward
from fdagym.rewards.combine import WEIGHTS
from fdagym.snapshot import Snapshot
from fdagym.train.config import TrainConfig
from fdagym.train.context import render_task_prompt
from fdagym.train.score import run_completion

EVAL_TEMPERATURE = 0.0


def reset_eval_outputs(results_dir: Path, run_id: str) -> None:
    """Start a clean jsonl. Eval run ids are not the GRPO checkpoint directory."""
    jsonl_path(results_dir, run_id).unlink(missing_ok=True)
    run_dir = results_dir / run_id
    for name in ("summary.json", "leaderboard.json", "tasks.json"):
        target = run_dir / name
        if target.exists():
            target.unlink()


def eval_split(
    *,
    generate,
    snapshot: Snapshot,
    tasks: list[dict],
    model_name: str,
    run_id: str,
    results_dir: Path,
    split: str = "test",
    seeds: list[int] | None = None,
    kind: str = "baseline",
    skip_types: frozenset[str] = SKIP_TYPES,
    config: TrainConfig | None = None,
    checkpoint_step: int | None = None,
    limit: int | None = None,
) -> dict:
    """Score ``generate(messages) -> str`` on the split. ``generate`` is injected so tests skip torch."""
    config = config or TrainConfig()
    seeds = seeds or [1]
    runnable = [task for task in tasks if task.get("type") not in skip_types]
    if limit is not None:
        runnable = runnable[:limit]
    skipped = {name: sum(1 for task in tasks if task.get("type") == name) for name in skip_types}
    skipped["already_done"] = 0
    reset_eval_outputs(results_dir, run_id)
    out = jsonl_path(results_dir, run_id)
    run_dir = results_dir / run_id
    model = {"name": model_name, "kind": kind, "checkpoint_step": checkpoint_step}
    by_seed: dict[int, list[float]] = defaultdict(list)
    dense_rows: list[dict] = []
    for seed in seeds:
        for task in runnable:
            messages = render_task_prompt(
                snapshot,
                tasks,
                task,
                n_distractors=config.n_distractors,
                max_chars=config.max_pack_chars,
            )
            try:
                completion = generate(messages, seed=seed)
            except TypeError:
                completion = generate(messages)
            trajectory = run_completion(
                snapshot,
                tasks,
                task,
                completion,
                run_id=run_id,
                seed=seed,
                model=model,
            )
            trajectory["id"] = f"{run_id}/{model_name}/{task['id']}/s{seed}"
            trajectory["run_id"] = run_id
            append_trajectory(out, trajectory)
            dense = score_reward(
                {"steps": trajectory["steps"], "final_answer": trajectory["final_answer"]},
                task,
                dense=True,
                snapshot=snapshot,
            )
            dense_rows.append(dense)
            strict_total = float(trajectory["reward"]["total"])
            by_seed[seed].append(strict_total)
            print(
                f"{model_name} {task['id']} s{seed} R={strict_total:.3f} dense={dense['total']:.3f}",
                flush=True,
            )
    summary = summarize(read_trajectories(out), skipped=skipped)
    summary["dense_reward"] = _dense_summary(dense_rows)
    summary["run_id"] = run_id
    summary["split"] = split
    summary["seeds"] = seeds
    summary["seed_means"] = [sum(vals) / len(vals) for _, vals in sorted(by_seed.items())]
    summary["eval_temperature"] = EVAL_TEMPERATURE
    summary["cost_usd_total"] = 0.0
    summary["protocol"] = "plan_a_single_turn"
    write_summaries(run_dir, summary)
    write_task_sidecar(results_dir, run_id, tasks)
    return summary


def _dense_summary(rows: list[dict]) -> dict:
    """Training reward, reported next to the strict score. The jsonl itself stays strict."""
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "mean_reward": 0.0,
            "pass_rate": 0.0,
            "mean_correctness": 0.0,
            "mean_citation": 0.0,
            "weights": WEIGHTS,
            "note": "trajectory.reward is the strict score, comparable to the 0.0185 baseline",
        }
    return {
        "n": n,
        "mean_reward": round(sum(row["total"] for row in rows) / n, 4),
        "pass_rate": round(
            sum(1 for row in rows if float(row.get("answer_correctness", row["correctness"])) > 0) / n, 4
        ),
        "mean_correctness": round(sum(row["correctness"] for row in rows) / n, 4),
        "mean_citation": round(sum(row["citation"] for row in rows) / n, 4),
        "weights": WEIGHTS,
        "note": "trajectory.reward is the strict score, comparable to the 0.0185 baseline",
    }


def eval_from_disk(
    *,
    generate,
    model_name: str,
    run_id: str,
    split: str = "test",
    results_dir: Path | None = None,
    snapshot_version: str = "2026-10-15",
    **kwargs,
) -> dict:
    root = Path(__file__).resolve().parents[2]
    results_dir = results_dir or (root / "results")
    tasks = load_split(split)
    snapshot = DiskSnapshot(snapshot_version)
    return eval_split(
        generate=generate,
        snapshot=snapshot,
        tasks=tasks,
        model_name=model_name,
        run_id=run_id,
        results_dir=results_dir,
        split=split,
        **kwargs,
    )
