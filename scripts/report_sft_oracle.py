"""Score v4 / budget-8 SFT oracle rows on the CPU. Does not start Modal.

    uv run python scripts/report_sft_oracle.py

The default pool is the current train split. ``--include-hard-train`` is a second
report and is not mixed into the default keep counts. Holdout and hard eval are
refused. Writes ``results/sft-v4-oracle.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.rollout import QwenToolCodec, RolloutLimits, system_prompt
from fdagym.train.rollout_dry import CharTokenizer
from fdagym.train.sft_config import SFTConfig
from fdagym.train.sft_select import sft_sources
from fdagym.train.sft_tools import MIN_SFT_REWARD, build_sft_rows, split_sft_by_drug


def audit(snapshot, tasks: list[dict], config: SFTConfig, *, hard_tasks: list[dict] | None = None) -> dict:
    """Keep rows at ``min_reward``. Char length is reported beside the token cutoff.

    ``CharTokenizer`` is one character per token, so a row longer than 8192 characters
    might still fit in 8192 BPE tokens. Rows at or under 8192 characters cannot miss
    that cutoff. The reward filter is the one the v3 warm-up used (``>= 0.97``).
    """
    selected, mix = sft_sources(tasks, config, hard_tasks)
    pool = [*tasks, *(hard_tasks or [])] if config.include_hard_train else tasks
    train_tasks, val_tasks = split_sft_by_drug(
        selected, fraction=config.val_drug_fraction, seed=config.seed
    )
    # CharTokenizer is one character per token. The 8192 training context would
    # close a long label page before submit, which is a tokenizer artifact.
    # Score the oracle with room to finish, then report character length against
    # the 8192 cutoff separately. Rows at or under 8192 characters fit in 8192 BPE tokens.
    limits = RolloutLimits(
        temperature=0.0,
        tool_budget=config.tool_budget,
        prompt_version=config.prompt_version,
        max_context_tokens=200_000,
        max_new_tokens=4096,
    )
    codec = QwenToolCodec(prompt_version=limits.prompt_version, tool_budget=limits.tool_budget)
    tokenizer = CharTokenizer()
    # Do not drop on character length. The reward threshold is the keep rule under test.
    train_built = build_sft_rows(
        snapshot,
        pool,
        [task["id"] for task in train_tasks],
        tokenizer,
        codec,
        limits,
        max_length=10**12,
        min_reward=config.min_reward,
        run_id="sft-v4-audit",
    )
    val_built = build_sft_rows(
        snapshot,
        pool,
        [task["id"] for task in val_tasks],
        tokenizer,
        codec,
        limits,
        max_length=10**12,
        min_reward=config.min_reward,
        run_id="sft-v4-audit-val",
    )
    return {
        "prompt_version": config.prompt_version,
        "tool_budget": config.tool_budget,
        "min_reward": config.min_reward,
        "include_hard_train": config.include_hard_train,
        "model": config.model,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length_tokens": config.max_length,
        "stage_mix": mix,
        "n_selected": len(selected),
        "n_fit": len(train_tasks),
        "n_drug_holdout": len(val_tasks),
        "fit": _side(train_built, config.max_length),
        "drug_holdout": _side(val_built, config.max_length),
        "baked_budget_line": f"- You get at most {config.tool_budget} tool calls, then you must submit_answer.",
        "baked_system_prompt": system_prompt(config.prompt_version, config.tool_budget),
        "example_row": _example(train_built, tokenizer, config),
    }


def _side(built: dict, token_limit: int) -> dict:
    by_reason: dict[str, int] = {}
    for row in built["skipped"]:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
    lengths = [len(roll.prompt_ids) + len(roll.completion_ids) for roll in built["rollouts"]]
    over_rows = [
        {"task_id": roll.task_id, "chars": length}
        for roll, length in zip(built["rollouts"], lengths, strict=True)
        if length > token_limit
    ]
    failures = [row for row in built["skipped"] if row["reason"] == "script"]
    reward_drops = [row for row in built["skipped"] if row["reason"] == "reward"]
    return {
        "n_kept": built["n_kept"],
        "n_skipped": built["n_skipped"],
        "by_reason": by_reason,
        "n_script_failures": len(failures),
        "script_failures": failures,
        "n_reward_dropped": len(reward_drops),
        "reward_dropped": reward_drops,
        "char_length_max": max(lengths) if lengths else 0,
        "char_length_over_token_limit": len(over_rows),
        "char_length_over_token_limit_ids": over_rows,
    }


def _example(built: dict, tokenizer: CharTokenizer, config: SFTConfig) -> dict | None:
    if not built["rollouts"]:
        return None
    rollout = built["rollouts"][0]
    prompt = tokenizer.decode(rollout.prompt_ids)
    baked = system_prompt(config.prompt_version, config.tool_budget)
    if baked not in prompt:
        raise RuntimeError("example prompt does not contain the system prompt for this budget")
    steps = rollout.trajectory.get("steps") or []
    return {
        "task_id": rollout.task_id,
        "prompt_version": config.prompt_version,
        "tool_budget": config.tool_budget,
        "reward_total": rollout.reward["total"],
        "tools": [step.get("tool") for step in steps],
        "prompt": prompt,
        "model_spans": [span.text for span in rollout.spans if span.role == "model"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU audit of v4 SFT oracle rows. Does not start Modal.")
    parser.add_argument("--include-hard-train", action="store_true")
    parser.add_argument("--snapshot", default="2026-10-15")
    parser.add_argument("--out", default="results/sft-v4-oracle.json")
    args = parser.parse_args(argv)
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot

    config = SFTConfig(include_hard_train=False)
    if config.min_reward != MIN_SFT_REWARD:
        raise RuntimeError("keep threshold drifted from MIN_SFT_REWARD")
    snapshot = DiskSnapshot(args.snapshot)
    tasks = load_split("train")
    report = audit(snapshot, tasks, config)
    if args.include_hard_train:
        hard_tasks = load_split("hard_train")
        report["hard_train_opt_in"] = audit(
            snapshot, tasks, SFTConfig(include_hard_train=True), hard_tasks=hard_tasks
        )
        # The opt-in block repeats the train rows. Keep the default example.
        report["hard_train_opt_in"].pop("example_row", None)
        report["hard_train_opt_in"].pop("baked_system_prompt", None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    fit = report["fit"]
    hold = report["drug_holdout"]
    print(
        f"fit kept={fit['n_kept']} skipped={fit['n_skipped']} "
        f"script_failures={fit['n_script_failures']} reward_dropped={fit['n_reward_dropped']} "
        f"chars_over_8192={fit['char_length_over_token_limit']}"
    )
    print(
        f"drug_holdout kept={hold['n_kept']} skipped={hold['n_skipped']} "
        f"script_failures={hold['n_script_failures']} reward_dropped={hold['n_reward_dropped']}"
    )
    example = report["example_row"]
    print(
        f"example {example['task_id']} budget={example['tool_budget']} "
        f"prompt={example['prompt_version']} reward={example['reward_total']} tools={example['tools']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
