"""Spawn kwargs for the hard-run commands in docs/RUNBOOK.md.

Importing this module does not call Modal. ``scripts/spawn_modal.py`` is what spawns.
"""

from __future__ import annotations

APP = "fdagym-grpo-3b"
SFT_RUN_ID = "sft-qwen3b-tools-v4"
ADAPTER = "/vol/runs/sft-qwen3b-tools-v4/adapter"

# New hard RL defaults. Unflagged train-tools stays at hard_train_mix 0.
HARD_RL_MIX = 0.6
HARD_RL_BUDGET = 8
HARD_RL_PROMPT = "v4"
DRY_RUN_STEPS = 3


def spawn_plan(which: str) -> dict:
    """Return ``{"app", "function", "kwargs"}`` for one deployed function.

    ``sft`` is the v4 / budget-8 warm-up from the base 3B, current train split only.
    ``eval-hard`` scores that adapter on ``data/tasks/hard.jsonl``.
    ``grpo-dry`` is a few GRPO steps at the 60/40 mix. Reward weights are not an argument.
    """
    if which == "sft":
        function = "sft"
        kwargs = {
            "run_id": SFT_RUN_ID,
            "tool_budget": HARD_RL_BUDGET,
            "prompt_version": HARD_RL_PROMPT,
            "include_hard_train": False,
        }
    elif which == "eval-hard":
        function = "eval_tools"
        kwargs = {
            "adapter": ADAPTER,
            "run_id": f"{SFT_RUN_ID}-hard",
            "split": "hard",
            "tool_budget": HARD_RL_BUDGET,
            "prompt_version": HARD_RL_PROMPT,
        }
    elif which == "grpo-dry":
        function = "train_tools"
        kwargs = {
            "resume": False,
            "run_id": "grpo-qwen3b-tools-hard-dry",
            "sft_run_id": SFT_RUN_ID,
            "init_adapter": ADAPTER,
            "tool_budget": HARD_RL_BUDGET,
            "prompt_version": HARD_RL_PROMPT,
            "hard_train_mix": HARD_RL_MIX,
            "max_steps": DRY_RUN_STEPS,
            "memorization_every": 10,
        }
    else:
        raise ValueError(f"unknown spawn {which!r}; use sft, eval-hard, or grpo-dry")
    return {"app": APP, "function": function, "kwargs": kwargs}
