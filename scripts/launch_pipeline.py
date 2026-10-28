"""Spawn a named server-side training pipeline on a deployed Modal app.

The returned call id can be used to track the job. A local disconnect does not
cancel the remote call. Deploy ``modal_train.py`` before using this command.

    python scripts/launch_pipeline.py tools
    python scripts/launch_pipeline.py plan-a
"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import modal

    function = _function(modal, args.which)
    call = function.spawn(**_kwargs(args))
    print("spawned", call.object_id)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=("tools", "plan-a"))
    parser.add_argument("--init-adapter", default="")
    parser.add_argument("--force", action="store_true", help="restart even if status.json says done or failed")
    parser.add_argument("--hard-train-mix", type=float, default=None)
    parser.add_argument("--tool-budget", type=int, default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--prompts-per-step", type=int, default=None)
    parser.add_argument("--max-resample-attempts", type=int, default=None)
    parser.add_argument("--dynamic-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--difficulty-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-hard-train", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    return parser


def _function(modal, pipeline: str):
    name = "pipeline_tools" if pipeline == "tools" else "pipeline_plan_a"
    return modal.Function.from_name("fdagym-grpo-3b", name)


def _kwargs(args: argparse.Namespace) -> dict:
    if args.which == "plan-a":
        return _present({"force": args.force, "wandb_project": args.wandb_project, "wandb_entity": args.wandb_entity})

    values = {
        "init_adapter": args.init_adapter,
        "force": args.force,
        "hard_train_mix": args.hard_train_mix,
        "tool_budget": args.tool_budget,
        "prompt_version": args.prompt_version,
        "max_steps": args.max_steps,
        "prompts_per_step": args.prompts_per_step,
        "max_resample_attempts": args.max_resample_attempts,
        "dynamic_sampling": args.dynamic_sampling,
        "difficulty_filter": args.difficulty_filter,
        "include_hard_train": True if args.include_hard_train else None,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }
    return _present(values)


def _present(values: dict) -> dict:
    """Drop omitted flags so the deployed function keeps its own defaults."""
    return {key: value for key, value in values.items() if value is not None and value != ""}


if __name__ == "__main__":
    raise SystemExit(main())
