"""Tool-mode GPU jobs on a machine that already has the GPU. Does not call Modal.

``WANDB_API_KEY`` and ``HF_TOKEN`` are read from the environment by the
libraries that need them. This script does not accept them as arguments and
does not print them.

    uv run python scripts/run_gpu.py probe --help
    uv run python scripts/run_gpu.py train-tools --help
    uv run python scripts/run_gpu.py eval-tools --help
    uv run python scripts/run_gpu.py distill-sft --help

The functions are ``run_probe_dispersion``, ``run_train_tools``,
``run_eval_tools``, and ``run_distill_sft`` in ``fdagym/train/jobs.py``.
Modal's wrappers call the first three. Distill SFT is the RunPod student job.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.curriculum import GRPO_POOL_TYPES_FLAG
from fdagym.train.distill import DEFAULT_RUN_ID, DEFAULT_SIZES, fixture_path
from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET, PROMPT_VERSIONS
from fdagym.train.tool_config import TOOL_TEMPERATURE


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    os.environ["FDAGYM_EXECUTOR"] = args.executor
    if os_hf_home_set():
        print(
            "HF_HOME is already set; the job will not move the Hub cache next to --runs-dir.",
            file=sys.stderr,
        )
    result = args.func(args)
    print(result)
    return 0


def os_hf_home_set() -> bool:
    import os

    return bool(os.environ.get("HF_HOME", "").strip())


def _train(args: argparse.Namespace) -> dict:
    from fdagym.train.jobs import run_train_tools

    return run_train_tools(
        runs_dir=args.runs_dir,
        on_save=None,
        resume=args.resume,
        run_id=args.run_id,
        sft_run_id=args.sft_run_id,
        init_adapter=args.init_adapter,
        tool_budget=args.tool_budget,
        prompt_version=args.prompt_version,
        hard_train_mix=args.hard_train_mix,
        max_steps=args.max_steps,
        memorization_every=args.memorization_every,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        prompts_per_step=args.prompts_per_step,
        dynamic_sampling=args.dynamic_sampling,
        difficulty_filter=args.difficulty_filter,
        max_resample_attempts=args.max_resample_attempts,
        temperature=args.temperature,
        pool_types=args.pool_types,
    )


def _eval(args: argparse.Namespace) -> dict:
    from fdagym.train.jobs import run_eval_tools

    return run_eval_tools(
        runs_dir=args.runs_dir,
        on_save=None,
        adapter=args.adapter,
        run_id=args.run_id,
        limit=args.limit,
        split=args.split,
        tool_budget=args.tool_budget,
        prompt_version=args.prompt_version,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def _probe(args: argparse.Namespace) -> dict:
    from fdagym.train.jobs import run_probe_dispersion

    return run_probe_dispersion(
        runs_dir=args.runs_dir,
        on_save=None,
        adapter=args.adapter,
        task_ids=_task_ids(args),
        task_ids_sha256=args.task_ids_sha256,
        run_id=args.run_id,
        k=args.k,
        seed=args.seed,
        hard_train_mix=args.hard_train_mix,
        tool_budget=args.tool_budget,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        pool_types=args.pool_types,
    )


def _distill(args: argparse.Namespace) -> dict:
    from fdagym.train.jobs import run_distill_sft

    paths = list(args.trajectories or [])
    if not paths:
        paths = [fixture_path()]
    return run_distill_sft(
        runs_dir=args.runs_dir,
        trajectories=paths,
        on_save=None,
        run_id=args.run_id,
        sizes=args.sizes,
        seed=args.seed,
        teachers=args.teacher,
        templates=args.template,
        max_attempts_per_question=args.max_attempts_per_question,
        init_adapter=args.init_adapter,
        model=args.model,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        tool_budget=args.tool_budget,
        prompt_version=args.prompt_version,
        eval_after=args.eval,
        eval_split=args.eval_split,
        eval_limit=args.eval_limit,
        dry_run=args.dry_run,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def _task_ids(args: argparse.Namespace) -> list[str]:
    repeated = list(args.task_id or [])
    if args.task_ids_file:
        text = Path(args.task_ids_file).read_text(encoding="utf-8")
        from_file = [line.strip() for line in text.splitlines() if line.strip()]
        if repeated:
            raise SystemExit("pass probe ids with --task-ids-file or --task-id, not both")
        repeated = from_file
    if not repeated:
        raise SystemExit("probe requires --task-ids-file or repeated --task-id")
    return repeated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tool-mode GPU job on this machine. Does not call Modal or rent a pod."
    )
    parser.add_argument(
        "--executor",
        choices=("runpod", "local"),
        default="runpod",
        help="record where this job runs (default: runpod)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train-tools", help="tool GRPO, same arguments as Modal train_tools")
    _runs_dir(train)
    train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--run-id", default="grpo-qwen3b-tools")
    train.add_argument("--sft-run-id", default="sft-qwen3b-tools")
    train.add_argument(
        "--init-adapter",
        default="",
        help="local LoRA directory or Hugging Face repo id (private repos need HF_TOKEN)",
    )
    _budget_and_prompt(train)
    _sampling(train)
    train.add_argument("--hard-train-mix", type=float, default=0.0)
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--memorization-every", type=int, default=10)
    train.add_argument("--save-steps", type=int, default=50)
    _wandb(train)
    train.add_argument("--prompts-per-step", type=int, default=None)
    _optional_bool(train, "dynamic-sampling")
    _optional_bool(train, "difficulty-filter")
    train.add_argument("--max-resample-attempts", type=int, default=None)
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("eval-tools", help="greedy tool eval, same arguments as Modal eval_tools")
    _runs_dir(evaluate)
    evaluate.add_argument("--adapter", default=None)
    evaluate.add_argument("--run-id", default="grpo-qwen3b-tools-eval")
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--split", default="test")
    _budget_and_prompt(evaluate)
    _wandb(evaluate)
    evaluate.set_defaults(func=_eval)

    probe = sub.add_parser("probe", help="reward-dispersion probe, same arguments as Modal probe_dispersion")
    _runs_dir(probe)
    probe.add_argument("--adapter", required=True)
    probe.add_argument("--task-id", action="append", default=None, help="one id; repeat for a list. Or use --task-ids-file")
    probe.add_argument("--task-ids-file", default=None, help="one task id per line, in draw order")
    probe.add_argument("--task-ids-sha256", required=True)
    probe.add_argument("--run-id", required=True)
    probe.add_argument("--k", type=int, default=8)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--hard-train-mix", type=float, default=0.6)
    _budget_and_prompt(probe)
    _sampling(probe)
    probe.set_defaults(func=_probe)

    distill = sub.add_parser(
        "distill-sft",
        help="teacher-trajectory LoRA SFT over nested data sizes (learning curve)",
    )
    _runs_dir(distill)
    distill.add_argument(
        "--trajectories",
        action="append",
        type=Path,
        default=None,
        help="teacher JSONL (repeat). Default: tests/train/fixtures/teacher_trajectories.jsonl",
    )
    distill.add_argument("--sizes", default=",".join(DEFAULT_SIZES), help="nested sizes, e.g. 100,250,500,1000,all")
    distill.add_argument("--seed", type=int, default=0)
    distill.add_argument("--teacher", action="append", default=None, help="keep this teacher_model; repeat to allow several")
    distill.add_argument("--template", action="append", default=None, help="keep this template; repeat to allow several")
    distill.add_argument("--max-attempts-per-question", type=int, default=None)
    distill.add_argument("--run-id", default=DEFAULT_RUN_ID)
    distill.add_argument("--init-adapter", default="", help="existing LoRA to continue; default is a fresh adapter on the base 3B")
    distill.add_argument("--model", default="", help="base model id (default Qwen/Qwen2.5-3B-Instruct)")
    distill.add_argument("--learning-rate", type=float, default=None)
    distill.add_argument("--num-train-epochs", type=float, default=None)
    distill.add_argument("--per-device-train-batch-size", type=int, default=None)
    distill.add_argument("--gradient-accumulation-steps", type=int, default=None)
    distill.add_argument("--max-length", type=int, default=None)
    distill.add_argument("--warmup-ratio", type=float, default=None)
    distill.add_argument("--save-steps", type=int, default=None)
    distill.add_argument("--max-steps", type=int, default=0, help="0 uses num_train_epochs (warm-up default). Set for a short GPU smoke.")
    _budget_and_prompt(distill)
    distill.add_argument("--eval", action=argparse.BooleanOptionalAction, default=False, help="run eval-tools on each adapter")
    distill.add_argument("--eval-split", default="hard")
    distill.add_argument("--eval-limit", type=int, default=None)
    distill.add_argument("--dry-run", action="store_true", help="CPU: filter, nest, mask-check fixtures. Does not load Qwen.")
    _wandb(distill)
    distill.set_defaults(func=_distill)
    return parser



def _runs_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="directory for run outputs (default: ./runs on this disk). Hub cache is <parent>/hf-cache unless HF_HOME is set",
    )


def _budget_and_prompt(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tool-budget", type=int, default=DEFAULT_TOOL_BUDGET)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=DEFAULT_PROMPT_VERSION)


def _sampling(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        default=TOOL_TEMPERATURE,
        help="rollout sampling temperature for probe and GRPO (default: 0.65)",
    )
    parser.add_argument(
        "--pool-types",
        default=GRPO_POOL_TYPES_FLAG,
        help="comma-separated current-train types in the GRPO pool (composite is excluded by default)",
    )


def _wandb(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")


def _optional_bool(parser: argparse.ArgumentParser, flag: str) -> None:
    """Tri-state flag. Omitted stays None, which is Modal's 'use the mix default'."""
    dest = flag.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{flag}", dest=dest, action="store_const", const=True, default=None)
    group.add_argument(f"--no-{flag}", dest=dest, action="store_const", const=False)


if __name__ == "__main__":
    raise SystemExit(main())
