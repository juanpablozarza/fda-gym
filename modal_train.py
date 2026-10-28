"""3B training on Modal. Does nothing until JP runs `modal run`.

Plan A (L40S, prefetched docs):

    modal run modal_train.py --action sft-plan-a
    modal run modal_train.py --action eval-adapter --adapter /vol/runs/sft-plan-a-qwen3b/adapter
    modal run modal_train.py --action train --init-adapter /vol/runs/sft-plan-a-qwen3b/adapter --max-steps 150
    modal run modal_train.py --action eval-base
    modal run modal_train.py --action train
    modal run modal_train.py --action eval-adapter
    modal run modal_train.py --action train --gpu A100-80GB   # if L40S OOMs

Tool-call GRPO (always A100-80GB, even if ``--gpu`` says otherwise). Order is SFT, then GRPO, then eval:

    modal run modal_train.py --action sft
    modal run modal_train.py --action train-tools
    modal run modal_train.py --action eval-tools-base
    modal run modal_train.py --action eval-tools

The same three job bodies (``train_tools``, ``eval_tools``, ``probe_dispersion``) run on a
plain GPU box through ``scripts/run_gpu.py``. Both paths call ``fdagym.train.jobs``.

Server side pipelines (preferred; a laptop disconnect cannot kill them):

    modal deploy modal_train.py
    python scripts/launch_pipeline.py tools [--init-adapter /vol/runs/grpo-qwen3b-sft150/adapter]
    python scripts/launch_pipeline.py plan-a

See docs/RUNBOOK.md. Checkpoints live on Volume ``fdagym-checkpoints`` at /vol/runs/<run_id>/checkpoint-*.
The function commits the volume every save (~50 steps) so a crash is resumable.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
VOL_MOUNT = "/vol"
RUN_ID_3B = "grpo-qwen3b"
BASELINE_ID = "baseline-qwen3b"
EVAL_ID = "grpo-qwen3b-eval"
SFT_ID = "sft-plan-a-qwen3b"
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
SFT_RUN_ID = "sft-qwen3b-tools"
TOOL_RUN_ID = "grpo-qwen3b-tools"
TOOL_BASELINE_ID = "baseline-qwen3b-tools"
TOOL_EVAL_ID = "grpo-qwen3b-tools-eval"
TOOL_GPU = "A100-80GB"

# Colocate stack from the TRL vLLM blog: trl 0.21 + vllm 0.8.5 + torch 2.6.
# The pins live in the [gpu] extra. This list is that extra plus the runtime deps.


def _modal_pip() -> list[str]:
    """Pinned image packages, loaded without importing fdagym.

    The container imports this module before ``_bind_paths`` puts ``/root/bio-rl``
    on ``sys.path``, so ``import fdagym`` fails there. The image install uses the
    list computed on the laptop, where ``pyproject.toml`` is readable either way.
    """
    import importlib.util

    path = ROOT / "fdagym" / "train" / "gpu_stack.py"
    spec = importlib.util.spec_from_file_location("_fdagym_gpu_stack_pins", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GPU pins from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.modal_pip()


PIP = _modal_pip()

IGNORE = [
    ".git",
    ".venv",
    ".env",
    "*.pyc",
    "web/node_modules",
    "web/.next",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "results",
    "learn/answers",
]


def _deploy_git_sha() -> str:
    """SHA baked into the image. The container has no .git (it is in IGNORE)."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


_DEPLOY_GIT_SHA = _deploy_git_sha()

# add_local_dir must be the last image step. Modal rejects a later .env() unless copy=True.
image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP)
if _DEPLOY_GIT_SHA:
    image = image.env({"FDAGYM_GIT_SHA": _DEPLOY_GIT_SHA})
image = image.add_local_dir(str(ROOT), remote_path="/root/bio-rl", ignore=IGNORE)

volume = modal.Volume.from_name("fdagym-checkpoints", create_if_missing=True)
# Key lives only in this secret. Never pass WANDB_API_KEY as a function argument.
WANDB_SECRET = modal.Secret.from_name("wandb")
app = modal.App("fdagym-grpo-3b", image=image)

GPU_DEFAULT = "L40S"
TIMEOUT_S = 4 * 60 * 60


def _commit(_state=None) -> None:
    volume.commit()


def _bind_paths() -> None:
    import os
    import sys

    os.environ["FDAGYM_EXECUTOR"] = "modal"
    os.environ.setdefault("HF_HOME", f"{VOL_MOUNT}/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", f"{VOL_MOUNT}/hf-cache")
    sys.path.insert(0, "/root/bio-rl")


_TRAIN_KW = {
    "timeout": TIMEOUT_S,
    "memory": 65536,
    "volumes": {VOL_MOUNT: volume},
    "retries": 0,
    "secrets": [WANDB_SECRET],
}


@app.function(gpu=GPU_DEFAULT, **_TRAIN_KW)
def train(
    resume: bool = True,
    run_id: str = RUN_ID_3B,
    init_adapter: str = "",
    max_steps: int = 0,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    return _train_impl(resume, run_id, init_adapter, max_steps, wandb_project, wandb_entity)


@app.function(gpu="A100-80GB", **_TRAIN_KW)
def train_a100(
    resume: bool = True,
    run_id: str = RUN_ID_3B,
    init_adapter: str = "",
    max_steps: int = 0,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    """Same as train, pinned to A100-80GB for the server side pipeline."""
    return _train_impl(resume, run_id, init_adapter, max_steps, wandb_project, wandb_entity)


def _train_impl(
    resume: bool,
    run_id: str,
    init_adapter: str,
    max_steps: int,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _bind_paths()
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.train.config import QWEN_3B, TrainConfig
    from fdagym.train.launch import run_grpo

    config = TrainConfig(
        model=QWEN_3B,
        use_vllm=True,
        gradient_checkpointing=True,
        init_adapter=init_adapter or None,
        max_steps=max_steps or TrainConfig.max_steps,
    )
    snapshot = DiskSnapshot("2026-10-15")
    tasks = load_split("train")
    results_dir = Path(VOL_MOUNT) / "runs"
    return run_grpo(
        snapshot,
        tasks,
        config,
        run_id=run_id,
        results_dir=results_dir,
        resume=resume,
        on_save=_commit,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )


@app.function(
    gpu=GPU_DEFAULT,
    timeout=2 * 60 * 60,
    scaledown_window=60,
    memory=32768,
    volumes={VOL_MOUNT: volume},
    retries=0,
)
def eval_plan_a(adapter: str | None = None, run_id: str = BASELINE_ID, limit: int | None = None) -> dict:
    _bind_paths()
    from fdagym.train.config import QWEN_3B
    from fdagym.train.eval_local import eval_from_disk
    from fdagym.train.local_lm import load_generator

    adapter_path = None
    kind = "baseline"
    model_name = QWEN_3B
    if adapter:
        adapter_path = adapter if adapter.startswith("/") else f"{VOL_MOUNT}/runs/{RUN_ID_3B}/{adapter}"
        kind = "trained"
        model_name = f"{QWEN_3B}+lora"
    generate = load_generator(QWEN_3B, adapter=adapter_path, max_new_tokens=256, temperature=0.0, seed=0)
    summary = eval_from_disk(
        generate=generate,
        model_name=model_name,
        run_id=run_id,
        split="test",
        results_dir=Path(VOL_MOUNT) / "runs",
        kind=kind,
        limit=limit,
    )
    _commit()
    return {
        "run_id": summary["run_id"],
        "models": summary.get("models"),
        "leaderboard": summary.get("leaderboard"),
        "dense_reward": summary.get("dense_reward"),
    }


@app.function(
    gpu=GPU_DEFAULT,
    timeout=2 * 60 * 60,
    memory=65536,
    volumes={VOL_MOUNT: volume},
    secrets=[WANDB_SECRET],
    retries=0,
)
def sft_plan_a(wandb_project: str = "", wandb_entity: str = "") -> dict:
    """One LoRA epoch. Adapter lands at /vol/runs/sft-plan-a-qwen3b/adapter."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _bind_paths()
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.train.config import QWEN_3B, TrainConfig
    from fdagym.train.sft import run_sft

    summary = run_sft(
        DiskSnapshot("2026-10-15"),
        load_split("train"),
        TrainConfig(model=QWEN_3B, gradient_checkpointing=True),
        output_dir=Path(VOL_MOUNT) / "runs" / SFT_ID,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    _commit()
    return summary


def _eval_run_id(adapter: str) -> str:
    if adapter in {"", "adapter"}:
        return EVAL_ID
    parent = Path(adapter.rstrip("/")).parent.name
    return f"{parent}-eval"


@app.function(
    gpu=TOOL_GPU,
    timeout=2 * 60 * 60,
    memory=65536,
    volumes={VOL_MOUNT: volume},
    secrets=[WANDB_SECRET],
    retries=0,
)
def sft(
    run_id: str = SFT_RUN_ID,
    tool_budget: int = 8,
    prompt_version: str = "v4",
    include_hard_train: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    init_adapter: str | None = None,
    task_ids: list[str] | None = None,
    task_ids_sha256: str | None = None,
    num_train_epochs: float = 1.0,
    learning_rate: float = 1e-4,
    save_steps: int = 50,
    val_drug_fraction: float = 0.05,
    warmup_ratio: float = 0.03,
) -> dict:
    """Warm up Qwen2.5-3B-Instruct. Omitted hyperparameters match ``SFTConfig`` (the v4 warm-up).

    tool_budget 8 and prompt v4 are the tool-mode defaults. Pass 6 and v3 to reproduce the old warm-up.
    ``include_hard_train`` defaults off, so the pool is the current train split. It never loads the
    known-drug holdout or the hard eval.

    ``init_adapter`` continues an existing LoRA with its weights trainable, instead of a fresh one.
    ``task_ids`` trains on exactly those ids from train and hard_train. ``task_ids_sha256`` is the
    digest of the sorted ids joined by newlines. ``val_drug_fraction`` 0 keeps every row in the fit.
    """
    _bind_paths()
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.train.sft_config import SFTConfig
    from fdagym.train.sft_launch import run_sft

    config = SFTConfig(
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        include_hard_train=include_hard_train,
        task_ids=task_ids,
        task_ids_sha256=task_ids_sha256,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=save_steps,
        val_drug_fraction=val_drug_fraction,
        warmup_ratio=warmup_ratio,
    )
    hard_tasks = load_split("hard_train") if config.include_hard_train or config.task_ids is not None else None
    summary = run_sft(
        DiskSnapshot("2026-10-15"),
        load_split("train"),
        config,
        run_id=run_id,
        results_dir=Path(VOL_MOUNT) / "runs",
        on_save=_commit,
        hard_tasks=hard_tasks,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        init_adapter=init_adapter,
    )
    return {
        "run_id": summary["run_id"],
        "adapter": summary["adapter"],
        "n_kept": summary["n_kept"],
        "n_skipped": summary["n_skipped"],
        "n_val": summary["n_val"],
        "prompt_version": summary["prompt_version"],
        "tool_budget": summary["tool_budget"],
        "include_hard_train": summary["include_hard_train"],
        "model": summary["model"],
        "init_adapter": summary["init_adapter"],
        "init_adapter_sha256": summary["init_adapter_sha256"],
        "task_ids_sha256": summary["task_ids_sha256"],
        "epochs": summary["epochs"],
    }


@app.function(
    gpu=TOOL_GPU,
    timeout=6 * 60 * 60,
    memory=65536,
    volumes={VOL_MOUNT: volume},
    secrets=[WANDB_SECRET],
    retries=0,
)
def train_tools(
    resume: bool = True,
    run_id: str = TOOL_RUN_ID,
    sft_run_id: str = SFT_RUN_ID,
    init_adapter: str = "",
    tool_budget: int = 8,
    prompt_version: str = "v4",
    hard_train_mix: float = 0.0,
    max_steps: int = 0,
    memorization_every: int = 10,
    save_steps: int = 50,
    wandb_project: str = "",
    wandb_entity: str = "",
    prompts_per_step: int | None = None,
    dynamic_sampling: bool | None = None,
    difficulty_filter: bool | None = None,
    max_resample_attempts: int | None = None,
    temperature: float = 0.65,
    pool_types: str = "boxed_warning,faers_top_reaction,approval,faers_count",
) -> dict:
    """init_adapter overrides the tool SFT adapter, e.g. /vol/runs/grpo-qwen3b-sft150/adapter
    or a Hugging Face repo id such as your-org/fdagym-adapter.

    tool_budget 8 and prompt v4 are the new defaults. Pass 6 and v3 to reproduce a v3 GRPO run.

    ``hard_train_mix`` 0 keeps the current type mix (existing train-tools commands).
    A new hard RL run passes 0.6: 60% hard_train, 40% current train, holdout excluded.
    That path logs eval-set sizes and a hash of the holdout and hard-eval ids, and
    refuses to start if either id is in the training pool. The memorization callback
    then runs every ``memorization_every`` steps (default 10). Reward weights are unchanged.
    ``max_steps`` 0 keeps the 150-step tool config. ``save_steps`` 50 is the unflagged
    checkpoint interval.

    A hard mix (``hard_train_mix`` above 0) defaults to 4 questions × 8 generations
    per optimizer step, dynamic sampling of tied groups, and the current-prompt
    difficulty filter. Pass ``prompts_per_step=1``, ``dynamic_sampling=False``, or
    ``difficulty_filter=False`` to turn one of those off. ``None`` keeps the default.
    An unflagged run (mix 0 and every signal argument ``None``) stays at 1 × 8.
    """
    _bind_paths()
    from fdagym.train.jobs import run_train_tools

    return run_train_tools(
        runs_dir=Path(VOL_MOUNT) / "runs",
        on_save=_commit,
        resume=resume,
        run_id=run_id,
        sft_run_id=sft_run_id,
        init_adapter=init_adapter,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        hard_train_mix=hard_train_mix,
        max_steps=max_steps,
        memorization_every=memorization_every,
        save_steps=save_steps,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        prompts_per_step=prompts_per_step,
        dynamic_sampling=dynamic_sampling,
        difficulty_filter=difficulty_filter,
        max_resample_attempts=max_resample_attempts,
        temperature=temperature,
        pool_types=pool_types,
    )


@app.function(
    gpu=TOOL_GPU,
    timeout=3 * 60 * 60,
    scaledown_window=60,
    memory=65536,
    volumes={VOL_MOUNT: volume},
    secrets=[WANDB_SECRET],
    retries=0,
)
def eval_tools(
    adapter: str | None = None,
    run_id: str = TOOL_EVAL_ID,
    limit: int | None = None,
    split: str = "test",
    hf_token: str = "",
    tool_budget: int = 8,
    prompt_version: str = "v4",
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    import os

    _bind_paths()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    from fdagym.train.jobs import run_eval_tools

    return run_eval_tools(
        runs_dir=Path(VOL_MOUNT) / "runs",
        on_save=_commit,
        adapter=adapter,
        run_id=run_id,
        limit=limit,
        split=split,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )


@app.function(
    gpu=TOOL_GPU,
    timeout=10 * 60,
    memory=32768,
    volumes={VOL_MOUNT: volume},
    retries=0,
)
def probe_dispersion(
    adapter: str,
    task_ids: list[str],
    task_ids_sha256: str,
    run_id: str,
    k: int = 8,
    seed: int = 0,
    hard_train_mix: float = 0.6,
    tool_budget: int = 8,
    prompt_version: str = "v4",
    temperature: float = 0.65,
    pool_types: str = "boxed_warning,faers_top_reaction,approval,faers_count",
) -> dict:
    """Sample k completions per training question and record the training reward.

    Deploy, then spawn. There is no ``modal run`` action: a laptop disconnect
    must not be what starts this, and ``main`` does not dispatch it.

        modal.Function.from_name("fdagym-grpo-3b", "probe_dispersion").spawn(
            adapter="/vol/runs/<sft-run>/adapter",
            task_ids=ids,
            task_ids_sha256=sha256,
            run_id="probe-dispersion-...",
            k=8,
            seed=0,
            hard_train_mix=0.6,
            tool_budget=8,
            prompt_version="v4",
            temperature=0.65,
            pool_types="boxed_warning,faers_top_reaction,approval,faers_count",
        )

    ``ids`` is the draw order. ``sha256`` is ``sha256("\\n".join(ids))`` with no
    trailing newline. ``k`` must be 8. The pool is the one ``train_tools`` builds
    at ``hard_train_mix=0.6``. An existing ``/vol/runs/<run_id>/`` is refused.
    No W&B secret and no HF token argument. Results land at
    ``/vol/runs/<run_id>/rollouts.jsonl`` and ``probe.json``.
    The same body is ``run_probe_dispersion``. ``scripts/run_gpu.py probe`` calls it on a local GPU.
    """
    _bind_paths()
    from fdagym.train.jobs import run_probe_dispersion

    return run_probe_dispersion(
        runs_dir=Path(VOL_MOUNT) / "runs",
        on_save=_commit,
        adapter=adapter,
        task_ids=task_ids,
        task_ids_sha256=task_ids_sha256,
        run_id=run_id,
        k=k,
        seed=seed,
        hard_train_mix=hard_train_mix,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        temperature=temperature,
        pool_types=pool_types,
    )


GATE_MARGIN = 0.03
PIPE_TIMEOUT_S = 16 * 60 * 60


@app.function(timeout=PIPE_TIMEOUT_S, volumes={VOL_MOUNT: volume}, retries=0)
def pipeline_plan_a(
    baseline: float = 0.0185,
    max_steps: int = 150,
    force: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    """Server side Plan A: SFT -> eval -> gate -> GRPO (A100) -> eval. Idempotent; launch with
    scripts/launch_pipeline.py after `modal deploy modal_train.py`, never plain `modal run`."""
    _bind_paths()
    from fdagym.train.pipeline import Pipeline, Stage, strict_mean

    runs = Path(VOL_MOUNT) / "runs"
    sft_ad, grpo_id = runs / SFT_ID / "adapter", f"grpo-qwen3b-sft{max_steps}"
    grpo_ad = runs / grpo_id / "adapter"
    gate = baseline + GATE_MARGIN
    stages = [
        Stage(
            "sft",
            lambda s: sft_plan_a.remote(wandb_project=wandb_project, wandb_entity=wandb_entity),
            output=sft_ad,
        ),
        Stage(
            "eval-sft",
            lambda s: eval_plan_a.remote(adapter=str(sft_ad), run_id=_eval_run_id(str(sft_ad))),
            gate=lambda s: strict_mean(s["results"].get("eval-sft")) >= gate,
        ),
        Stage(
            "grpo",
            lambda s: train_a100.remote(
                resume=True,
                run_id=grpo_id,
                init_adapter=str(sft_ad),
                max_steps=max_steps,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            ),
            output=grpo_ad,
        ),
        Stage("eval-grpo", lambda s: eval_plan_a.remote(adapter=str(grpo_ad), run_id=_eval_run_id(str(grpo_ad)))),
    ]
    return Pipeline(runs / "pipeline-plan-a", stages, commit=volume.commit).run(force=force)


@app.function(timeout=PIPE_TIMEOUT_S, volumes={VOL_MOUNT: volume}, retries=0)
def pipeline_tools(
    init_adapter: str = "",
    force: bool = False,
    tool_budget: int = 8,
    prompt_version: str = "v4",
    hard_train_mix: float = 0.0,
    max_steps: int = 0,
    include_hard_train: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    prompts_per_step: int | None = None,
    dynamic_sampling: bool | None = None,
    difficulty_filter: bool | None = None,
    max_resample_attempts: int | None = None,
) -> dict:
    """Server side tool run: eval-tools-base -> sft (tools) -> gate -> train-tools -> eval-tools.

    init_adapter: start tool GRPO from an existing adapter (e.g. tonight's RL model, re-uploaded with
    scripts/upload_adapter.sh to /vol/runs/grpo-qwen3b-sft150/adapter) instead of the tool SFT adapter.
    The gate compares the starting adapter's tool eval against eval-tools-base + GATE_MARGIN.

    Each stage sets ``output``. A fresh volume that already has the base summary, the SFT adapter,
    and the gate summary skips those three and starts GRPO. See docs/RUNBOOK.md.

    Defaults match an unflagged tool run: prompt v4, budget 8, no hard mix, 150 GRPO steps,
    SFT on the current train split. A new hard RL run passes ``hard_train_mix=0.6``.
    """
    _bind_paths()
    from fdagym.train.pipeline import Pipeline, Stage, strict_mean, tool_artifact_paths

    runs = Path(VOL_MOUNT) / "runs"
    paths = tool_artifact_paths(
        runs,
        baseline_id=TOOL_BASELINE_ID,
        sft_id=SFT_RUN_ID,
        grpo_id=TOOL_RUN_ID,
        grpo_eval_id=TOOL_EVAL_ID,
        init_adapter=init_adapter,
    )
    start_ad = paths["sft_adapter"]
    tool_ad = paths["grpo_adapter"]

    def gate(s):
        base = strict_mean(s["results"].get("eval-tools-base"))
        return strict_mean(s["results"].get("eval-tools-start")) >= base + GATE_MARGIN

    stages = [
        Stage(
            "eval-tools-base",
            lambda s: eval_tools.remote(
                adapter=None,
                run_id=TOOL_BASELINE_ID,
                tool_budget=tool_budget,
                prompt_version=prompt_version,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            ),
            output=paths["eval_base"],
        ),
    ]
    if not init_adapter:
        stages.append(
            Stage(
                "sft-tools",
                lambda s: sft.remote(
                    run_id=SFT_RUN_ID,
                    tool_budget=tool_budget,
                    prompt_version=prompt_version,
                    include_hard_train=include_hard_train,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                ),
                output=start_ad,
            )
        )
    stages += [
        Stage(
            "eval-tools-start",
            lambda s: eval_tools.remote(
                adapter=str(start_ad),
                run_id=f"{start_ad.parent.name}-tools-eval",
                tool_budget=tool_budget,
                prompt_version=prompt_version,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            ),
            output=paths["eval_start"],
            gate=gate,
        ),
        Stage(
            "train-tools",
            lambda s: train_tools.remote(
                resume=True,
                run_id=TOOL_RUN_ID,
                init_adapter=str(start_ad),
                tool_budget=tool_budget,
                prompt_version=prompt_version,
                hard_train_mix=hard_train_mix,
                max_steps=max_steps,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                prompts_per_step=prompts_per_step,
                dynamic_sampling=dynamic_sampling,
                difficulty_filter=difficulty_filter,
                max_resample_attempts=max_resample_attempts,
            ),
            output=tool_ad,
        ),
        Stage(
            "eval-tools",
            lambda s: eval_tools.remote(
                adapter=str(tool_ad),
                run_id=TOOL_EVAL_ID,
                tool_budget=tool_budget,
                prompt_version=prompt_version,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            ),
            output=paths["eval_grpo"],
        ),
    ]
    return Pipeline(runs / "pipeline-tools", stages, commit=volume.commit).run(force=force)


@app.local_entrypoint()
def main(
    action: str = "train",
    gpu: str = GPU_DEFAULT,
    resume: bool = True,
    limit: int | None = None,
    init_adapter: str = "",
    adapter: str = "",
    max_steps: int = 0,
    run_id: str = "",
    split: str = "test",
    tool_budget: int = 8,
    prompt_version: str = "v4",
    hard_train_mix: float = 0.0,
    include_hard_train: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    prompts_per_step: int | None = None,
    dynamic_sampling: bool | None = None,
    difficulty_filter: bool | None = None,
    max_resample_attempts: int | None = None,
):
    """Plan A: sft-plan-a | eval-base | train | eval-adapter.

    Tools: sft | train-tools | eval-tools-base | eval-tools.
    Does nothing until `modal run`. Tool actions always request an A100-80GB.
    """
    from fdagym.train.snapshot_fs import require_snapshot

    require_snapshot(ROOT)
    plan_a = {"eval-base", "train", "eval-adapter", "sft-plan-a"}
    tools = {"sft", "train-tools", "eval-tools-base", "eval-tools"}
    if action not in plan_a and action not in tools:
        raise SystemExit(
            "unknown action "
            f"{action!r}; use sft-plan-a, eval-base, train, eval-adapter, "
            "sft, train-tools, eval-tools-base, or eval-tools"
        )
    if action in tools:
        gpu = TOOL_GPU
    if action == "sft-plan-a":
        result = sft_plan_a.with_options(gpu=gpu).remote(wandb_project=wandb_project, wandb_entity=wandb_entity)
    elif action == "eval-base":
        result = eval_plan_a.with_options(gpu=gpu).remote(adapter=None, run_id=BASELINE_ID, limit=limit)
    elif action == "eval-adapter":
        chosen = adapter or "adapter"
        result = eval_plan_a.with_options(gpu=gpu).remote(adapter=chosen, run_id=_eval_run_id(chosen), limit=limit)
    elif action == "sft":
        result = sft.with_options(gpu=gpu).remote(
            run_id=SFT_RUN_ID,
            tool_budget=tool_budget,
            prompt_version=prompt_version,
            include_hard_train=include_hard_train,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
    elif action == "train-tools":
        result = train_tools.with_options(gpu=gpu).remote(
            resume=resume,
            run_id=TOOL_RUN_ID,
            tool_budget=tool_budget,
            prompt_version=prompt_version,
            hard_train_mix=hard_train_mix,
            max_steps=max_steps,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            prompts_per_step=prompts_per_step,
            dynamic_sampling=dynamic_sampling,
            difficulty_filter=difficulty_filter,
            max_resample_attempts=max_resample_attempts,
        )
    elif action == "eval-tools-base":
        result = eval_tools.with_options(gpu=gpu).remote(
            adapter=None,
            run_id=TOOL_BASELINE_ID,
            limit=limit,
            tool_budget=tool_budget,
            prompt_version=prompt_version,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
    elif action == "eval-tools":
        import os

        chosen = adapter or f"{VOL_MOUNT}/runs/{TOOL_RUN_ID}/adapter"
        rid = run_id or (f"{TOOL_EVAL_ID}-{split}" if split != "test" else TOOL_EVAL_ID)
        result = eval_tools.with_options(gpu=gpu).remote(
            adapter=chosen,
            run_id=rid,
            limit=limit,
            split=split,
            hf_token=os.environ.get("HF_TOKEN", ""),
            tool_budget=tool_budget,
            prompt_version=prompt_version,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
    else:
        result = train.with_options(gpu=gpu).remote(
            resume=resume,
            run_id=run_id or RUN_ID_3B,
            init_adapter=init_adapter,
            max_steps=max_steps,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
    print(result)
    return result
