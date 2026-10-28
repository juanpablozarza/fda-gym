"""Tool-mode GPU jobs shared by Modal and a plain GPU box.

``modal_train.py`` calls these with ``runs_dir=/vol/runs`` and ``on_save`` set
to the volume commit. ``scripts/run_gpu.py`` calls them with a directory on
local disk and ``on_save=None``. Sampling, the task pool, and the defaults are
the arguments below. This module does not implement a reward.

``run_distill_sft`` is the student LoRA job over teacher JSONL. It does not
call a teacher API and does not rescore JP's reward.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fdagym.train.run_record import RunConflictError, finish_run, start_run, validate_run_id

_SECRET_ENV = ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_REPO = Path(__file__).resolve().parents[2]


def prepare_runtime(runs_dir: Path | str) -> Path:
    """Point the Hub cache at the disk that holds ``runs_dir``, and record the git SHA.

    ``setdefault`` keeps a value that is already there. Modal's ``_bind_paths``
    sets ``HF_HOME`` to ``/vol/hf-cache`` before the call, and the image sets
    ``FDAGYM_GIT_SHA`` because the container has no ``.git``. A box that has
    neither gets ``<parent of runs_dir>/hf-cache`` and ``git rev-parse HEAD``.
    """
    runs = Path(runs_dir)
    cache = str(runs.parent / "hf-cache")
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_HUB_CACHE", cache)
    if not os.environ.get("FDAGYM_GIT_SHA", "").strip():
        sha = _git_sha()
        if sha:
            os.environ["FDAGYM_GIT_SHA"] = sha
    return runs


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _redact(text: str) -> str:
    redacted = text
    for key in _SECRET_ENV:
        value = os.environ.get(key, "")
        if value:
            redacted = redacted.replace(value, "***")
    return redacted


def _mark(run_dir: Path, status: str, exc: BaseException | None = None, on_save=None) -> None:
    """Write ``DONE`` or ``FAILED`` and commit it when ``on_save`` is the volume.

    A missing directory means the job refused to start (bad ids, existing
    probe dir). Leave that directory alone.
    """
    if not run_dir.is_dir():
        return
    failed = status != "done"
    name = "FAILED" if failed else "DONE"
    other = run_dir / ("DONE" if failed else "FAILED")
    payload: dict = {"status": "failed" if failed else "done"}
    if exc is not None:
        payload["error"] = _redact(f"{type(exc).__name__}: {exc}")[:2000]
    finish_run(run_dir, payload["status"], payload.get("error"))
    (run_dir / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if other.exists():
        other.unlink()
    if on_save is not None:
        on_save()


def run_train_tools(
    *,
    runs_dir: Path | str,
    on_save=None,
    resume: bool = True,
    run_id: str = "grpo-qwen3b-tools",
    sft_run_id: str = "sft-qwen3b-tools",
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
    """Tool GRPO. Defaults match ``train_tools`` on Modal, including mix 0.

    ``on_save`` commits run metadata, trainer checkpoints, and the final ``DONE`` marker.
    ``init_adapter`` is a local directory or a Hugging Face repo id; private
    Hub adapters need ``HF_TOKEN`` in the environment.
    """
    validate_run_id(run_id)
    runs = prepare_runtime(runs_dir)
    run_dir = runs / run_id
    try:
        payload = _train_tools(
            runs=runs,
            on_save=on_save,
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
    except RunConflictError:
        raise
    except Exception as exc:
        _mark(run_dir, "failed", exc, on_save)
        raise
    _mark(run_dir, "done", on_save=on_save)
    return payload


def _train_tools(
    *,
    runs: Path,
    on_save,
    resume: bool,
    run_id: str,
    sft_run_id: str,
    init_adapter: str,
    tool_budget: int,
    prompt_version: str,
    hard_train_mix: float,
    max_steps: int,
    memorization_every: int,
    save_steps: int,
    wandb_project: str,
    wandb_entity: str,
    prompts_per_step: int | None,
    dynamic_sampling: bool | None,
    difficulty_filter: bool | None,
    max_resample_attempts: int | None,
    temperature: float,
    pool_types: str,
) -> dict:
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.train.grpo_tools import run_tool_grpo
    from fdagym.train.hard_mix import format_pool_exclusion_log, hard_mix_weights, prove_training_pool
    from fdagym.train.tool_config import configure_tool_run
    from fdagym.train.tool_guard import ParseRateGuard

    weights = hard_mix_weights(hard_train_mix)
    tool = configure_tool_run(
        hard=weights is not None,
        max_steps=max_steps,
        prompts_per_step=prompts_per_step,
        dynamic_sampling=dynamic_sampling,
        difficulty_filter=difficulty_filter,
        max_resample_attempts=max_resample_attempts,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        hard_train_mix=weights,
        save_steps=save_steps,
        temperature=temperature,
        pool_types=pool_types,
    )
    from dataclasses import asdict

    start_run(
        runs,
        run_id,
        job="tool-train",
        protocol="tool_rollout",
        resume=resume,
        config={
            "snapshot_version": "2026-10-15",
            "train": asdict(tool.train),
            "rollout": asdict(tool.limits),
            "tool_parse_bonus": tool.tool_parse_bonus,
            "hard_train_mix": weights,
            "memorization_every": memorization_every,
            "source_adapter": "explicit" if init_adapter else sft_run_id,
            "pool_types": list(tool.pool_types),
        },
    )
    if on_save is not None:
        on_save()
    tasks = load_split("train")
    hard_tasks = None
    proof = None
    if weights is not None:
        hard_tasks = load_split("hard_train")
        proof = prove_training_pool(tasks, hard_tasks)
        print(format_pool_exclusion_log(proof))
    summary = run_tool_grpo(
        DiskSnapshot("2026-10-15"),
        tasks,
        tool.train,
        run_id=run_id,
        results_dir=runs,
        sft_adapter=init_adapter.strip() if init_adapter else str(runs / sft_run_id / "adapter"),
        tool_parse_bonus=tool.tool_parse_bonus,
        limits=tool.limits,
        resume=resume,
        on_save=on_save,
        type_mix=tool.type_mix,
        mix_rows=tool.mix_rows,
        mix_seed=tool.mix_seed,
        guard=ParseRateGuard(window=tool.parse_stop_steps, min_rate=tool.parse_stop_min),
        hard_train_mix=weights,
        hard_tasks=hard_tasks,
        memorization=weights is not None,
        memorization_every=memorization_every,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        dynamic_sampling=tool.dynamic_sampling,
        max_resample_attempts=tool.max_resample_attempts,
        prompt_filter=tool.prompt_filter,
        prompts_per_step=tool.prompts_per_step,
        pool_types=tool.pool_types,
    )
    payload = {
        "run_id": summary["run_id"],
        "adapter": summary["adapter"],
        "n_prompts": summary["n_prompts"],
        "resumed": summary["resumed"],
    }
    if proof is not None:
        payload["pool_exclusion"] = {
            "n_holdout": proof["n_holdout"],
            "n_hard_eval": proof["n_hard_eval"],
            "n_hard_train": proof["n_hard_train"],
            "n_current": proof["n_current"],
            "n_pool": proof["n_pool"],
            "holdout_sha256": proof["holdout_sha256"],
            "hard_eval_sha256": proof["hard_eval_sha256"],
        }
    return payload


def run_eval_tools(
    *,
    runs_dir: Path | str,
    on_save=None,
    adapter: str | None = None,
    run_id: str = "grpo-qwen3b-tools-eval",
    limit: int | None = None,
    split: str = "test",
    tool_budget: int = 8,
    prompt_version: str = "v4",
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    """Greedy tool eval. ``HF_TOKEN`` is whatever the process environment already has.

    This function does not take the token. Modal copies its ``hf_token``
    argument into the environment before calling. The CLI never has that argument.
    """
    validate_run_id(run_id)
    runs = prepare_runtime(runs_dir)
    run_dir = runs / run_id
    try:
        run_dir = start_run(
            runs,
            run_id,
            job="tool-eval",
            protocol="tool_rollout",
            config={
                "snapshot_version": "2026-10-15",
                "split": split,
                "limit": limit,
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "adapter": adapter,
                "temperature": 0.0,
                "tool_budget": tool_budget,
                "prompt_version": prompt_version,
            },
        )
        if on_save is not None:
            on_save()
        payload = _eval_tools(
            runs=runs,
            adapter=adapter,
            run_id=run_id,
            limit=limit,
            split=split,
            tool_budget=tool_budget,
            prompt_version=prompt_version,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
    except RunConflictError:
        raise
    except Exception as exc:
        _mark(run_dir, "failed", exc, on_save)
        raise
    _mark(run_dir, "done", on_save=on_save)
    return payload


def _eval_tools(
    *,
    runs: Path,
    adapter: str | None,
    run_id: str,
    limit: int | None,
    split: str,
    tool_budget: int,
    prompt_version: str,
    wandb_project: str,
    wandb_entity: str,
) -> dict:
    from fdagym.train.config import QWEN_3B
    from fdagym.train.eval_tools import eval_tools_from_disk, tool_eval_model
    from fdagym.train.hf_rollout import load_hf_rollout
    from fdagym.train.rollout import RolloutLimits

    adapter_path = adapter or None
    generator, tokenizer = load_hf_rollout(QWEN_3B, adapter=adapter_path)
    summary = eval_tools_from_disk(
        tokenizer=tokenizer,
        generator=generator,
        model=tool_eval_model(adapter_path),
        run_id=run_id,
        split=split,
        results_dir=runs,
        limit=limit,
        limits=RolloutLimits(temperature=0.0, tool_budget=tool_budget, prompt_version=prompt_version),
        adapter=adapter_path,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    return {
        "run_id": summary["run_id"],
        "models": summary.get("models"),
        "protocol": summary.get("protocol"),
        "dense_reward": summary.get("dense_reward"),
        "prompt_version": summary.get("prompt_version"),
        "tool_budget": summary.get("tool_budget"),
    }


def run_probe_dispersion(
    *,
    runs_dir: Path | str,
    on_save=None,
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

    ``k`` must be 8. The pool is the one ``train_tools`` builds at
    ``hard_train_mix=0.6``. An existing ``<runs_dir>/<run_id>/`` is refused.
    No W&B and no HF token argument. Results land at
    ``<runs_dir>/<run_id>/rollouts.jsonl`` and ``probe.json``.
    ``on_save`` runs after the directory is created, after each chunk of 10
    groups, and after each ``probe.json`` write. The sampler is ``VLLMGenerator``.
    """
    import gc
    import shutil
    import time

    runs = prepare_runtime(runs_dir)

    def _commit(_state=None) -> None:
        if on_save is not None:
            on_save(_state)

    from fdagym.train.config import QWEN_3B
    from fdagym.train.dispersion import probe_task_ids, run_probe, sampling_param_kwargs
    from fdagym.train.rollout import RolloutTokenizer, VLLMGenerator
    from fdagym.train.sft_launch import adapter_sha256

    if k != 8:
        raise ValueError("probe k must be 8, the GRPO num_generations")
    if not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError(f"run_id must be one path segment, got {run_id!r}")
    # Id check before the directory, the adapter read, or any weight load.
    tool, env_tasks, ids = probe_task_ids(
        task_ids,
        task_ids_sha256,
        hard_train_mix=hard_train_mix,
        tool_budget=tool_budget,
        prompt_version=prompt_version,
        temperature=temperature,
        pool_types=pool_types,
    )
    run_dir = runs / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run {run_dir}")
    weights_sha = adapter_sha256(adapter)
    run_dir = start_run(
        runs,
        run_id,
        job="dispersion-probe",
        protocol="tool_rollout",
        config={
            "adapter_sha256": weights_sha,
            "task_ids": ids,
            "task_ids_sha256": task_ids_sha256,
            "k": k,
            "seed": seed,
            "hard_train_mix": hard_train_mix,
            "tool_budget": tool_budget,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "pool_types": list(tool.pool_types),
            "snapshot_version": "2026-10-15",
        },
    )
    _commit()

    from fdagym.openfda import DiskSnapshot

    snapshot = DiskSnapshot("2026-10-15")
    rollouts_path = run_dir / "rollouts.jsonl"
    probe_path = run_dir / "probe.json"
    seconds: dict[str, float] = {}
    versions: dict[str, str] = {}
    sampling_repr = None
    summary = None

    def _write(status: str) -> None:
        from dataclasses import asdict

        payload = {
            "status": status,
            "args": {
                "adapter": adapter,
                "task_ids": ids,
                "task_ids_sha256": task_ids_sha256,
                "run_id": run_id,
                "k": k,
                "seed": seed,
                "hard_train_mix": hard_train_mix,
                "tool_budget": tool_budget,
                "prompt_version": prompt_version,
                "temperature": temperature,
                "pool_types": list(tool.pool_types),
            },
            "adapter_sha256": weights_sha,
            "task_ids_sha256": task_ids_sha256,
            "limits": asdict(tool.limits),
            "tool_parse_bonus": tool.tool_parse_bonus,
            "dense_reward": tool.train.dense_reward,
            "num_generations": tool.train.num_generations,
            "sampling_params": sampling_repr,
            "sampling_param_kwargs": sampling_param_kwargs(tool.limits.temperature, tool.limits.max_new_tokens),
            "generator": "VLLMGenerator",
            "git_sha": os.environ.get("FDAGYM_GIT_SHA", ""),
            "versions": versions,
            "seconds": seconds,
            "summary": summary,
            "rollouts": str(rollouts_path),
        }
        probe_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _commit()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        versions["torch"] = torch.__version__
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(QWEN_3B, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(QWEN_3B, torch_dtype=torch.bfloat16, trust_remote_code=True)
        seconds["load"] = time.perf_counter() - started

        started = time.perf_counter()
        merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        merged_dir = Path("/tmp/merged")
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        del base, merged
        gc.collect()
        seconds["merge"] = time.perf_counter() - started

        started = time.perf_counter()
        import vllm
        from vllm import LLM, SamplingParams

        versions["vllm"] = vllm.__version__
        llm = LLM(
            model=str(merged_dir),
            tokenizer=QWEN_3B,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            seed=seed,
            enable_prefix_caching=True,
        )
        # Same constructor generate_with_vllm uses, recorded for probe.json.
        # VLLMGenerator is the only object that samples.
        sampling_repr = repr(
            SamplingParams(**sampling_param_kwargs(tool.limits.temperature, tool.limits.max_new_tokens))
        )
        seconds["engine"] = time.perf_counter() - started
        _write("sampling")

        started = time.perf_counter()
        summary = run_probe(
            snapshot,
            env_tasks,
            ids,
            RolloutTokenizer(tokenizer),
            VLLMGenerator(llm),
            tool,
            k=k,
            seed=seed,
            out_path=rollouts_path,
            on_chunk=_commit,
            run_id=run_id,
        )
        seconds["sampling"] = time.perf_counter() - started
        _write("complete")
        payload = {
            "n_groups": summary["n_groups"],
            "n_episodes": summary["n_episodes"],
            "frac_mixed_shaped": summary["frac_mixed_shaped"],
            "mean_within_std_shaped": summary["mean_within_std_shaped"],
            "probe_json": str(probe_path),
            "rollouts": str(rollouts_path),
            "adapter_sha256": weights_sha,
        }
    except Exception as exc:
        _mark(run_dir, "failed", exc, on_save)
        raise
    _mark(run_dir, "done", on_save=on_save)
    return payload


def run_distill_sft(
    *,
    runs_dir: Path | str,
    trajectories: list[Path] | Path,
    on_save=None,
    run_id: str = "distill-sft-qwen3b",
    sizes: str | list[str] = "100,250,500,1000,all",
    seed: int = 0,
    teachers: list[str] | None = None,
    templates: list[str] | None = None,
    max_attempts_per_question: int | None = None,
    init_adapter: str = "",
    model: str = "",
    learning_rate: float | None = None,
    num_train_epochs: float | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    max_length: int | None = None,
    warmup_ratio: float | None = None,
    save_steps: int | None = None,
    max_steps: int = 0,
    tool_budget: int = 8,
    prompt_version: str = "v4",
    eval_after: bool = False,
    eval_split: str = "hard",
    eval_limit: int | None = None,
    dry_run: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
) -> dict:
    """Nested-subset LoRA SFT on teacher trajectories.

    ``dry_run`` tokenizes with the CPU char tokenizer and never loads Qwen.
    ``eval_after`` chains each adapter into ``run_eval_tools`` (skipped on dry run).
    """
    from fdagym.train.config import QWEN_3B
    from fdagym.train.distill import DistillConfig, format_prepare_report, prepare_distill

    runs = prepare_runtime(runs_dir)
    kwargs: dict = {
        "run_id": run_id,
        "sizes": sizes,
        "seed": seed,
        "teachers": tuple(teachers or ()),
        "templates": tuple(templates or ()),
        "max_attempts_per_question": max_attempts_per_question,
        "init_adapter": init_adapter,
        "tool_budget": tool_budget,
        "prompt_version": prompt_version,
        "max_steps": max_steps,
        "eval_after": eval_after and not dry_run,
        "eval_split": eval_split,
        "eval_limit": eval_limit,
        "model": model or QWEN_3B,
    }
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        kwargs["num_train_epochs"] = num_train_epochs
    if per_device_train_batch_size is not None:
        kwargs["per_device_train_batch_size"] = per_device_train_batch_size
    if gradient_accumulation_steps is not None:
        kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
    if max_length is not None:
        kwargs["max_length"] = max_length
    if warmup_ratio is not None:
        kwargs["warmup_ratio"] = warmup_ratio
    if save_steps is not None:
        kwargs["save_steps"] = save_steps
    config = DistillConfig(**kwargs)
    parent = runs / run_id
    parent.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare_distill(trajectories, config)
        print(format_prepare_report(prepared), flush=True)
        (parent / "prepare.json").write_text(
            json.dumps(
                {"filter": prepared["filter"], "reports": prepared["reports"], "dry_run": dry_run},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if on_save is not None:
            on_save()
        if dry_run:
            return _distill_dry(prepared, parent, on_save)
        return _distill_fit(prepared, runs, on_save, wandb_project, wandb_entity)
    except Exception as exc:
        _mark(parent, "failed", exc, on_save)
        raise


def _distill_dry(prepared: dict, parent: Path, on_save) -> dict:
    from fdagym.train.distill import size_run_id
    from fdagym.train.distill_launch import dry_tokenize, write_manifest
    from fdagym.train.rollout_dry import CharTokenizer

    checks = dry_tokenize(prepared, CharTokenizer())
    config = prepared["config"]
    sizes = {}
    for name, check in checks.items():
        write_manifest(
            parent / f"n{name}",
            {
                "run_id": size_run_id(config.run_id, name),
                "size": name,
                "n": check["n"],
                "n_tokenized": check["n_tokenized"],
                "n_token_skipped": check["n_token_skipped"],
                "data_hash": check["data_hash"],
                "question_ids": check["question_ids"],
                "per_template_counts": check["per_template"],
                "teacher_mix": check["teacher_mix"],
                "hyperparameters": config.hyperparameters(),
                "git_sha": os.environ.get("FDAGYM_GIT_SHA", ""),
                "dry_run": True,
            },
        )
        sizes[name] = {
            "n": check["n"],
            "n_questions": check["n_questions"],
            "n_tokenized": check["n_tokenized"],
            "n_token_skipped": check["n_token_skipped"],
            "per_template": check["per_template"],
            "teacher_mix": check["teacher_mix"],
            "data_hash": check["data_hash"],
            "supervised_tokens": check["supervised_tokens"],
            "skipped": check["skipped"],
        }
    payload = {
        "run_id": config.run_id,
        "dry_run": True,
        "filter": prepared["filter"],
        "sizes": sizes,
    }
    (parent / "dry-run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _mark(parent, "done", on_save=on_save)
    return payload


def _distill_fit(prepared: dict, runs: Path, on_save, wandb_project: str, wandb_entity: str) -> dict:
    from fdagym.train.distill import size_run_id
    from fdagym.train.distill_launch import default_eval_run_id, fit_subset

    config = prepared["config"]
    parent = runs / config.run_id
    fits = []
    evals = []
    for name, rows in prepared["subsets"].items():
        run_id = size_run_id(config.run_id, name)
        run_dir = runs / run_id
        try:
            report = fit_subset(
                rows,
                config,
                run_id=run_id,
                results_dir=runs,
                size=name,
                filter_stats=prepared["filter"],
                on_save=on_save,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            )
            _mark(run_dir, "done", on_save=on_save)
        except Exception as exc:
            _mark(run_dir, "failed", exc, on_save)
            raise
        fits.append({"size": name, "run_id": run_id, "adapter": report["adapter"], "n": report["n"]})
        if not config.eval_after:
            continue
        eval_id = default_eval_run_id(run_id, config.eval_split)
        evaluated = run_eval_tools(
            runs_dir=runs,
            on_save=on_save,
            adapter=report["adapter"],
            run_id=eval_id,
            limit=config.eval_limit,
            split=config.eval_split,
            tool_budget=config.tool_budget,
            prompt_version=config.prompt_version,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
        evals.append({"size": name, "run_id": eval_id, **evaluated})
    payload = {
        "run_id": config.run_id,
        "dry_run": False,
        "filter": prepared["filter"],
        "fits": fits,
        "evals": evals,
        "reports": prepared["reports"],
    }
    (parent / "curve.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _mark(parent, "done", on_save=on_save)
    return payload

