"""Weights & Biases as the run record. The API key is read from the environment only.

Modal injects it from the secret named ``wandb`` (``WANDB_API_KEY``). This module
never takes the key as an argument and never prints it. With no key, or if W&B
itself fails, training continues and ``report_to`` stays empty.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SENSITIVE = ("api_key", "token", "secret", "password", "credential")
_FLAG_LEVEL = {"ok": 0, "warn": 1, "stop and review": 2}


def wandb_key_present() -> bool:
    return bool(os.environ.get("WANDB_API_KEY", "").strip())


def resolve_wandb_target(project: str | None = None, entity: str | None = None) -> tuple[str, str | None]:
    """Project defaults to ``bio-rl``. An empty flag falls through to the environment."""
    chosen_project = (project or "").strip() or os.environ.get("WANDB_PROJECT", "").strip() or "bio-rl"
    chosen_entity = (entity or "").strip() or os.environ.get("WANDB_ENTITY", "").strip() or None
    return chosen_project, chosen_entity


def git_sha() -> str:
    baked = os.environ.get("FDAGYM_GIT_SHA", "").strip()
    if baked:
        return baked
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def step_from_checkpoint(path: str | Path | None) -> int | None:
    if not path:
        return None
    suffix = Path(path).name.split("-", 1)[-1]
    if Path(path).name.startswith("checkpoint-") and suffix.isdigit():
        return int(suffix)
    return None


def resume_floor(*, resumed: bool, summary_step: int | None, resume_step: int | None) -> int | None:
    """Last step already stored on a resumed run. Logs at or below it are dropped.

    A new run (``resumed`` false) logs from the trainer's first step, even when a
    checkpoint exists in a project that has never seen this id.
    """
    if not resumed:
        return None
    steps = [step for step in (summary_step, resume_step) if step is not None]
    if not steps:
        return None
    return max(steps)


def should_skip_log(data, step, floor: int | None) -> bool:
    if floor is None or not isinstance(data, dict):
        return False
    logged = data.get("train/global_step", step)
    if isinstance(logged, bool) or not isinstance(logged, (int, float)):
        return False
    return int(logged) <= floor


def sensitive_key(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _SENSITIVE)


def scrub_config(payload: dict) -> dict:
    """Drop credential-shaped keys and any value equal to the live API key."""
    secret = os.environ.get("WANDB_API_KEY", "").strip()

    def clean(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if sensitive_key(str(key)):
                    continue
                cleaned[key] = clean(item)
            return cleaned
        if isinstance(value, list):
            return [clean(item) for item in value]
        if secret and value == secret:
            return None
        return value

    return clean(payload)


def build_run_config(
    *,
    run_id: str,
    job: str,
    prompt_version: str | None,
    tool_budget: int | None,
    hard_train_mix,
    seeds,
    adapter: str | None,
    model: str | None = None,
    extra: dict | None = None,
) -> dict:
    config = {
        "run_id": run_id,
        "job": job,
        "prompt_version": prompt_version,
        "tool_budget": tool_budget,
        "hard_train_mix": hard_train_mix,
        "seeds": list(seeds) if seeds is not None else [],
        "adapter": None if adapter is None else str(adapter),
        "git_sha": git_sha(),
    }
    if model is not None:
        config["model"] = model
    if extra:
        config.update(extra)
    return scrub_config(config)


def is_reward_log(logs: dict) -> bool:
    """True for a trainer log that carries the shaped reward. Eval-loss rows do not."""
    if "reward" in logs:
        return True
    return any(key.endswith("/mean") and "reward" in key for key in logs)


def _metric_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _flatten_numbers(node, prefix: str, out: dict) -> None:
    """Walk a report. ``swapped_facts`` is logged as ``memorization/swapped``."""
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "flags":
            continue
        name = "swapped" if key == "swapped_facts" else key
        path = f"{prefix}/{name}"
        number = _metric_number(value)
        if number is not None:
            out[path] = number
        elif isinstance(value, dict):
            _flatten_numbers(value, path, out)


def flatten_memorization(report: dict) -> dict:
    """Every numeric field under ``memorization/*``, plus the flag names.

    Missing rates stay absent. The look-rule string stays out of the metric
    dict. ``memorization/no_look`` remains the short name for the pooled rate.
    """
    out: dict = {}
    if not isinstance(report, dict):
        return out
    _flatten_numbers(report, "memorization", out)
    looked = report.get("answered_without_looking")
    no_look = _metric_number(looked.get("rate")) if isinstance(looked, dict) else None
    if no_look is not None:
        out["memorization/no_look"] = no_look
    flags = report.get("flags") or {}
    if isinstance(flags, dict):
        for name in ("gap", "answered_without_looking", "swapped_facts", "level"):
            label = flags.get(name)
            if isinstance(label, str) and label in _FLAG_LEVEL:
                out[f"memorization/flags/{name}"] = _FLAG_LEVEL[label]
                out[f"memorization/flags/{name}_name"] = label
    if "auto_stop" in report:
        out["memorization/auto_stop"] = bool(report["auto_stop"])
    return out


def write_memorization_report(report: dict, directory: str | Path, step: int) -> Path:
    """Write the full report beside the run outputs.

    The path is ``<directory>/memorization-<step>.json``. On Modal that directory
    is ``/vol/runs/<run_id>``.
    """
    path = Path(directory) / f"memorization-{int(step)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def merge_memorization(logs: dict, step: int, measure, *, every: int, report_dir=None) -> dict:
    """Attach flat memorization fields on a reward log every ``every`` steps.

    The measurement function is not called for eval-loss rows or off-cycle steps.
    A produced report is also written under ``report_dir`` when that path is set.
    Nothing here stops the trainer.
    """
    from fdagym.memorization import record_training_memorization

    if not is_reward_log(logs):
        return logs
    attached = record_training_memorization({"step": step}, step, measure, every=every)
    report = (attached or {}).get("memorization")
    if report:
        logs.update(flatten_memorization(report))
        if report_dir:
            write_memorization_report(report, report_dir, step)
    return logs


def attach_grpo_train_logs(
    logs,
    *,
    step: int,
    measure=None,
    every: int = 10,
    guard=None,
    control=None,
    report_dir=None,
) -> dict:
    """Mutate GRPO logs before Hugging Face's W&B callback sees them.

    ``Trainer.log`` runs default callbacks (including ``WandbCallback``) before
    any user ``on_log``. The caller must already have merged averaged
    ``reward`` and ``rewards/*`` metrics into ``logs``. TRL 0.21 adds those
    only inside ``GRPOTrainer.log``, which is too late for this hook.
    """
    if not isinstance(logs, dict):
        return logs
    merge_memorization(logs, step, measure, every=every, report_dir=report_dir)
    if guard is None:
        return logs
    reason = guard.observe(step, logs)
    if reason:
        logs["parse_stop"] = reason
        if control is not None and not getattr(control, "should_training_stop", False):
            control.should_training_stop = True
    return logs


def eval_history_metrics(summary: dict, tasks: list[dict], trajectories: list[dict]) -> dict:
    """Strict, pass, and per-type rates for one eval summary. The summary file is not rewritten."""
    models = summary.get("models") or {}
    if not models:
        return {}
    name = next(iter(models))
    board = summary.get("leaderboard") or []
    if board and board[0].get("name") in models:
        name = board[0]["name"]
    row = models[name]
    metrics: dict = {}
    if isinstance(row.get("mean_reward"), (int, float)):
        metrics["eval/strict"] = float(row["mean_reward"])
    if isinstance(row.get("pass_rate"), (int, float)):
        metrics["eval/pass"] = float(row["pass_rate"])
    by_id = {task.get("id"): task.get("type") for task in tasks}
    groups: dict[str, list[dict]] = {}
    for trajectory in trajectories:
        kind = by_id.get(trajectory.get("task_id"))
        reward = trajectory.get("reward") or {}
        if not isinstance(kind, str) or not kind:
            continue
        if not isinstance(reward.get("total"), (int, float)):
            continue
        groups.setdefault(kind, []).append(trajectory)
    for kind, rows in sorted(groups.items()):
        n = len(rows)
        metrics[f"eval/type/{kind}/n"] = n
        metrics[f"eval/type/{kind}/strict"] = sum(float(item["reward"]["total"]) for item in rows) / n
        metrics[f"eval/type/{kind}/pass"] = (
            sum(1 for item in rows if float(item["reward"].get("correctness") or 0) > 0) / n
        )
    memo = summary.get("memorization")
    if isinstance(memo, dict):
        metrics.update(flatten_memorization(memo))
    return metrics


def _summary_step(run) -> int | None:
    try:
        summary = getattr(run, "summary", None)
        if summary is None:
            return None
        value = summary.get("train/global_step")
    except Exception:  # noqa: BLE001 — a summary read must not kill the run
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _install_log_guard(module, floor: int | None) -> None:
    current = module.log
    if getattr(current, "_fdagym_guard", False):
        current._fdagym_floor = floor
        return

    def guarded(data=None, step=None, commit=None, **kwargs):
        try:
            if should_skip_log(data, step, guarded._fdagym_floor):
                return None
            return current(data, step=step, commit=commit, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a W&B outage must not kill training
            print(f"wandb log failed ({type(exc).__name__}); training continues", flush=True)
            return None

    guarded._fdagym_guard = True
    guarded._fdagym_floor = floor
    module.log = guarded


def _install_config_guard(run) -> None:
    config = getattr(run, "config", None)
    if config is None or not hasattr(config, "update"):
        return
    current = config.update
    if getattr(current, "_fdagym_guard", False):
        return

    def guarded(payload=None, *args, **kwargs):
        try:
            if isinstance(payload, dict):
                payload = scrub_config(payload)
            return current(payload, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a W&B outage must not kill training
            print(f"wandb config update failed ({type(exc).__name__}); training continues", flush=True)
            return None

    guarded._fdagym_guard = True
    config.update = guarded


@dataclass
class WandbSession:
    enabled: bool
    report_to: list[str]
    project: str
    entity: str | None
    run_id: str
    floor: int | None = None
    _finished: bool = field(default=False, repr=False)

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        if not self.enabled or not metrics:
            return
        try:
            import wandb
        except ImportError:
            return
        payload = dict(metrics)
        if step is not None and "train/global_step" not in payload:
            payload["train/global_step"] = step
        wandb.log(payload)

    def finish(self) -> None:
        if not self.enabled or self._finished:
            return
        self._finished = True
        try:
            import wandb

            wandb.finish()
        except Exception as exc:  # noqa: BLE001 — a W&B outage must not kill training
            print(f"wandb finish failed ({type(exc).__name__}); training continues", flush=True)


def start_wandb(
    run_id: str,
    config: dict,
    *,
    project: str | None = None,
    entity: str | None = None,
    resume_step: int | None = None,
) -> WandbSession:
    """Open run ``run_id`` with ``resume='allow'``, or a disabled session.

    The key is not an argument. A missing key, a missing install, or an init
    error all return ``report_to=[]`` so TRL does not load the W&B callback.
    """
    chosen_project, chosen_entity = resolve_wandb_target(project, entity)
    disabled = WandbSession(
        enabled=False,
        report_to=[],
        project=chosen_project,
        entity=chosen_entity,
        run_id=run_id,
    )
    if not wandb_key_present():
        return disabled
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing with W&B disabled", flush=True)
        return disabled
    safe_config = scrub_config(dict(config))
    init_kwargs = {
        "project": chosen_project,
        "id": run_id,
        "name": run_id,
        "resume": "allow",
        "config": safe_config,
    }
    if chosen_entity:
        init_kwargs["entity"] = chosen_entity
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:  # noqa: BLE001 — init failure disables W&B and training continues
        print(f"wandb init failed ({type(exc).__name__}); continuing with W&B disabled", flush=True)
        return disabled
    floor = resume_floor(
        resumed=bool(getattr(run, "resumed", False)),
        summary_step=_summary_step(run),
        resume_step=resume_step,
    )
    _install_log_guard(wandb, floor)
    _install_config_guard(run)
    url = getattr(run, "url", None)
    if isinstance(url, str) and url and "api_key" not in url.lower():
        print(f"wandb: {url}", flush=True)
    return WandbSession(
        enabled=True,
        report_to=["wandb"],
        project=chosen_project,
        entity=chosen_entity,
        run_id=run_id,
        floor=floor,
    )


def log_eval_summary(
    *,
    run_id: str,
    summary: dict,
    tasks: list[dict],
    trajectories: list[dict],
    prompt_version: str | None,
    tool_budget: int | None,
    adapter: str | None,
    model: str | None,
    project: str | None = None,
    entity: str | None = None,
) -> WandbSession:
    """One history row for an eval run. ``summary.json`` is left untouched."""
    session = start_wandb(
        run_id,
        build_run_config(
            run_id=run_id,
            job="eval",
            prompt_version=prompt_version,
            tool_budget=tool_budget,
            hard_train_mix=None,
            seeds=summary.get("seeds") or [],
            adapter=adapter,
            model=model,
            extra={"split": summary.get("split")},
        ),
        project=project,
        entity=entity,
    )
    try:
        # No train/global_step: an eval is one row, and a later memorization attach can log again.
        session.log(eval_history_metrics(summary, tasks, trajectories))
    finally:
        session.finish()
    return session
