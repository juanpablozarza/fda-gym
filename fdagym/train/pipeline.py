"""Idempotent, gated stage runner for the server side Modal pipelines.

Lessons from the 2026-10-18 run: a surprise container restart re-ran the whole
chain from SFT and nearly overwrote finished adapters. So every stage here:
  * is skipped when its output already exists (never overwrite a finished adapter),
  * records progress in status.json on the volume,
  * and the runner refuses to start when status says the pipeline already finished
    or failed, unless ``force=True``.
Pure Python, no Modal imports, so it is unit tested.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TERMINAL = ("done", "gate-failed")


@dataclass
class Stage:
    name: str
    run: Callable[[dict], Any]
    # Path whose existence means the stage already finished (adapter dir or eval summary).
    output: Path | None = None
    # Optional gate: given the status, return False to stop the pipeline after this stage.
    gate: Callable[[dict], bool] | None = None


def find_mean(obj: Any) -> list[float]:
    """Every ``mean_reward`` number in a nested eval summary, depth first."""
    found: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "dense_reward":
                continue  # gate on the strict score only
            if k == "mean_reward" and isinstance(v, (int, float)):
                found.append(float(v))
            else:
                found += find_mean(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += find_mean(v)
    return found


def strict_mean(summary: Any) -> float:
    means = find_mean(summary)
    return means[-1] if means else 0.0


def adapter_done(path: Path) -> bool:
    return (Path(path) / "adapter_model.safetensors").exists()


def tool_artifact_paths(
    runs: Path,
    *,
    baseline_id: str,
    sft_id: str,
    grpo_id: str,
    grpo_eval_id: str,
    init_adapter: str = "",
) -> dict[str, Path]:
    """Volume paths the tool pipeline uses to skip a stage that already finished.

    ``eval_start`` matches ``pipeline_tools``: ``<adapter parent>-tools-eval``.
    With no ``init_adapter`` that parent is ``sft_id``, so the gate file is
    ``<sft_id>-tools-eval/summary.json``.
    """
    runs = Path(runs)
    sft_adapter = Path(init_adapter) if init_adapter else runs / sft_id / "adapter"
    return {
        "eval_base": runs / baseline_id / "summary.json",
        "sft_adapter": sft_adapter,
        "eval_start": runs / f"{sft_adapter.parent.name}-tools-eval" / "summary.json",
        "grpo_adapter": runs / grpo_id / "adapter",
        "eval_grpo": runs / grpo_eval_id / "summary.json",
    }


class Pipeline:
    def __init__(self, out_dir: Path, stages: list[Stage], commit: Callable[[], None] = lambda: None):
        self.out = Path(out_dir)
        self.stages = stages
        self.commit = commit
        self.status_path = self.out / "status.json"

    def load(self) -> dict:
        if self.status_path.exists():
            return json.loads(self.status_path.read_text())
        return {"stage": "start", "history": [], "results": {}}

    def save(self, status: dict, stage: str, **kw: Any) -> None:
        status["stage"] = stage
        status.update(kw)
        status.setdefault("history", []).append({"t": time.time(), "stage": stage})
        self.out.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(status, indent=2, default=str))
        with open(self.out / "pipeline.log", "a") as f:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            f.write(f"{stamp} {stage} {json.dumps(kw, default=str)[:2000]}\n")
        self.commit()

    def run(self, force: bool = False) -> dict:
        status = self.load()
        if not force and (status["stage"] in TERMINAL or status["stage"].startswith("failed")):
            raise RuntimeError(
                f"pipeline already at {status['stage']!r} ({self.status_path}); "
                "refusing to restart. Pass force=True after checking outputs."
            )
        status.setdefault("results", {})
        for st in self.stages:
            out = Path(st.output) if st.output is not None else None
            if out is not None and (adapter_done(out) or out.is_file()):
                # A skipped eval must still fill results, or the gate reads 0 and stops.
                if out.is_file() and st.name not in status["results"]:
                    status["results"][st.name] = json.loads(out.read_text())
                self.save(status, f"skip-{st.name}", note=f"{out} exists, not overwriting")
            else:
                self.save(status, st.name)
                try:
                    status["results"][st.name] = st.run(status)
                except Exception as e:
                    self.save(status, f"failed-{st.name}", error=repr(e))
                    raise
            if st.gate is not None and not st.gate(status):
                self.save(status, "gate-failed", gate_stage=st.name)
                return status
        self.save(status, "done")
        return status
