"""Compare two Plan A eval summaries.

Greedy eval (temperature 0) still wanders a bit across seeds. A 0.001 bump is not a win.
Default gate: mean delta >= MIN_MARGIN (0.03). If both runs logged >=3 seed means, also
require the 95% CI on the difference to stay non-negative.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MIN_MARGIN = 0.03


def load_summary(path: str | Path) -> dict:
    target = Path(path)
    if target.is_dir():
        target = target / "summary.json"
    return json.loads(target.read_text())


def _model_row(summary: dict, name: str | None = None) -> dict:
    models = summary.get("models")
    if isinstance(models, dict):
        if not models:
            raise ValueError(f"no models in summary {summary.get('run_id')!r}")
        key = name if name is not None else next(iter(models))
        row = dict(models[key])
        row.setdefault("name", key)
        if "pass_rate" not in row and "pass" in row:
            row["pass_rate"] = row["pass"]
        return row
    if isinstance(models, list) and models:
        if name is None:
            row = dict(models[0])
        else:
            row = dict(next(item for item in models if item["name"] == name))
        if "pass_rate" not in row and "pass" in row:
            row["pass_rate"] = row["pass"]
        return row
    raise ValueError(f"no models in summary {summary.get('run_id')!r}")


def _ci95_diff(trained: list[float], baseline: list[float]) -> tuple[float, float] | None:
    if len(trained) < 3 or len(baseline) < 3:
        return None
    t_mean = sum(trained) / len(trained)
    b_mean = sum(baseline) / len(baseline)
    t_var = sum((x - t_mean) ** 2 for x in trained) / (len(trained) - 1)
    b_var = sum((x - b_mean) ** 2 for x in baseline) / (len(baseline) - 1)
    se = math.sqrt(t_var / len(trained) + b_var / len(baseline))
    delta = t_mean - b_mean
    half = 1.96 * se
    return delta - half, delta + half


def compare_summaries(
    baseline: dict,
    trained: dict,
    *,
    baseline_name=None,
    trained_name=None,
    min_margin: float = MIN_MARGIN,
) -> dict:
    base = _model_row(baseline, baseline_name)
    ours = _model_row(trained, trained_name)
    delta = float(ours["mean_reward"]) - float(base["mean_reward"])
    beats = delta >= min_margin
    ci = _ci95_diff(list(trained.get("seed_means") or []), list(baseline.get("seed_means") or []))
    if ci is not None:
        beats = beats and ci[0] >= 0
    return {
        "baseline_run": baseline.get("run_id"),
        "trained_run": trained.get("run_id"),
        "baseline_mean_reward": float(base["mean_reward"]),
        "trained_mean_reward": float(ours["mean_reward"]),
        "delta_mean_reward": round(delta, 4),
        "min_margin": min_margin,
        "ci95_delta": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "baseline_pass_rate": float(base.get("pass_rate") or 0),
        "trained_pass_rate": float(ours.get("pass_rate") or 0),
        "beats_baseline": beats,
    }


def format_compare(row: dict) -> str:
    verdict = "BEATS baseline" if row["beats_baseline"] else "does not beat baseline"
    extra = f" margin={row['min_margin']}"
    if row.get("ci95_delta"):
        extra += f" ci95={row['ci95_delta']}"
    return (
        f"{row['trained_run']} {verdict} vs {row['baseline_run']}: "
        f"{row['baseline_mean_reward']:.4f} → {row['trained_mean_reward']:.4f} "
        f"(Δ {row['delta_mean_reward']:+.4f},{extra})"
    )
