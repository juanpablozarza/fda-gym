"""Collect hard-eval summaries into a learning-curve CSV and SVG.

Reads ``<runs-dir>/<prefix>-n<size>-<split>/summary.json`` plus the jsonl and
``tasks.json`` sidecar that ``eval-tools`` already writes. Does not touch
``data/tasks/hard.jsonl``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from fdagym.train.distill import parse_sizes, size_run_id


def eval_run_id(prefix: str, size: str, split: str = "hard") -> str:
    return f"{size_run_id(prefix, size)}-{split}"


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_reward_from_summary(summary: dict) -> dict:
    models = summary.get("models") or {}
    if not models:
        return {"n": 0, "mean_reward": None, "pass_rate": None, "mean_citation": None, "mean_tools": None}
    name = next(iter(models))
    board = summary.get("leaderboard") or []
    if board and board[0].get("name") in models:
        name = board[0]["name"]
    row = models[name]
    return {
        "model": name,
        "n": row.get("n"),
        "mean_reward": row.get("mean_reward"),
        "pass_rate": row.get("pass_rate"),
        "mean_citation": row.get("mean_citation"),
        "mean_tools": row.get("mean_tools"),
    }


def variant_of(task: dict | None, task_id: str) -> str:
    if not task:
        return "unknown"
    question = task.get("question") or ""
    if question:
        try:
            from fdagym.tasks.hard_solver import Unsolvable, parse_question

            parsed = parse_question(question)
        except Unsolvable:
            parsed = {}
        variant = parsed.get("variant") if isinstance(parsed, dict) else None
        if isinstance(variant, str) and variant:
            return variant
    kind = task.get("type")
    if isinstance(kind, str) and kind:
        return kind
    return task_id


def per_variant_from_jsonl(jsonl_path: Path, tasks: list[dict]) -> list[dict]:
    if not jsonl_path.is_file():
        return []
    by_id = {task.get("id"): task for task in tasks if isinstance(task, dict)}
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        reward = row.get("reward") or {}
        if not isinstance(reward.get("total"), (int, float)):
            continue
        groups[variant_of(by_id.get(task_id), str(task_id))].append(row)
    out = []
    for variant, rows in sorted(groups.items()):
        n = len(rows)
        out.append(
            {
                "variant": variant,
                "n": n,
                "mean_reward": round(sum(float(item["reward"]["total"]) for item in rows) / n, 4),
                "pass_rate": round(
                    sum(1 for item in rows if float(item["reward"].get("correctness") or 0) > 0) / n, 4
                ),
            }
        )
    return out


def collect_curve(
    runs_dir: Path,
    prefix: str,
    sizes=None,
    *,
    split: str = "hard",
) -> list[dict]:
    """One overall row per size, plus one row per variant when jsonl is present."""
    names = parse_sizes(sizes)
    runs = Path(runs_dir)
    records: list[dict] = []
    for name in names:
        run_id = eval_run_id(prefix, name, split)
        run_dir = runs / run_id
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            records.append(
                {
                    "size": name,
                    "size_n": None if name == "all" else int(name),
                    "run_id": run_id,
                    "variant": "",
                    "missing": True,
                }
            )
            continue
        summary = load_json(summary_path)
        overall = mean_reward_from_summary(summary)
        train_id = size_run_id(prefix, name)
        manifest_path = runs / train_id / "manifest.json"
        size_n = None
        if manifest_path.is_file():
            size_n = load_json(manifest_path).get("n")
        if size_n is None and name != "all":
            size_n = int(name)
        base = {
            "size": name,
            "size_n": size_n,
            "run_id": run_id,
            "missing": False,
            **overall,
        }
        records.append({**base, "variant": ""})
        tasks_path = run_dir / "tasks.json"
        tasks = load_json(tasks_path) if tasks_path.is_file() else []
        jsonl = runs / f"{run_id}.jsonl"
        for variant_row in per_variant_from_jsonl(jsonl, tasks if isinstance(tasks, list) else []):
            records.append(
                {
                    **base,
                    "n": variant_row["n"],
                    "mean_reward": variant_row["mean_reward"],
                    "pass_rate": variant_row["pass_rate"],
                    "variant": variant_row["variant"],
                    "mean_citation": None,
                    "mean_tools": None,
                }
            )
    return records


def write_csv(records: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "size",
        "size_n",
        "run_id",
        "variant",
        "n",
        "mean_reward",
        "pass_rate",
        "mean_citation",
        "mean_tools",
        "missing",
        "model",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fields})
    return path


def write_svg(records: list[dict], path: Path, *, metric: str = "mean_reward") -> Path:
    """Minimal SVG so the curve does not need matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        if row.get("missing"):
            continue
        value = row.get(metric)
        x = row.get("size_n")
        if not isinstance(value, (int, float)) or not isinstance(x, (int, float)):
            continue
        label = row.get("variant") or "overall"
        series[str(label)].append((float(x), float(value)))
    width, height = 720, 420
    pad_l, pad_r, pad_t, pad_b = 56, 24, 28, 48
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    xs = [point[0] for points in series.values() for point in points]
    ys = [point[1] for points in series.values() for point in points]
    min_x, max_x = (min(xs), max(xs)) if xs else (0.0, 1.0)
    min_y, max_y = (0.0, 1.0)
    if ys:
        min_y = min(0.0, min(ys))
        max_y = max(1.0, max(ys)) if max(ys) > 1 else 1.0
        if max_y == min_y:
            max_y = min_y + 1.0
    if max_x == min_x:
        max_x = min_x + 1.0

    def px(x: float) -> float:
        return pad_l + (x - min_x) / (max_x - min_x) * inner_w

    def py(y: float) -> float:
        return pad_t + (1 - (y - min_y) / (max_y - min_y)) * inner_h

    colors = ["#F5A524", "#67B7D1", "#E45756", "#54A24B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f1115"/>',
        f'<text x="{pad_l}" y="18" fill="#F5A524" font-size="14">distill SFT {metric} vs training size</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#444" />',
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#444" />',
        f'<text x="{pad_l}" y="{height - 16}" fill="#aaa" font-size="11">n trajectories</text>',
    ]
    for index, (label, points) in enumerate(sorted(series.items(), key=lambda item: (item[0] != "overall", item[0]))):
        points = sorted(points)
        color = colors[index % len(colors)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')
        parts.append(
            f'<text x="{width - pad_r - 4}" y="{pad_t + 14 + index * 16}" fill="{color}" '
            f'font-size="11" text-anchor="end">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def write_curve(records: list[dict], out_prefix: Path) -> dict:
    out_prefix = Path(out_prefix)
    csv_path = write_csv(records, out_prefix.with_suffix(".csv") if out_prefix.suffix != ".csv" else out_prefix)
    stem = csv_path.with_suffix("")
    svg_path = write_svg(records, stem.with_suffix(".svg"))
    return {"csv": str(csv_path), "svg": str(svg_path), "n": len(records)}
