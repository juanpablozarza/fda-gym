"""No-GPU check: one application-number search per gold Drugs@FDA id.

Reports, for train and test approval tasks, whether that search returns the gold
application and whether the id plus the answer evidence sit in the first 3,500
characters the model would see. Also replays FG-0002 through FDAEnv.

    uv run python scripts/check_approval_reachability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdagym.env import FDAEnv
from fdagym.openfda.store import DiskSnapshot
from fdagym.tools import EpisodeState, execute
from fdagym.train.gold import gold_submit
from fdagym.train.rollout import clip_text, observation_payload

ROOT = Path(__file__).resolve().parents[1]
VISIBLE_CHARS = 3500


def _tasks(task_dir: Path) -> list[dict]:
    rows = []
    for name in ("train.jsonl", "test.jsonl"):
        path = task_dir / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            if task.get("type") == "approval":
                task["_split"] = name.removesuffix(".jsonl")
                rows.append(task)
    return rows


def _gold_ids(task: dict) -> list[str]:
    ids: list[str] = []
    for gold in task.get("gold_citations") or []:
        for cid in gold.get("ids") or []:
            if cid not in ids:
                ids.append(cid)
    return ids


def _visible(output: dict) -> str:
    observation = {
        "tool": "search_approvals",
        "output": output,
        "error": None,
        "tool_calls_remaining": 5,
    }
    return clip_text(observation_payload(observation), VISIBLE_CHARS)


def _orig_dates(record: dict) -> list[str]:
    dates = []
    for sub in record.get("submissions") or []:
        if not isinstance(sub, dict):
            continue
        if str(sub.get("submission_type", "")).upper() != "ORIG":
            continue
        date = sub.get("submission_status_date")
        if isinstance(date, str) and date:
            dates.append(date)
    return dates


def _answer_visible(task: dict, record: dict, visible: str) -> bool:
    """The cited field's answer text survived the 3,500-character clip."""
    field = (task.get("gold_citations") or [{}])[0].get("fields", [""])[0]
    if field == "submissions":
        year = str(task["gold"]["value"])
        return any(date.startswith(year) and date in visible for date in _orig_dates(record))
    wanted = str(task["gold"]["value"]).casefold()
    return wanted in visible.casefold()


def _check_id(snapshot, task: dict, application_id: str) -> dict:
    output, error = execute(
        EpisodeState(snapshot),
        "search_approvals",
        {"application_number": application_id},
        step_index=1,
    )
    results = output.get("results") if isinstance(output, dict) else None
    record = next(
        (row for row in results or [] if row.get("application_number") == application_id),
        None,
    )
    visible = _visible(output) if error is None else ""
    return {
        "id": application_id,
        "error": error,
        "reachable": record is not None,
        "id_visible": application_id in visible,
        "answer_visible": record is not None and _answer_visible(task, record, visible),
        "observation_chars": len(json.dumps(output, ensure_ascii=False)) if error is None else None,
    }


def _summarize(rows: list[dict]) -> dict:
    tasks = len(rows)
    ids = sum(len(row["ids"]) for row in rows)
    reachable_ids = sum(1 for row in rows for item in row["ids"] if item["reachable"])
    visible_ids = sum(
        1 for row in rows for item in row["ids"] if item["reachable"] and item["answer_visible"] and item["id_visible"]
    )
    id_only = sum(1 for row in rows for item in row["ids"] if item["id_visible"] and not item["answer_visible"])
    tasks_ok = sum(
        1
        for row in rows
        if row["ids"]
        and all(item["reachable"] and item["id_visible"] and item["answer_visible"] for item in row["ids"])
    )
    return {
        "tasks": tasks,
        "gold_ids": ids,
        "ids_reachable": reachable_ids,
        "ids_visible": visible_ids,
        "ids_id_visible_answer_clipped": id_only,
        "tasks_reachable_and_visible": tasks_ok,
        "fraction_ids_reachable": round(reachable_ids / ids, 4) if ids else None,
        "fraction_ids_visible": round(visible_ids / ids, 4) if ids else None,
        "fraction_tasks_reachable_and_visible": round(tasks_ok / tasks, 4) if tasks else None,
    }


def _fg0002(snapshot, tasks: list[dict]) -> dict:
    task = next(row for row in tasks if row["id"] == "FG-0002")
    env = FDAEnv(snapshot, tasks, run_id="fg-0002-check")
    env.reset(task["id"])
    obs, _reward, done, _info = env.step({"tool": "search_approvals", "arguments": {"application_number": "NDA020977"}})
    if done or obs["error"]:
        return {"ok": False, "error": obs["error"], "done": done}
    visible = _visible(obs["output"])
    _obs, _reward, done, info = env.step({"tool": "submit_answer", "arguments": gold_submit(task, env.steps)})
    reward = info["reward"]
    return {
        "ok": done and reward["total"] == 1.0,
        "application_number": "NDA020977",
        "brand_in_result": "ZIAGEN" in (obs["output"]["results"][0].get("brand") or []),
        "reward_total": reward["total"],
        "correctness": reward["correctness"],
        "citation": reward["citation"],
        "id_visible": "NDA020977" in visible,
        "answer_visible": "ziagen" in visible.casefold(),
        "visible_chars": len(visible),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="2026-10-15")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "tasks")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "approval-reachability.json")
    args = parser.parse_args()

    snapshot = DiskSnapshot(args.snapshot)
    tasks = _tasks(args.tasks)
    checked = []
    for task in tasks:
        ids = [_check_id(snapshot, task, application_id) for application_id in _gold_ids(task)]
        checked.append({"task_id": task["id"], "split": task["_split"], "ids": ids})

    by_split = {}
    for split in ("test", "train"):
        by_split[split] = _summarize([row for row in checked if row["split"] == split])
    report = {
        "snapshot": args.snapshot,
        "visible_chars": VISIBLE_CHARS,
        "splits": by_split,
        "fg_0002": _fg0002(snapshot, tasks),
        "misses": [
            {"task_id": row["task_id"], "split": row["split"], "ids": row["ids"]}
            for row in checked
            if not all(item["reachable"] and item["id_visible"] and item["answer_visible"] for item in row["ids"])
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"splits": by_split, "fg_0002": report["fg_0002"], "misses": len(report["misses"])}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
