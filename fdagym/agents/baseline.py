"""Load a task split, resume a jsonl run, and write summary plus B1 leaderboard rows."""

import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from fdagym.agents.litellm_agent import LiteLLMAgent
from fdagym.env import FDAEnv
from fdagym.openfda import DiskSnapshot

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "data" / "tasks"
SCHEMAS = ROOT / "schemas"
SKIP_TYPES = frozenset({"label_open"})
COST_CAP_USD = 5.0


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


_TRAJECTORY_SCHEMA = _load_schema("trajectory.schema.json")
_REGISTRY = Registry().with_resources(
    (schema["$id"], Resource.from_contents(schema))
    for schema in (_TRAJECTORY_SCHEMA, _load_schema("answer.schema.json"))
)
_TRAJECTORY_VALIDATOR = Draft202012Validator(_TRAJECTORY_SCHEMA, registry=_REGISTRY)


def load_split(split: str) -> list[dict]:
    path = TASKS_DIR / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / f"{run_id}.jsonl"


def done_keys(path: Path) -> set[tuple[str, str, int]]:
    found: set[tuple[str, str, int]] = set()
    if not path.exists():
        return found
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found.add((row["model"]["name"], row["task_id"], int(row["seed"])))
    return found


def append_trajectory(path: Path, trajectory: dict) -> None:
    _TRAJECTORY_VALIDATOR.validate(trajectory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")


def append_failure(
    path: Path,
    *,
    model: str,
    task_id: str,
    seed: int,
    error: Exception | None = None,
    error_type: str | None = None,
) -> dict:
    """Persist an episode failure without provider error text that may contain secrets."""
    failure = {
        "model": model,
        "task_id": task_id,
        "seed": seed,
        "error_type": error_type or type(error).__name__,
        "failed_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    return failure


def read_trajectories(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_failures(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unresolved_failures(trajectories: list[dict], failures: list[dict]) -> list[dict]:
    """Keep the latest failure for each episode unless a later successful trajectory exists."""
    successful = {
        (row["model"]["name"], row["task_id"], int(row["seed"]))
        for row in trajectories
        if row["terminated_by"] != "error"
    }
    latest = {}
    for failure in failures:
        key = (failure["model"], failure["task_id"], int(failure["seed"]))
        if key not in successful:
            latest[key] = failure
    return list(latest.values())


def summarize(
    trajectories: list[dict],
    *,
    skipped: dict[str, int],
    failures: list[dict] | None = None,
    aborted: str | None = None,
) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        by_model[row["model"]["name"]].append(row)
    models = {}
    leaderboard = []
    for name, rows in sorted(by_model.items(), key=lambda item: -_mean(item[1], "total")):
        kind = rows[0]["model"]["kind"]
        n = len(rows)
        mean_reward = _mean(rows, "total")
        pass_rate = sum(1 for row in rows if row["reward"]["correctness"] > 0) / n if n else 0.0
        mean_citation = sum(row["reward"]["citation"] for row in rows) / n if n else 0.0
        mean_tools = sum(_tool_calls(row) for row in rows) / n if n else 0.0
        cost_usd = sum(row["usage"].get("cost_usd") or 0.0 for row in rows)
        models[name] = {
            "n": n,
            "mean_reward": round(mean_reward, 4),
            "pass_rate": round(pass_rate, 4),
            "mean_citation": round(mean_citation, 4),
            "mean_tools": round(mean_tools, 4),
            "cost_usd": round(cost_usd, 6),
        }
        leaderboard.append(
            {
                "name": name,
                "kind": kind,
                "mean_reward": round(mean_reward, 4),
                "pass": round(pass_rate, 4),
                "citations": round(mean_citation, 4),
                "tools": round(mean_tools, 4),
            }
        )
    failures = failures or []
    payload = {
        "status": "incomplete" if failures or aborted else "complete",
        "complete": not failures and not aborted,
        "failures": failures,
        "failure_count": len(failures),
        "skipped": skipped,
        "models": models,
        "leaderboard": leaderboard,
    }
    if aborted:
        payload["aborted"] = aborted
    return payload


def write_summaries(run_dir: Path, summary: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "leaderboard.json").write_text(json.dumps(summary["leaderboard"], indent=2) + "\n")


_SIDECAR_FIELDS = ("id", "question", "type", "sources", "difficulty", "entity")


def task_cards(tasks: list[dict]) -> list[dict]:
    """Dashboard join key is id. Include skipped types so label_open still has a card."""
    return [{key: task[key] for key in _SIDECAR_FIELDS} for task in tasks]


def write_task_sidecar(results_dir: Path, run_id: str, tasks: list[dict]) -> list[dict]:
    cards = task_cards(tasks)
    payload = json.dumps(cards, indent=2, ensure_ascii=False) + "\n"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks.json").write_text(payload)
    (results_dir / "tasks.json").write_text(payload)
    return cards


def run_baseline(
    *,
    split: str,
    models: list[str],
    seeds: list[int],
    run_id: str,
    results_dir: Path | None = None,
    completion=None,
    cost_cap: float = COST_CAP_USD,
    skip_types: frozenset[str] = SKIP_TYPES,
    snapshot_version: str | None = None,
    snapshot=None,
    tasks: list[dict] | None = None,
    pause_s: float = 0.0,
) -> dict:
    results_dir = results_dir or (ROOT / "results")
    tasks = tasks if tasks is not None else load_split(split)
    if not tasks:
        raise ValueError(f"task split {split!r} is empty")
    version = snapshot_version or tasks[0]["snapshot_version"]
    snapshot = snapshot or DiskSnapshot(version)
    runnable = [task for task in tasks if task["type"] not in skip_types]
    skipped: dict[str, int] = {name: sum(1 for task in tasks if task["type"] == name) for name in skip_types}
    skipped["already_done"] = 0
    env = FDAEnv(snapshot, tasks, run_id=run_id)
    out = jsonl_path(results_dir, run_id)
    run_dir = results_dir / run_id
    failures_path = run_dir / "failures.jsonl"
    already = done_keys(out)
    spent = sum(row["usage"].get("cost_usd") or 0.0 for row in read_trajectories(out))
    aborted = None
    for model in models:
        agent = LiteLLMAgent(model, completion=completion)
        for seed in seeds:
            env.seed = seed
            for task in runnable:
                key = (model, task["id"], seed)
                if key in already:
                    skipped["already_done"] += 1
                    continue
                if spent >= cost_cap:
                    aborted = f"cost cap ${cost_cap:.2f} exceeded (spent ${spent:.4f})"
                    break
                env.run_id = run_id
                try:
                    trajectory = agent.run(env, task["id"])
                except Exception as exc:  # noqa: BLE001 — keep the run going on a single model error
                    failure = append_failure(
                        failures_path, model=model, task_id=task["id"], seed=seed, error=exc
                    )
                    print(
                        f"FAIL {model} {task['id']} s{seed} {failure['error_type']} (recorded)",
                        flush=True,
                    )
                    continue
                trajectory["id"] = f"{run_id}/{model}/{task['id']}/s{seed}"
                trajectory["run_id"] = run_id
                append_trajectory(out, trajectory)
                already.add(key)
                spent += trajectory["usage"].get("cost_usd") or 0.0
                if trajectory["terminated_by"] == "error":
                    append_failure(
                        failures_path,
                        model=model,
                        task_id=task["id"],
                        seed=seed,
                        error_type="AgentEpisodeError",
                    )
                print(
                    f"{model} {task['id']} s{seed} R={trajectory['reward']['total']:.3f} spent=${spent:.4f}",
                    flush=True,
                )
                if pause_s:
                    time.sleep(pause_s)
            if aborted:
                break
        if aborted:
            break
    trajectories = read_trajectories(out)
    failures = unresolved_failures(trajectories, read_failures(failures_path))
    summary = summarize(trajectories, skipped=skipped, failures=failures, aborted=aborted)
    summary["run_id"] = run_id
    summary["split"] = split
    summary["seeds"] = seeds
    summary["cost_usd_total"] = round(spent, 6)
    write_summaries(run_dir, summary)
    write_task_sidecar(results_dir, run_id, tasks)
    return summary


def _mean(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(row["reward"][field] for row in rows) / len(rows)


def _tool_calls(row: dict) -> int:
    return sum(1 for step in row["steps"] if step.get("tool") != "submit_answer")
