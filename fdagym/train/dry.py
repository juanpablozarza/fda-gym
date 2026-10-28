"""Offline sanity check: gold scores 1.0, junk scores 0, equal rewards have zero advantage."""

from fdagym.env import FDAEnv
from fdagym.snapshot import Snapshot
from fdagym.train.advantage import advantages
from fdagym.train.config import TrainConfig
from fdagym.train.context import play_oracle
from fdagym.train.curriculum import stage_pool
from fdagym.train.gold import gold_submit
from fdagym.train.prompt_len import fit_context
from fdagym.train.score import score_completion


def run_dry(snapshot: Snapshot, tasks: list[dict], config: TrainConfig | None = None) -> dict:
    config = config or TrainConfig()
    pool, mix = stage_pool(tasks, config.stage)
    if not pool:
        raise RuntimeError("stage pool is empty")
    task = pool[0]
    env = FDAEnv(snapshot, tasks, run_id="dry")
    steps = play_oracle(env, task)
    gold = gold_submit(task, steps)
    gold_reward = score_completion(snapshot, tasks, task, json_dumps(gold))
    junk_reward = score_completion(snapshot, tasks, task, "I guess NAUSEA")
    format_only = score_completion(
        snapshot,
        tasks,
        task,
        json_dumps({"answer": "NOT_FOUND", "citations": []}),
    )
    equal = advantages([1.0, 1.0, 1.0, 1.0])
    mixed = advantages([1.0, 0.1, 0.1, 0.1])
    from fdagym.train.dataset import build_rows

    rows, _mix = build_rows(snapshot, tasks, config)
    fitted, length = fit_context(config, rows)
    return {
        "task_id": task["id"],
        "gold": gold_reward,
        "junk": junk_reward,
        "format_only": format_only,
        "advantages_equal": equal,
        "advantages_mixed": mixed,
        "stage": config.stage,
        "mix": mix,
        "n_prompts": len(rows),
        "prompt_preview": rows[0]["prompt"][1]["content"][:400] if rows else "",
        "prompt_length": length,
        "vllm_max_model_len": fitted.vllm_max_model_len,
        "max_pack_chars": fitted.max_pack_chars,
    }


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def format_report(report: dict) -> str:
    gold = report["gold"]
    junk = report["junk"]
    fmt = report["format_only"]
    mixed = " ".join(f"{v:.3f}" for v in report["advantages_mixed"])
    equal = " ".join(str(v) for v in report["advantages_equal"])
    length = report.get("prompt_length") or {}
    return "\n".join(
        [
            f"stage {report['stage']} n_prompts={report['n_prompts']} mix={report['mix']}",
            (
                f"gold {report['task_id']} total={gold['total']:.4f} correctness={gold['correctness']:.4f} "
                f"citation={gold['citation']:.4f} format={gold['format']:.4f} efficiency={gold['efficiency']:.4f}"
            ),
            f"junk {report['task_id']} total={junk['total']:.4f}",
            f"format-only {report['task_id']} total={fmt['total']:.4f} (NOT_FOUND, no citations)",
            f"advantages equal=[{equal}]",
            f"advantages mixed=[{mixed}]",
            (
                f"prompt tokens measured={length.get('measured_prompt_tokens')} "
                f"vllm_max_model_len={length.get('vllm_max_model_len')} "
                f"max_pack_chars={length.get('max_pack_chars')} "
                f"fits={length.get('fits')}"
            ),
        ]
    )
