"""The audit sheet JP marks up before a task type is allowed into training."""

import json
import random

from fdagym.tasks.config import GenConfig

TYPES = (
    "approval",
    "boxed_warning",
    "faers_top_reaction",
    "faers_count",
    "label_open",
    "composite",
)
TIERS = ("easy", "medium", "hard")


def render_review(tasks: list[dict], cfg: GenConfig | None = None, per_cell: int = 10) -> str:
    cfg = cfg or GenConfig()
    rng = random.Random(cfg.seed)
    lines = [
        "# Task review sheet",
        "",
        "Up to 10 tasks per tier and type. Mark a row bad when the gold does not follow from the citations.",
        "A type with more than 2 bad rows out of 10 gets fixed before training.",
        "",
    ]
    for tier_name in TIERS:
        for task_type in TYPES:
            pool = [task for task in tasks if task["difficulty"] == tier_name and task["type"] == task_type]
            sample = pool[:]
            rng.shuffle(sample)
            sample = sample[:per_cell]
            lines.append(f"## {tier_name} · {task_type} ({len(sample)} of {len(pool)})")
            lines.append("")
            if not sample:
                lines.append("_none_")
                lines.append("")
                continue
            for task in sample:
                name = task["entity"]["generic_name"]
                lines.append(f"### {task['id']} · {task['split']} · {name}")
                lines.append("")
                lines.append(task["question"])
                lines.append("")
                lines.append(f"- answerable: `{str(task['answerable']).lower()}`")
                lines.append(f"- gold: `{json.dumps(task['gold'], ensure_ascii=False)}`")
                lines.append(f"- citations: `{json.dumps(task['gold_citations'], ensure_ascii=False)}`")
                lines.append(f"- features: `{json.dumps(task['difficulty_features'])}`")
                lines.append(f"- tier: `{task['difficulty']}`")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"
