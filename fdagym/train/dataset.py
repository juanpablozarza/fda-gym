"""Build the GRPO dataset: chat prompts with oracle docs plus distractors."""

from fdagym.snapshot import Snapshot
from fdagym.train.config import TrainConfig
from fdagym.train.context import render_task_prompt
from fdagym.train.curriculum import stage_pool


def build_rows(
    snapshot: Snapshot,
    tasks: list[dict],
    config: TrainConfig,
) -> tuple[list[dict], dict[str, int]]:
    selected, mix = stage_pool(tasks, config.stage)
    rows = []
    for task in selected:
        rows.append(
            {
                "prompt": render_task_prompt(
                    snapshot,
                    tasks,
                    task,
                    n_distractors=config.n_distractors,
                    max_chars=config.max_pack_chars,
                ),
                "task_id": task["id"],
            }
        )
    return rows, mix
