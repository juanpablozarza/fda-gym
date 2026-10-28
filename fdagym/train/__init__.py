"""Plan A GRPO: prefetch oracle_calls, score the completion through FDAEnv."""

from fdagym.train.advantage import advantages
from fdagym.train.checkpoints import latest_checkpoint
from fdagym.train.compare import compare_summaries
from fdagym.train.config import QWEN_3B, QWEN_7B, TrainConfig, select_grpo_kwargs
from fdagym.train.curriculum import stage_pool
from fdagym.train.parse import parse_answer
from fdagym.train.score import run_completion, score_completion

__all__ = [
    "QWEN_3B",
    "QWEN_7B",
    "TrainConfig",
    "advantages",
    "compare_summaries",
    "latest_checkpoint",
    "parse_answer",
    "run_completion",
    "score_completion",
    "select_grpo_kwargs",
    "stage_pool",
]
