"""Public FDA-Gym wrapper pieces that do not import ``verifiers``.

The v1 Taskset lives in ``fdagym.prime.taskset`` and is lazy-exported from
``fdagym`` so ``import fdagym`` stays extra-free.
"""

from fdagym.prime.episode import (
    ADAPTER_SYSTEM_PROMPT,
    ADAPTER_TOOLS,
    DEFAULT_TASK_ID,
    TOOL_BUDGET,
    DemoError,
    grade_episode,
    make_public_env,
    play_public_adapter_episode,
    step_tool,
)

__all__ = [
    "ADAPTER_SYSTEM_PROMPT",
    "ADAPTER_TOOLS",
    "DEFAULT_TASK_ID",
    "TOOL_BUDGET",
    "DemoError",
    "grade_episode",
    "make_public_env",
    "play_public_adapter_episode",
    "step_tool",
]
