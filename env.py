"""HUD SDK Environment entrypoint for public FDA-Gym.

Local eval (no GPU, no Docker)::

    uv sync --extra hud
    uv run hud eval tasks.py claude
    uv run hud eval tasks.py claude --all

``Dockerfile.hud`` is optional and only for ``hud deploy``.
"""

from __future__ import annotations

from fdagym.hud.sdk import load_hud_environment, mint_public_tasks

env, public_task = load_hud_environment()

__all__ = ["env", "mint_public_tasks", "public_task"]
