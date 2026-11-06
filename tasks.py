"""Concrete HUD tasks: one public.jsonl row per template call (FG-9101–FG-9193)."""

from __future__ import annotations

from env import env, mint_public_tasks

tasks = mint_public_tasks()

__all__ = ["env", "tasks"]
