"""JSON Schema contracts for answers, tasks, and trajectories."""

from __future__ import annotations

import json
from importlib.resources import files


def load(name: str) -> dict:
    """Load a packaged schema or example, e.g. ``answer.schema.json`` or ``examples/task.example.json``."""
    return json.loads(files("fdagym.schemas").joinpath(name).read_text(encoding="utf-8"))
