"""Pinned GPU stack. ``pyproject.toml`` ``[project.optional-dependencies] gpu`` is the list.

Install with ``uv sync --extra gpu``. Importing this module does not import torch or vLLM.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def gpu_requirements() -> list[str]:
    """Exact pins from the ``[gpu]`` extra, in file order."""
    return list(pyproject()["project"]["optional-dependencies"]["gpu"])
