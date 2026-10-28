"""Pinned GPU stack. ``pyproject.toml`` ``[project.optional-dependencies] gpu`` is the list.

Modal's image and ``scripts/runpod_setup.sh`` both install it. Importing this
module does not import torch, vLLM, or Modal.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The laptop agent uses litellm. The GPU image never imports it, and the
# historical Modal pip list left it out. Keep the image on that list.
_IMAGE_SKIP = frozenset({"litellm"})


def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _distribution(requirement: str) -> str:
    name = requirement.split(";", 1)[0].strip()
    for marker in (">", "<", "=", "!", "[", " "):
        name = name.split(marker, 1)[0]
    return name.strip().lower()


def gpu_requirements() -> list[str]:
    """Exact pins from the ``[gpu]`` extra, in file order."""
    return list(pyproject()["project"]["optional-dependencies"]["gpu"])


def modal_pip() -> list[str]:
    """Packages ``modal_train.py`` installs: the GPU pins, then the runtime deps.

    Runtime deps are the project dependencies except ``litellm``. That matches
    the pip list the image used before this extra existed.
    """
    project = pyproject()["project"]
    runtime = [dep for dep in project["dependencies"] if _distribution(dep) not in _IMAGE_SKIP]
    return [*project["optional-dependencies"]["gpu"], *runtime]
