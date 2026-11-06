"""FDA-Gym: a tool-use RL environment over a frozen openFDA snapshot."""

from __future__ import annotations

from typing import Any


def load_env() -> None:
    """Load a cwd ``.env`` if present. Call from scripts; importing ``fdagym`` does not."""
    from dotenv import load_dotenv

    load_dotenv()


def __getattr__(name: str) -> Any:
    if name == "FDAGymTaskset":
        from fdagym.prime.taskset import FDAGymTaskset

        return FDAGymTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FDAGymTaskset", "load_env"]  # pyright: ignore[reportUnsupportedDunderAll]
