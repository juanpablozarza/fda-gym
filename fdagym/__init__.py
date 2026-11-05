"""FDA-Gym: a tool-use RL environment over a frozen openFDA snapshot."""

from __future__ import annotations


def load_env() -> None:
    """Load a cwd ``.env`` if present. Call from scripts; importing ``fdagym`` does not."""
    from dotenv import load_dotenv

    load_dotenv()
