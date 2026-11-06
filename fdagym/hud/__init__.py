"""HUD SDK adapter for the published FDA-Gym sheet.

The in-process episode wrapper lives here and does not import the HUD SDK.
``uv sync --extra hud`` is only required to serve ``env.py`` / ``tasks.py``.
"""

from fdagym.hud.session import PublicEpisode, public_task_ids

__all__ = ["PublicEpisode", "public_task_ids"]
