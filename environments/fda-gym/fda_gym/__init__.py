"""Prime Environments Hub package for public FDA-Gym (verifiers v1).

Re-exports ``FDAGymTaskset``. The Hub wheel depends on ``verifiers==0.4.0``.
"""

from fdagym.prime.taskset import FDAGymTaskset

__all__ = ["FDAGymTaskset"]
