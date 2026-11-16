"""Versions stamped on every score() result.

``snapshot_version`` on a task is the openFDA freeze the tools read. It stays
``2026-10-15`` until a new recording. ``DATA_VERSION`` is the task sheet this
grader was written against. Bump it when questions or gold change and the freeze
does not. ``GRADER_VERSION`` bumps when ``score()`` changes what a trace is worth.
A freeze bump does not by itself change either version.

Hub package ``fda-gym`` 0.1.x is grader 5 / data 2026-10-28 / freeze 2026-10-15.
0.2.x is this tree: grader 6 / data 2026-11-16 / freeze 2026-10-15.
"""

GRADER_VERSION = "6"
DATA_VERSION = "2026-11-16"
LEGACY_GRADER_VERSION = "5"
LEGACY_DATA_VERSION = "2026-10-28"
SNAPSHOT_FREEZE = "2026-10-15"

# Hub package series → grader / data / freeze. Keep in lockstep with
# environments/fda-gym/pyproject.toml, README.md, and environments/fda-gym/README.md.
HUB_RELEASES = {
    "0.1": {
        "grader_version": LEGACY_GRADER_VERSION,
        "data_version": LEGACY_DATA_VERSION,
        "snapshot_version": SNAPSHOT_FREEZE,
    },
    "0.2": {
        "grader_version": GRADER_VERSION,
        "data_version": DATA_VERSION,
        "snapshot_version": SNAPSHOT_FREEZE,
    },
}
