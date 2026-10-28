"""Versions stamped on every score() result.

``snapshot_version`` on a task is the openFDA freeze the tools read. It stays
``2026-10-15`` until a new recording. ``DATA_VERSION`` is the task sheet this
grader was written against. Bump it when questions or gold change and the freeze
does not. ``GRADER_VERSION`` bumps when ``score()`` changes what a trace is worth.
A freeze bump does not by itself change either version.
"""

GRADER_VERSION = "4"
DATA_VERSION = "2026-10-28"
