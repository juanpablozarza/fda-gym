"""Refuse mixed grader versions when comparing or aggregating runs.

Distill curve code imports this module instead of ``fdagym.rewards`` so a
learning-curve job does not load the grader.
"""

from __future__ import annotations

import warnings

from fdagym.rewards.version import DATA_VERSION, GRADER_VERSION, LEGACY_DATA_VERSION, LEGACY_GRADER_VERSION


class MixedGraderVersionError(ValueError):
    """Raised when a comparison or aggregate mixes grader versions."""


def read_grader_version(value: object, *, source: str = "artifact") -> str:
    """Return a grader version. A missing value is legacy v5, with a warning."""
    if value is None or value == "":
        warnings.warn(
            f"{source} has no grader_version; treating as legacy grader v{LEGACY_GRADER_VERSION}",
            UserWarning,
            stacklevel=2,
        )
        return LEGACY_GRADER_VERSION
    return str(value)


def read_data_version(value: object, *, source: str = "artifact") -> str:
    """Return a data version. A missing value is the v5 sheet, with a warning."""
    if value is None or value == "":
        warnings.warn(
            f"{source} has no data_version; treating as legacy data {LEGACY_DATA_VERSION}",
            UserWarning,
            stacklevel=2,
        )
        return LEGACY_DATA_VERSION
    return str(value)


def require_uniform_grader(versions: list[str], *, action: str) -> str:
    """Return the single grader version, or raise if the inputs disagree."""
    unique = sorted(set(versions))
    if len(unique) > 1:
        raise MixedGraderVersionError(
            f"cannot {action} across grader versions {unique}; scores from different graders are not comparable"
        )
    if not unique:
        return GRADER_VERSION
    return unique[0]


def versions_of(rows: list[dict], *, source: str) -> tuple[str, str]:
    """Return ``(grader_version, data_version)`` for trajectories or reward dicts.

    An empty list stamps the live grader. Mixed grader versions raise.
    Mixed data versions warn and keep the first seen.
    """
    if not rows:
        return GRADER_VERSION, DATA_VERSION
    graders: list[str] = []
    datas: list[str] = []
    for index, row in enumerate(rows):
        reward = row["reward"] if isinstance(row.get("reward"), dict) else row
        label = f"{source}[{index}]"
        graders.append(read_grader_version(reward.get("grader_version"), source=label))
        datas.append(read_data_version(reward.get("data_version"), source=label))
    grader = require_uniform_grader(graders, action=source)
    seen: list[str] = []
    for item in datas:
        if item not in seen:
            seen.append(item)
    data = seen[0]
    if len(seen) > 1:
        warnings.warn(
            f"{source} mixes data_version values {seen}; using {data}",
            UserWarning,
            stacklevel=2,
        )
    return grader, data
