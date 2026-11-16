"""CI lock: grader, data, Hub package, README text, and the git pin stay in agreement."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import fdagym
from fdagym.openfda.store import SNAPSHOT_VERSION
from fdagym.rewards.version import (
    DATA_VERSION,
    GRADER_VERSION,
    HUB_RELEASES,
    LEGACY_DATA_VERSION,
    LEGACY_GRADER_VERSION,
    SNAPSHOT_FREEZE,
)

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "environments" / "fda-gym"
GIT_DEP = re.compile(r"fdagym @ git\+https://github.com/juanpablozarza/fda-gym\.git(?:@([^\s\"]+))?")
SHA = re.compile(r"^[0-9a-f]{40}$")
SERIES = re.compile(r"^(\d+\.\d+)")


def _hub_pyproject() -> str:
    return (HUB / "pyproject.toml").read_text()


def _hub_version(text: str) -> str:
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match, "Hub pyproject.toml has no version"
    return match.group(1)


def _hub_series(version: str) -> str:
    match = SERIES.match(version)
    assert match, version
    return match.group(1)


def _pin_ref(text: str) -> str:
    match = GIT_DEP.search(text)
    assert match, "Hub pyproject.toml has no fdagym git+ dependency"
    ref = match.group(1)
    assert ref, "Hub fdagym git+ URL is floating; pin a SHA or tag grader-v6"
    assert SHA.fullmatch(ref) or ref == "grader-v6", ref
    return ref


def test_package_exports_match_the_reward_constants():
    assert fdagym.grader_version == fdagym.GRADER_VERSION == GRADER_VERSION == "6"
    assert fdagym.data_version == fdagym.DATA_VERSION == DATA_VERSION == "2026-11-16"
    assert SNAPSHOT_FREEZE == SNAPSHOT_VERSION == "2026-10-15"


def test_hub_series_maps_to_this_tree():
    text = _hub_pyproject()
    version = _hub_version(text)
    series = _hub_series(version)
    assert version.startswith("0.2."), version
    assert series == "0.2"
    current = HUB_RELEASES[series]
    assert current["grader_version"] == GRADER_VERSION
    assert current["data_version"] == DATA_VERSION
    assert current["snapshot_version"] == SNAPSHOT_FREEZE
    legacy = HUB_RELEASES["0.1"]
    assert legacy["grader_version"] == LEGACY_GRADER_VERSION == "5"
    assert legacy["data_version"] == LEGACY_DATA_VERSION == "2026-10-28"
    assert legacy["snapshot_version"] == SNAPSHOT_FREEZE


def test_hub_description_names_grader_data_and_package():
    text = _hub_pyproject()
    description = re.search(r'(?m)^description = "([^"]+)"', text)
    assert description, text
    blob = description.group(1)
    assert "0.2.x" in blob
    assert "grader v6" in blob
    assert "data 2026-11-16" in blob
    assert "freeze 2026-10-15" in blob


def test_readme_and_hub_readme_carry_the_release_table():
    mapping = HUB_RELEASES["0.2"]
    needles = [
        "0.1.x",
        "0.2.x",
        f"| 0.1.x | {HUB_RELEASES['0.1']['grader_version']} | {HUB_RELEASES['0.1']['data_version']} | {HUB_RELEASES['0.1']['snapshot_version']} |",
        f"| 0.2.x | {mapping['grader_version']} | {mapping['data_version']} | {mapping['snapshot_version']} |",
        "Current grader: v6",
        "Data version: 2026-11-16",
    ]
    root_readme = (ROOT / "README.md").read_text()
    for needle in needles:
        assert needle in root_readme, needle
    hub_readme = (HUB / "README.md").read_text()
    for needle in needles[:4]:
        assert needle in hub_readme, needle
    assert "This Hub package is `0.2.x`" in hub_readme
    assert "Grader v6" in hub_readme
    assert "Data version `2026-11-16`" in hub_readme


def _ensure_commit(ref: str) -> None:
    """GitHub Actions checks out the PR merge at depth 1, so the pin SHA is often absent."""
    probed = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{ref}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probed.returncode == 0:
        return
    fetched = subprocess.run(
        ["git", "-C", str(ROOT), "fetch", "--depth=1", "origin", ref],
        check=False,
        capture_output=True,
        text=True,
    )
    assert fetched.returncode == 0, (
        f"pinned SHA {ref} is not in this clone and could not be fetched: {fetched.stderr.strip()}"
    )


def test_pinned_fdagym_ref_is_immutable_and_matches_this_grader():
    text = _hub_pyproject()
    ref = _pin_ref(text)
    if SHA.fullmatch(ref):
        _ensure_commit(ref)
        shown = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{ref}:fdagym/rewards/version.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        tree = ast.parse(shown.stdout)
        pinned: dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"GRADER_VERSION", "DATA_VERSION"}:
                assert isinstance(node.value, ast.Constant)
                pinned[target.id] = str(node.value.value)
        assert pinned["GRADER_VERSION"] == GRADER_VERSION
        assert pinned["DATA_VERSION"] == DATA_VERSION
    else:
        assert ref == "grader-v6"


def test_v5_openrouter_artifacts_are_labeled_without_rescoring():
    summary = (ROOT / "results" / "public-haiku-luna-2026-11-05" / "summary.json").read_text()
    assert '"grader_version": "5"' in summary
    assert '"data_version": "2026-10-28"' in summary
    assert '"mean_reward": 0.5817' in summary
    assert '"mean_reward": 0.4795' in summary
    board = (ROOT / "results" / "public-haiku-luna-2026-11-05" / "leaderboard.json").read_text()
    assert '"grader_version": "5"' in board
    assert '"data_version": "2026-10-28"' in board
    assert '"mean_reward": 0.5817' in board
