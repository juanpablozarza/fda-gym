"""Grader/data version helpers: missing is legacy v5, mixed graders are refused."""

from __future__ import annotations

import json

import pytest

from fdagym.env import FDAEnv
from fdagym.rewards.version import DATA_VERSION, GRADER_VERSION
from fdagym.run_versions import (
    MixedGraderVersionError,
    read_data_version,
    read_grader_version,
    require_uniform_grader,
    versions_of,
)
from fdagym.train.compare import compare_summaries
from tests.env.support import ALL_TASKS, JsonSnapshot


def test_missing_grader_version_is_legacy_v5_with_a_warning():
    with pytest.warns(UserWarning, match="no grader_version"):
        assert read_grader_version(None, source="row") == "5"
    with pytest.warns(UserWarning, match="no data_version"):
        assert read_data_version("", source="row") == "2026-10-28"


def test_mixed_grader_versions_are_refused():
    with pytest.raises(MixedGraderVersionError, match="grader versions"):
        require_uniform_grader(["5", "6"], action="summarize")
    assert require_uniform_grader(["6", "6"], action="summarize") == "6"


def test_versions_of_empty_stamps_the_live_grader():
    assert versions_of([], source="empty") == (GRADER_VERSION, DATA_VERSION)


def test_versions_of_reads_reward_or_flat_dicts():
    rows = [
        {"reward": {"grader_version": "6", "data_version": "2026-11-16"}},
        {"grader_version": "6", "data_version": "2026-11-16"},
    ]
    assert versions_of(rows, source="rows") == ("6", "2026-11-16")


def test_compare_summaries_refuses_mixed_graders():
    baseline = {
        "run_id": "v5",
        "grader_version": "5",
        "models": [{"name": "3b", "mean_reward": 0.21, "pass_rate": 0.12}],
    }
    trained = {
        "run_id": "v6",
        "grader_version": "6",
        "models": [{"name": "3b+lora", "mean_reward": 0.34, "pass_rate": 0.22}],
    }
    with pytest.raises(MixedGraderVersionError, match="compare summaries"):
        compare_summaries(baseline, trained)


def test_compare_summaries_warns_when_versions_are_missing():
    baseline = {
        "run_id": "baseline-qwen3b",
        "models": [{"name": "3b", "mean_reward": 0.21, "pass_rate": 0.12}],
    }
    trained = {
        "run_id": "grpo-qwen3b-eval",
        "models": [{"name": "3b+lora", "mean_reward": 0.34, "pass_rate": 0.22}],
    }
    with pytest.warns(UserWarning, match="no grader_version"):
        row = compare_summaries(baseline, trained)
    assert row["beats_baseline"] is True
    assert row["grader_version"] == "5"


def test_collect_curve_refuses_mixed_grader_summaries(tmp_path):
    from fdagym.train.distill_curve import collect_curve

    prefix = "mix"
    for size, grader in (("3", "5"), ("5", "6")):
        run_id = f"{prefix}-n{size}-hard"
        directory = tmp_path / run_id
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "grader_version": grader,
                    "models": {"x": {"n": 1, "mean_reward": 0.2, "pass_rate": 0.0}},
                    "leaderboard": [{"name": "x", "mean_reward": 0.2}],
                }
            )
        )
    with pytest.raises(MixedGraderVersionError, match="distill curve"):
        collect_curve(tmp_path, prefix, "3,5", split="hard")


def test_env_metadata_and_terminal_info_carry_versions():
    env = FDAEnv(JsonSnapshot(), ALL_TASKS)
    assert env.metadata["grader_version"] == GRADER_VERSION
    assert env.metadata["data_version"] == DATA_VERSION
    env.reset("FG-0001")
    _obs, _reward, done, info = env.step({"tool": "submit_answer", "arguments": {"answer": "unknown", "citations": []}})
    assert done is True
    assert info["grader_version"] == GRADER_VERSION
    assert info["data_version"] == DATA_VERSION
    assert env.trajectory["reward"]["grader_version"] == GRADER_VERSION
    assert env.trajectory["reward"]["data_version"] == DATA_VERSION
