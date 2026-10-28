"""Portable run metadata and compatibility checks."""

import json
import subprocess
import sys

import pytest

from fdagym.train.run_record import RunConflictError, finish_run, start_run


def test_run_record_tracks_config_executor_revision_and_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("FDAGYM_EXECUTOR", "runpod")
    monkeypatch.setenv("FDAGYM_GIT_SHA", "abc123")
    run_dir = start_run(
        tmp_path / "runs",
        "grpo-tools",
        job="tool-train",
        protocol="tool_rollout",
        config={"tool_budget": 8, "prompt_version": "v4"},
        resume=False,
    )
    record = json.loads((run_dir / "run.json").read_text())
    assert record["record_version"] == 1
    assert record["status"] == "running"
    assert record["source_revision"] == "abc123"
    assert record["environment"]["executor"] == "runpod"
    assert record["config"] == {"prompt_version": "v4", "tool_budget": 8}
    assert record["attempts"][0]["status"] == "running"

    finish_run(run_dir, "done")
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "done"
    assert record["attempts"][0]["finished_at"]

    monkeypatch.setenv("FDAGYM_EXECUTOR", "modal")
    start_run(
        tmp_path / "runs",
        "grpo-tools",
        job="tool-train",
        protocol="tool_rollout",
        config={"tool_budget": 8, "prompt_version": "v4"},
        resume=True,
    )
    record = json.loads((run_dir / "run.json").read_text())
    assert record["environment"]["executor"] == "modal"
    assert [attempt["environment"]["executor"] for attempt in record["attempts"]] == ["runpod", "modal"]
    assert not (run_dir / "DONE").exists()
    assert not (run_dir / "FAILED").exists()


def test_resuming_with_a_different_config_preserves_the_existing_run(tmp_path):
    run_dir = start_run(
        tmp_path,
        "same-run",
        job="tool-train",
        protocol="tool_rollout",
        config={"max_steps": 150},
    )
    finish_run(run_dir, "done")
    before = (run_dir / "run.json").read_text()
    (run_dir / "DONE").write_text("done\n")

    with pytest.raises(RunConflictError, match="config differs"):
        start_run(
            tmp_path,
            "same-run",
            job="tool-train",
            protocol="tool_rollout",
            config={"max_steps": 60},
            resume=True,
        )

    assert (run_dir / "run.json").read_text() == before
    assert (run_dir / "DONE").is_file()


def test_run_id_must_be_one_path_segment(tmp_path):
    for run_id in ("", "../escape", "nested/run"):
        with pytest.raises(ValueError, match="one path segment"):
            start_run(
                tmp_path,
                run_id,
                job="tool-eval",
                protocol="tool_rollout",
                config={},
            )


def test_active_run_rejects_a_second_process_until_finish_releases_lock(tmp_path):
    run_dir = start_run(
        tmp_path,
        "same-run",
        job="tool-train",
        protocol="tool_rollout",
        config={"max_steps": 150},
    )
    code = f"""
from fdagym.train.run_record import RunConflictError, start_run
try:
    start_run({str(tmp_path)!r}, 'same-run', job='tool-train', protocol='tool_rollout', config={{}})
except RunConflictError as exc:
    print(exc)
else:
    raise SystemExit('second writer unexpectedly acquired the run')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "already has an active writer" in result.stdout

    finish_run(run_dir, "done")
    resumed = start_run(
        tmp_path,
        "same-run",
        job="tool-train",
        protocol="tool_rollout",
        config={"max_steps": 150},
        resume=True,
    )
    record = json.loads((resumed / "run.json").read_text())
    assert [attempt["status"] for attempt in record["attempts"]] == ["done", "running"]
    finish_run(resumed, "done")
