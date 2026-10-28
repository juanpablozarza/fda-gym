"""Run the TRL 0.21 GRPO smoke step in a pinned interpreter.

The default ``uv sync`` environment does not install torch or TRL. This test
shells out so importing it does not load either package (see
``test_tool_modules_import_without_torch``). It skips when no pinned interpreter
is available.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent / "trl_grpo_smoke.py"


def _train_python() -> str | None:
    override = os.environ.get("FDAGYM_TRAIN_PYTHON")
    if override:
        return override
    for candidate in (
        Path("/tmp/fdagym-trl021/bin/python"),
        ROOT / ".venv-train" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def test_trl_021_grpo_step_uses_real_logprobs():
    python = _train_python()
    if python is None:
        pytest.skip("trl==0.21.0 interpreter not found (set FDAGYM_TRAIN_PYTHON)")
    probe = subprocess.run(
        [python, "-c", "import trl; assert trl.__version__ == '0.21.0'"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"{python} does not have trl==0.21.0\n{probe.stderr}")
    proc = subprocess.run(
        [python, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "PASS" in proc.stdout
    assert "logprob_calls=" in proc.stdout
