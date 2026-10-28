"""The TRL 0.21 log path feeds reward metrics to memorization and the parse guard.

Shells out so the default suite does not import torch or TRL. Skips when the
pinned interpreter is missing.
"""

from __future__ import annotations

import subprocess

import pytest

from tests.train.test_trl_grpo_smoke import ROOT, _train_python

SCRIPT = ROOT / "tests" / "train" / "trl_grpo_log_flow.py"


def test_trl_021_log_feeds_metrics_to_memorization_and_the_parse_guard():
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
    assert "memorization=1" in proc.stdout
    assert "parse_stop=1" in proc.stdout
