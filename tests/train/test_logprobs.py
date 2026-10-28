"""Chunked GRPO log-probs match the unchunked matmul.

The parent process does not import torch. ``test_tool_modules_import_without_torch``
asserts torch stays out of ``sys.modules`` for this interpreter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent / "logprob_equiv.py"


def test_chunked_logprobs_match_the_unchunked_matmul():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if proc.returncode == 2:
        pytest.skip(proc.stdout.strip() or "torch is not installed")
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "PASS" in proc.stdout
