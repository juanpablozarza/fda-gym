"""Scoring compare and sponsor_hop must not import fdagym.train."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SCORE_SCRIPT = r"""
import json
import sys
from pathlib import Path

from fdagym.rewards import score

rows = []
seen = set()
for line in Path("data/tasks/public.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    task = json.loads(line)
    kind = task.get("type")
    if kind in {"compare", "sponsor_hop"} and kind not in seen:
        rows.append(task)
        seen.add(kind)
    if len(rows) == 2:
        break
assert seen == {"compare", "sponsor_hop"}, seen
for task in rows:
    score({"steps": [], "final_answer": None}, task)
train = [name for name in sys.modules if name == "fdagym.train" or name.startswith("fdagym.train.")]
assert train == [], train
"""


def test_scoring_compare_and_sponsor_hop_does_not_import_train():
    result = subprocess.run(
        [sys.executable, "-c", _SCORE_SCRIPT],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
