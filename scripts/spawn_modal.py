"""Spawn one deployed tool-mode function and exit. Does not call ``modal run``.

    modal deploy modal_train.py
    python scripts/spawn_modal.py sft
    python scripts/spawn_modal.py eval-hard
    python scripts/spawn_modal.py grpo-dry

Run this from the directory that contains ``modal_train.py``. The image packs that
directory. See docs/RUNBOOK.md. This file does not start a job on import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdagym.train.spawn_plan import spawn_plan


def main(argv: list[str] | None = None) -> int:
    which = (argv if argv is not None else sys.argv[1:])
    if len(which) != 1:
        print("usage: python scripts/spawn_modal.py {sft,eval-hard,grpo-dry}", file=sys.stderr)
        return 2
    plan = spawn_plan(which[0])
    kwargs = dict(plan["kwargs"])
    if plan["function"] == "eval_tools":
        token = os.environ.get("HF_TOKEN", "")
        if token:
            kwargs["hf_token"] = token
    import modal

    fn = modal.Function.from_name(plan["app"], plan["function"])
    call = fn.spawn(**kwargs)
    print("spawned", plan["function"], call.object_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
