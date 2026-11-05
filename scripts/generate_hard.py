"""Generate the hard training or published public task splits from the snapshot.

    uv run python scripts/generate_hard.py train
    uv run python scripts/generate_hard.py public

Both commands are CPU-only and leave the core train/test task files untouched.
``train`` writes ``hard_train.jsonl`` and ``hard_train_holdout.jsonl`` from drugs that
do not appear on ``public.jsonl`` or ``test.jsonl``. Owner drugs of cited applications
are excluded too. The holdout is a ~10% in-distribution question carve, not a disjoint eval.
``public`` writes ``data/tasks/public.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "public"))
    return _generate(parser.parse_args(argv).split)


def _generate(split: str) -> int:
    from fdagym.openfda import DiskSnapshot

    snapshot = DiskSnapshot("2026-10-15")
    if split == "public":
        from fdagym.agents.baseline import load_split
        from fdagym.tasks.disjoint import OverlapError, assert_disjoint
        from fdagym.tasks.hard import PUBLIC_ID_START, PUBLIC_SEED, generate_hard

        held_out = load_split("hard")
        tasks, report = generate_hard(
            snapshot,
            seed=PUBLIC_SEED,
            id_start=PUBLIC_ID_START,
            split="public",
            holdout_tasks=held_out,
            hold_back=0,
        )
        try:
            report["disjoint"] = assert_disjoint(tasks, held_out, snapshot)
        except (OverlapError, RuntimeError) as exc:
            report["disjoint_error"] = str(exc)
            _write_json(ROOT / "results" / "public-generation.json", report)
            print(str(exc), file=sys.stderr)
            return 1
        output = ROOT / "data" / "tasks" / "public.jsonl"
        report_path = ROOT / "results" / "public-generation.json"
        summary = {key: report[key] for key in ("seed", "n", "by_type", "by_variant", "pool") if key in report}
        outputs = [(output, tasks)]
    else:
        from fdagym.tasks.disjoint import OverlapError, assert_disjoint
        from fdagym.tasks.hard_train import (
            carve_clean_holdout,
            generate_hard_train,
            load_exclusion_splits,
            strip_clear_leaks,
        )
        from fdagym.tasks.holdout import assert_question_holdout
        from fdagym.tasks.plan import load_universe

        generated, report = generate_hard_train(snapshot)
        n_before = len(generated)
        generated = strip_clear_leaks(snapshot, generated)
        report["stripped_leaks"] = n_before - len(generated)
        tasks, question_holdout = carve_clean_holdout(snapshot, generated)
        report["stripped_after_carve"] = n_before - report["stripped_leaks"] - len(tasks) - len(question_holdout)
        report["question_holdout"] = assert_question_holdout(tasks, question_holdout, snapshot)
        outputs = [
            (ROOT / "data" / "tasks" / "hard_train.jsonl", tasks),
            (ROOT / "data" / "tasks" / "hard_train_holdout.jsonl", question_holdout),
        ]
        report_path = ROOT / "results" / "hard-train-generation.json"
        try:
            held_out = load_exclusion_splits()
            report["disjoint"] = assert_disjoint(tasks, held_out, snapshot)
            report["holdout_disjoint"] = assert_disjoint(question_holdout, held_out, snapshot)
        except (OverlapError, RuntimeError) as exc:
            report["disjoint_error"] = str(exc)
            _write_json(report_path, report)
            print(str(exc), file=sys.stderr)
            return 1
        summary = {
            key: report[key]
            for key in ("n", "by_type", "by_variant", "min_calls", "pool")
        }
        summary["universe"] = len(load_universe())

    for path, rows in outputs:
        _write_jsonl(path, rows)
    _write_json(report_path, report)
    print(json.dumps(summary, indent=2))
    for path, rows in outputs:
        print(f"wrote {len(rows)} tasks to {path}")
    return 0 if all(rows for _, rows in outputs) else 1


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
