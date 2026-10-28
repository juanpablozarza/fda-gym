"""Audit hard-task split integrity, tool reachability, and memorization risk.

    uv run python scripts/audit_hard.py disjointness
    uv run python scripts/audit_hard.py public
    uv run python scripts/audit_hard.py reachability --split hard --budget 6 --budget 8
    uv run python scripts/audit_hard.py memorization

These are local snapshot checks. None starts Modal or a GPU job.
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
    args = _parser().parse_args(argv)
    if args.audit == "disjointness":
        return _disjointness()
    if args.audit == "public":
        return _public()
    if args.audit == "reachability":
        return _reachability(args)
    return _memorization()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="audit", required=True)

    sub.add_parser("disjointness", help="check held-out drugs, applications, and sponsors")
    sub.add_parser("public", help="check the public sheet against the private hard sheet")

    reach = sub.add_parser("reachability", help="re-solve a split through the real tools")
    reach.add_argument("--split", default="hard", help="task file stem under data/tasks")
    reach.add_argument("--budget", dest="budgets", type=_positive, action="append", default=None)
    reach.add_argument("--out", type=Path, default=None, help="report path")

    sub.add_parser("memorization", help="score known-drug holdout and swapped facts")
    return parser


def _disjointness() -> int:
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.tasks.disjoint import OverlapError, assert_disjoint
    from fdagym.tasks.holdout import assert_question_holdout

    snapshot = DiskSnapshot("2026-10-15")
    train = load_split("hard_train")
    question_holdout = load_split("hard_train_holdout")
    held_out = [*load_split("hard"), *load_split("test")]
    try:
        report = assert_disjoint(train, held_out, snapshot)
        question = assert_disjoint(question_holdout, held_out, snapshot)
        known = assert_question_holdout(train, question_holdout, snapshot)
    except (OverlapError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        for kind, values in getattr(exc, "overlaps", {}).items():
            print(f"{kind}: {', '.join(values)}", file=sys.stderr)
        return 1

    report["question_holdout_vs_eval"] = {
        "train_tasks": question["train_tasks"],
        "overlap": question["overlap"],
        "train": question["train"],
    }
    report["known_drug_holdout"] = known
    path = ROOT / "results" / "hard-train-disjoint.json"
    _write_json(path, report)
    print(json.dumps(report, indent=2))
    return 0


def _public() -> int:
    """The published sheet must not reuse a drug, application, or sponsor from hard.jsonl."""
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.tasks.disjoint import OverlapError, assert_disjoint

    public_path = ROOT / "data" / "tasks" / "public.jsonl"
    hard_path = ROOT / "data" / "tasks" / "hard.jsonl"
    if not public_path.exists():
        print("data/tasks/public.jsonl is missing", file=sys.stderr)
        return 1
    if not hard_path.exists():
        print("data/tasks/hard.jsonl is not in this tree. Run this check in the private repo.", file=sys.stderr)
        return 1
    try:
        report = assert_disjoint(load_split("public"), load_split("hard"), DiskSnapshot("2026-10-15"))
    except (OverlapError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        for kind, values in getattr(exc, "overlaps", {}).items():
            print(f"{kind}: {', '.join(values)}", file=sys.stderr)
        return 1
    path = ROOT / "results" / "public-disjoint.json"
    _write_json(path, report)
    print(json.dumps(report, indent=2))
    return 0


def _reachability(args: argparse.Namespace) -> int:
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.tasks.reachability import check_reachability
    from fdagym.train.rollout import DEFAULT_TOOL_BUDGET

    budgets = args.budgets or [DEFAULT_TOOL_BUDGET]
    snapshot = DiskSnapshot("2026-10-15")
    tasks = load_split(args.split)
    report = check_reachability(snapshot, tasks, budgets, score=_score)
    if len(budgets) == 1 and args.split == "hard" and args.out is None:
        path = ROOT / "results" / f"hard-reachability-budget{budgets[0]}.json"
    else:
        path = args.out or (ROOT / "results" / f"{args.split}-reachability.json")
    _write_json(path, report)

    summary = {
        key: report[key]
        for key in ("n", "budgets", "env_tool_budget", "min_calls", "by_budget")
    }
    summary["split"] = args.split
    print(json.dumps(summary, indent=2))
    widest = report["by_budget"][-1]
    print(f"failures at budget {widest['budget']}: {widest['n_failures']}")
    print(f"wrote {path}")
    return 0 if tasks and widest["solved"] == len(tasks) else 1


def _score(snapshot, task: dict, solution, budget: int) -> dict:
    from fdagym.env import FDAEnv

    env = FDAEnv(snapshot, [task], run_id="hard-reach", seed=0, tool_budget=budget)
    env.reset(task["id"])
    for call in solution.calls:
        _obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            return env.trajectory["reward"]
    env.step({"tool": "submit_answer", "arguments": {"answer": solution.answer, "citations": solution.citations}})
    return env.trajectory["reward"]


def _memorization() -> int:
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.swapped import scripted_oracle_report
    from fdagym.train.rollout import DEFAULT_TOOL_BUDGET

    report = scripted_oracle_report(
        DiskSnapshot("2026-10-15"),
        load_split("hard_train_holdout"),
        load_split("hard"),
        tool_budget=DEFAULT_TOOL_BUDGET,
    )
    path = ROOT / "results" / "memorization-oracle.json"
    _write_json(path, report)
    shown = {
        "known": report["known"],
        "unseen": report["unseen"],
        "gap": report["gap"],
        "answered_without_looking": {
            key: report["answered_without_looking"][key] for key in ("rate", "correct", "without_looking")
        },
        "mean_tool_calls": report["mean_tool_calls"],
        "swapped_facts": report["swapped_facts"],
        "flags": report["flags"],
        "auto_stop": report["auto_stop"],
    }
    print(json.dumps(shown, indent=2))
    print(f"wrote {path}")
    return 0 if _memorization_ok(report) else 1


def _memorization_ok(report: dict) -> bool:
    return (
        report["gap"]["strict"] == 0.0
        and report["answered_without_looking"]["rate"] == 0.0
        and report["swapped_facts"]["follow_rate"] == 1.0
        and report["auto_stop"] is False
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("budget must be an integer >= 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
