"""Write the review fixture into web/public/results/. Synthetic, fictional drugs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "results"
W = {"correctness": 0.60, "citation": 0.25, "format": 0.10, "efficiency": 0.05}


def total(**components: float) -> float:
    return round(sum(W[k] * v for k, v in components.items()), 10)


T1_ID = "7c1e9a42-0d5e-4b8a-9f21-c06e1a7db31f"
WARFARIN_ID = "0cbce382-9c88-4f58-ae0f-532a841e8f95"

t1 = {
    "id": "tr-fx-9001-s0",
    "run_id": "fixture-run",
    "task_id": "FG-9001",
    "model": {
        "name": "openrouter/qwen/qwen-2.5-7b-instruct",
        "kind": "baseline",
        "checkpoint_step": None,
    },
    "seed": 0,
    "snapshot_version": "2026-10-15",
    "started_at": "2026-10-16T15:00:00Z",
    "steps": [
        {
            "index": 1,
            "thought": "Find the current label, then the serious FAERS count for 2025.",
            "tool": "search_drug_label",
            "arguments": {"generic_name": "zorvetinib"},
            "output": {
                "total": 1,
                "results": [
                    {
                        "set_id": T1_ID,
                        "brand_name": "Zorvenil",
                        "sections": ["boxed_warning", "indications_and_usage", "warnings_and_cautions"],
                    }
                ],
            },
            "error": None,
            "latency_ms": 42,
        },
        {
            "index": 2,
            "thought": None,
            "tool": "get_label_section",
            "arguments": {"set_id": T1_ID, "section": "boxed_warning"},
            "output": {
                "set_id": T1_ID,
                "version": 4,
                "effective_time": "20250611",
                "boxed_warning": [
                    "WARNING: HEPATOTOXICITY. Zorvenil can cause severe and fatal liver injury. Monitor hepatic function."
                ],
            },
            "error": None,
            "latency_ms": 38,
        },
        {
            "index": 3,
            "thought": None,
            "tool": "query_adverse_events",
            "arguments": {"drug": "zorvetinib", "serious": 1, "year": 2025},
            "output": {
                "meta": {"results": {"total": 1284}},
                "results": [{"safetyreportid": "25-US-0418823", "serious": "1"}],
            },
            "error": None,
            "latency_ms": 211,
        },
        {
            "index": 4,
            "thought": None,
            "tool": "submit_answer",
            "arguments": {
                "answer": "1284 serious reports in 2025. Yes, boxed warning for hepatotoxicity.",
                "citations": [
                    {
                        "endpoint": "drug/label",
                        "id": T1_ID,
                        "field": "boxed_warning",
                        "quote": "Zorvenil can cause severe and fatal liver injury",
                    },
                    {
                        "endpoint": "drug/event",
                        "id": "step:3",
                        "field": "meta.results.total",
                        "quote": "1284",
                    },
                    {
                        "endpoint": "drug/event",
                        "id": "25-US-0418823",
                        "field": "results",
                        "quote": "25-US-0418823",
                    },
                ],
            },
            "output": None,
            "error": None,
            "latency_ms": 0,
        },
    ],
    "final_answer": {
        "answer": "1284 serious reports in 2025. Yes, boxed warning for hepatotoxicity.",
        "citations": [
            {
                "endpoint": "drug/label",
                "id": T1_ID,
                "field": "boxed_warning",
                "quote": "Zorvenil can cause severe and fatal liver injury",
            },
            {
                "endpoint": "drug/event",
                "id": "step:3",
                "field": "meta.results.total",
                "quote": "1284",
            },
            {
                "endpoint": "drug/event",
                "id": "25-US-0418823",
                "field": "results",
                "quote": "25-US-0418823",
            },
        ],
    },
    "terminated_by": "submit",
    "reward": {
        "total": total(correctness=1, citation=1, format=1, efficiency=1),
        "correctness": 1.0,
        "citation": 1.0,
        "format": 1.0,
        "efficiency": 1.0,
        "citations_valid": 3,
        "citations_total": 3,
        "notes": [
            "citation 3/3 valid",
            f"valid: {T1_ID} boxed_warning (quote fuzzy 1.00)",
            "valid: step:3 meta.results.total (quote fuzzy 1.00)",
            "valid: 25-US-0418823 results (quote fuzzy 1.00)",
            "boolean: polarity and keyword match",
            "efficiency 1.00: min_tool_calls 3 over 3 tool calls",
            "total 1.0000 = 0.60*1.0000 + 0.25*1.0000 + 0.10*1.0000 + 0.05*1.0000",
        ],
    },
    "usage": {
        "prompt_tokens": 2100,
        "completion_tokens": 310,
        "cost_usd": 0.0008,
        "latency_ms": 1900,
    },
}

t2 = {
    "id": "tr-fx-9002-s0",
    "run_id": "fixture-run",
    "task_id": "FG-9002",
    "model": {
        "name": "openrouter/qwen/qwen-2.5-7b-instruct",
        "kind": "baseline",
        "checkpoint_step": None,
    },
    "seed": 0,
    "snapshot_version": "2026-10-15",
    "started_at": "2026-10-16T15:02:00Z",
    "steps": [
        {
            "index": 1,
            "thought": "Search the label. I already know warfarin bleeds; reuse that quote.",
            "tool": "search_drug_label",
            "arguments": {"generic_name": "velmorastat"},
            "output": {
                "total": 1,
                "results": [
                    {
                        "set_id": "aa11bb22-3344-5566-7788-9900aabbccdd",
                        "brand_name": "Velmora",
                        "sections": ["indications_and_usage", "adverse_reactions"],
                    }
                ],
            },
            "error": None,
            "latency_ms": 40,
        },
        {
            "index": 2,
            "thought": None,
            "tool": "submit_answer",
            "arguments": {
                "answer": "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
                "citations": [
                    {
                        "endpoint": "drug/label",
                        "id": WARFARIN_ID,
                        "field": "boxed_warning",
                        "quote": "Warfarin sodium can cause major or fatal bleeding",
                    }
                ],
            },
            "output": None,
            "error": None,
            "latency_ms": 0,
        },
    ],
    "final_answer": {
        "answer": "Yes. Bleeding risk: warfarin can cause major or fatal bleeding.",
        "citations": [
            {
                "endpoint": "drug/label",
                "id": WARFARIN_ID,
                "field": "boxed_warning",
                "quote": "Warfarin sodium can cause major or fatal bleeding",
            }
        ],
    },
    "terminated_by": "submit",
    "reward": {
        "total": total(correctness=0, citation=0, format=1, efficiency=0),
        "correctness": 0.0,
        "citation": 0.0,
        "format": 1.0,
        "efficiency": 0.0,
        "citations_valid": 0,
        "citations_total": 1,
        "notes": [
            f"invalid: {WARFARIN_ID} boxed_warning id not in this trajectory",
            "correctness gated: no valid citation",
            "efficiency not counted: correctness is 0 (1 tool calls)",
            "total 0.1000 = 0.60*0.0000 + 0.25*0.0000 + 0.10*1.0000 + 0.05*0.0000",
        ],
    },
    "usage": {
        "prompt_tokens": 980,
        "completion_tokens": 120,
        "cost_usd": 0.0003,
        "latency_ms": 640,
    },
}

t3 = {
    "id": "tr-fx-9003-s0",
    "run_id": "fixture-run",
    "task_id": "FG-9003",
    "model": {
        "name": "openrouter/qwen/qwen-2.5-7b-instruct",
        "kind": "baseline",
        "checkpoint_step": None,
    },
    "seed": 0,
    "snapshot_version": "2026-10-15",
    "started_at": "2026-10-16T15:04:00Z",
    "steps": [
        {
            "index": 1,
            "thought": "Count serious FAERS rows for pemradine in 2025.",
            "tool": "query_adverse_events",
            "arguments": {"drug": "pemradine", "serious": 1, "year": 2025},
            "output": {"meta": {"results": {"total": 412}}, "results": []},
            "error": None,
            "latency_ms": 188,
        },
        {
            "index": 2,
            "thought": None,
            "tool": "submit_answer",
            "arguments": {
                "answer": 1284,
                "citations": [
                    {
                        "endpoint": "drug/event",
                        "id": "step:1",
                        "field": "meta.results.total",
                        "quote": "412",
                    }
                ],
            },
            "output": None,
            "error": None,
            "latency_ms": 0,
        },
    ],
    "final_answer": {
        "answer": 1284,
        "citations": [
            {
                "endpoint": "drug/event",
                "id": "step:1",
                "field": "meta.results.total",
                "quote": "412",
            }
        ],
    },
    "terminated_by": "submit",
    "reward": {
        "total": total(correctness=0, citation=1, format=1, efficiency=0),
        "correctness": 0.0,
        "citation": 1.0,
        "format": 1.0,
        "efficiency": 0.0,
        "citations_valid": 1,
        "citations_total": 1,
        "notes": [
            "citation 1/1 valid",
            "valid: step:1 meta.results.total (quote fuzzy 1.00)",
            "number: 1284 vs gold 412 (rel_error > tolerance)",
            "efficiency not counted: correctness is 0 (1 tool calls)",
            "total 0.3500 = 0.60*0.0000 + 0.25*1.0000 + 0.10*1.0000 + 0.05*0.0000",
        ],
    },
    "usage": {
        "prompt_tokens": 740,
        "completion_tokens": 40,
        "cost_usd": 0.0002,
        "latency_ms": 510,
    },
}

TRAJECTORIES = [t1, t2, t3]

TASKS = [
    {
        "id": "FG-9001",
        "question": "How many serious adverse event reports listing zorvetinib as primary suspect were received in 2025, and does its current label carry a boxed warning?",
        "type": "composite",
        "sources": ["drug/label", "drug/event"],
        "difficulty": "medium",
        "entity": {"generic_name": "zorvetinib", "brand_name": "Zorvenil"},
    },
    {
        "id": "FG-9002",
        "question": "Does the label for velmorastat list a boxed warning?",
        "type": "boxed_warning",
        "sources": ["drug/label"],
        "difficulty": "easy",
        "entity": {"generic_name": "velmorastat", "brand_name": "Velmora"},
    },
    {
        "id": "FG-9003",
        "question": "How many serious FAERS reports for pemradine were received in 2025?",
        "type": "faers_count",
        "sources": ["drug/event"],
        "difficulty": "medium",
        "entity": {"generic_name": "pemradine"},
    },
]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def tool_calls(traj: dict) -> int:
    return sum(1 for s in traj["steps"] if s["tool"] != "submit_answer")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "fixture-run"
    run_dir.mkdir(exist_ok=True)

    jsonl = OUT / "fixture-run.jsonl"
    with jsonl.open("w") as f:
        for traj in TRAJECTORIES:
            f.write(json.dumps(traj, separators=(",", ":")) + "\n")

    models = [
        {
            "name": "openrouter/qwen/qwen-2.5-7b-instruct",
            "kind": "baseline",
            "detail": "no fine-tune",
            "mean_reward": mean([t["reward"]["total"] for t in TRAJECTORIES]),
            "pass_rate": mean([1.0 if t["reward"]["correctness"] > 0 else 0.0 for t in TRAJECTORIES]),
            "citation_validity": mean([t["reward"]["citation"] for t in TRAJECTORIES]),
            "mean_tool_calls": mean([tool_calls(t) for t in TRAJECTORIES]),
            "cost_usd": round(sum(t["usage"]["cost_usd"] for t in TRAJECTORIES), 6),
            "ours": False,
        }
    ]

    leaderboard = {
        "example": True,
        "split": "fixture",
        "n_tasks": 3,
        "dataset_tasks": 96,
        "n_seeds": 1,
        "temperature": 0.2,
        "snapshot_version": "2026-10-15",
        "env": "openfda-reg-v0.3",
        "n_sources": 3,
        "kpis": {
            "reward_lift": None,
            "citation_validity": models[0]["citation_validity"],
            "tool_calls_per_task": models[0]["mean_tool_calls"],
            "tool_calls_base": None,
            "gap_to_frontier": None,
        },
        "models": models,
    }
    (OUT / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2) + "\n")
    (OUT / "tasks.json").write_text(json.dumps(TASKS, indent=2) + "\n")
    summary = {
        "run_id": "fixture-run",
        "example": True,
        "n_trajectories": len(TRAJECTORIES),
        "models": models,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {jsonl} ({len(TRAJECTORIES)} lines)")
    for t in TRAJECTORIES:
        print(f"  {t['id']}  R={t['reward']['total']:.2f}  cite={t['reward']['citations_valid']}/{t['reward']['citations_total']}")


if __name__ == "__main__":
    main()
