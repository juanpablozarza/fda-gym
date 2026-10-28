"""Run a task split through OpenRouter models via LiteLLM. Writes results/<run_id>.jsonl."""

from __future__ import annotations

import argparse
import os
import sys

import fdagym  # noqa: F401  loads .env
from fdagym.agents.baseline import COST_CAP_USD, run_baseline

DEFAULT_MODELS = [
    "openrouter/openai/gpt-4o-mini",
    "openrouter/anthropic/claude-3.5-haiku",
    "openrouter/qwen/qwen-2.5-7b-instruct",
]
HAIKU_FALLBACKS = [
    "openrouter/anthropic/claude-3.5-haiku",
    "openrouter/anthropic/claude-3-haiku",
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/anthropic/claude-3.5-haiku-20241022",
]
QWEN_FALLBACKS = [
    "openrouter/qwen/qwen-2.5-7b-instruct",
    "openrouter/qwen/qwen2.5-7b-instruct",
    "openrouter/qwen/qwen-2.5-7b-instruct:free",
]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _prefixed(name: str) -> str:
    return name if name.startswith("openrouter/") or name == "fake" else f"openrouter/{name}"


def resolve_model(name: str) -> str:
    """Try the requested OpenRouter slug, then known fallbacks on 404."""
    from litellm import completion

    requested = _prefixed(name)
    candidates = [requested]
    if "haiku" in requested:
        candidates.extend(slug for slug in HAIKU_FALLBACKS if slug not in candidates)
    if "qwen" in requested:
        candidates.extend(slug for slug in QWEN_FALLBACKS if slug not in candidates)
    last_error = None
    for slug in candidates:
        try:
            completion(
                model=slug,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
                drop_params=True,
            )
            if slug != requested:
                print(f"resolved {requested} -> {slug}", file=sys.stderr)
            return slug
        except Exception as exc:  # noqa: BLE001 — OpenRouter slug probe
            last_error = exc
            text = str(exc)
            if "404" in text or "not found" in text.lower() or "No endpoints found" in text:
                print(f"slug miss {slug}: {type(exc).__name__}", file=sys.stderr)
                continue
            if "402" in text or "Insufficient credits" in text:
                print(f"slug {slug} 402 insufficient credits", file=sys.stderr)
                continue
            print(f"slug {slug} raised {type(exc).__name__}; using it anyway", file=sys.stderr)
            return slug
    raise RuntimeError(f"no working slug for {requested}: {type(last_error).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), type=_csv)
    parser.add_argument("--seeds", default="1", type=_ints)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cost-cap", type=float, default=COST_CAP_USD)
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to wait after each episode (rate limits).")
    parser.add_argument("--resolve-slugs", action="store_true", help="Probe OpenRouter and replace 404 slugs.")
    args = parser.parse_args(argv)

    print("OPENROUTER_API_KEY loaded", bool(os.getenv("OPENROUTER_API_KEY")))
    models = [_prefixed(name) for name in args.models]
    if args.resolve_slugs:
        models = [resolve_model(name) for name in models]
    print("models", models)
    summary = run_baseline(
        split=args.split,
        models=models,
        seeds=args.seeds,
        run_id=args.run_id,
        cost_cap=args.cost_cap,
        pause_s=args.pause,
    )
    print(json_summary(summary))
    if summary.get("aborted") or summary.get("failure_count", 0):
        if summary.get("aborted"):
            print("aborted:", summary["aborted"], file=sys.stderr)
        if summary.get("failure_count", 0):
            print(
                f"incomplete: {summary['failure_count']} episode failure(s) recorded in "
                f"results/{args.run_id}/failures.jsonl",
                file=sys.stderr,
            )
        return 2
    return 0


def json_summary(summary: dict) -> str:
    import json

    return json.dumps(summary, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
