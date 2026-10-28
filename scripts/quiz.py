"""Terminal quiz over the hidden-answer blocks in learn/*.md.

Usage:
  python scripts/quiz.py learn/00-orientation.md            # all questions, in order
  python scripts/quiz.py learn/00-orientation.md --kind predict
  python scripts/quiz.py learn/00-orientation.md --missed   # only ones you got wrong last time
  python scripts/quiz.py --status                           # progress across all lessons

Answers are appended to learn/answers/<lesson>.jsonl so the agent that wrote the
lesson can read them at its checkpoint.
"""

import argparse
import json
import re
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEARN = ROOT / "learn"
ANSWERS = LEARN / "answers"
BLOCK = re.compile(
    r"<details>\s*<summary>\s*<b>\[(?P<kind>[\w-]+)\]</b>\s*(?P<q>.*?)</summary>(?P<a>.*?)</details>",
    re.DOTALL,
)
TAG = re.compile(r"</?\w+>")


def parse(lesson: Path) -> list[dict]:
    text = lesson.read_text()
    return [
        {"id": f"{lesson.stem}#{i}", "kind": m["kind"], "q": TAG.sub("", m["q"]).strip(), "a": m["a"].strip()}
        for i, m in enumerate(BLOCK.finditer(text), 1)
    ]


def history(lesson: Path) -> dict[str, dict]:
    path = ANSWERS / f"{lesson.stem}.jsonl"
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        latest[rec["id"]] = rec
    return latest


def ask(block: dict, n: int, total: int) -> dict | None:
    print(f"\n\033[33m[{n}/{total}] {block['kind'].upper()}\033[0m  {block['q']}")
    print("\033[2mType your answer (empty line to finish, 'skip' to skip, 'quit' to stop):\033[0m")
    lines = []
    while (line := input("> ")) != "":
        lines.append(line)
    answer = "\n".join(lines).strip()
    if answer == "quit":
        raise KeyboardInterrupt
    if answer in ("", "skip"):
        return None
    print("\n\033[36m--- answer ---\033[0m")
    print(textwrap.indent(block["a"], "  "))
    grade = ""
    while grade not in ("y", "p", "n"):
        grade = input("\nDid you get it? [y]es / [p]artly / [n]o: ").strip().lower()
    return {
        "id": block["id"],
        "kind": block["kind"],
        "question": block["q"],
        "answer": answer,
        "self_grade": grade,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def status() -> None:
    for lesson in sorted(LEARN.glob("[0-9][0-9]-*.md")):
        blocks = parse(lesson)
        if not blocks:
            continue
        hist = history(lesson)
        got = sum(1 for b in blocks if hist.get(b["id"], {}).get("self_grade") == "y")
        tried = sum(1 for b in blocks if b["id"] in hist)
        print(f"{lesson.name:40} {got:>2}/{len(blocks)} solid, {tried - got} shaky, {len(blocks) - tried} untouched")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("lesson", nargs="?", type=Path)
    p.add_argument("--kind")
    p.add_argument("--missed", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()

    if args.status or not args.lesson:
        status()
        return

    blocks = parse(args.lesson)
    if args.kind:
        blocks = [b for b in blocks if b["kind"] == args.kind]
    if args.missed:
        hist = history(args.lesson)
        blocks = [b for b in blocks if hist.get(b["id"], {}).get("self_grade") != "y"]
    if not blocks:
        sys.exit("No matching questions.")

    ANSWERS.mkdir(exist_ok=True)
    out = ANSWERS / f"{args.lesson.stem}.jsonl"
    try:
        for n, block in enumerate(blocks, 1):
            if rec := ask(block, n, len(blocks)):
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    print(f"\nSaved to {out.relative_to(ROOT)}. Tell the lesson's agent you're done so it can review.")


if __name__ == "__main__":
    main()
