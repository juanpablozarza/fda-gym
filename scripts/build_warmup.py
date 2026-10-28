"""Rebuild the multi-hop SFT warm-up id list. CPU only. Does not start Modal.

The list this repo trains on is ``data/warmup/mh1_ids.txt``. Its sha256 is the
sorted ids joined by newlines (``1686e70ac36e0ab6…``). This script writes a new
directory. It does not edit ``data/tasks`` or that committed file.

    uv run --extra train python scripts/build_warmup.py --tokenizer /path/to/tokenizer.json

The tokenizer is the Qwen tokenizer saved with the SFT v4 adapter. Replaying the
oracle traces needs the frozen 2026-10-15 snapshot on disk.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_PER_VARIANT = {"sponsor_drug": 16, "sponsor_faers": 16, "sponsor_count": 16}
N_COMPARE = 24
SEED = 0
LEAK_TRIES = 6000
COMMITTED_IDS = ROOT / "data" / "warmup" / "mh1_ids.txt"
_APP = re.compile(r"\b(?:NDA|ANDA|BLA)\d+")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    out = Path(args.out).resolve()
    tasks_dir = (ROOT / "data" / "tasks").resolve()
    if out == tasks_dir or tasks_dir in out.parents:
        print("refusing to write into data/tasks", file=sys.stderr)
        return 2
    from tokenizers import Tokenizer

    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.rewards.hops import hop_target_searched
    from fdagym.tasks.disjoint import holdout_entities, task_entities
    from fdagym.tasks.hard import sponsor_surface_caches
    from fdagym.tasks.hard_solver import parse_question
    from fdagym.tasks.leaks import clear_leaks, leak_report
    from fdagym.train.rollout import QwenToolCodec, RolloutLimits
    from fdagym.train.sft_select import task_ids_digest
    from fdagym.train.sft_tools import IGNORE_INDEX, build_sft_rows

    class QwenTok:
        def __init__(self):
            self.t = Tokenizer.from_file(str(args.tokenizer))
            self.eos_token_id = self.t.token_to_id("<|im_end|>")
            self.pad_token_id = self.t.token_to_id("<|endoftext|>")

        def encode(self, text):
            return self.t.encode(text, add_special_tokens=False).ids

        def decode(self, ids):
            return self.t.decode(list(ids), skip_special_tokens=False)

    snap = DiskSnapshot(args.snapshot)
    hard_train = load_split("hard_train")
    hard = load_split("hard")
    holdout = load_split("hard_train_holdout")
    test = load_split("test")
    blocked_ids = {task["id"] for task in [*hard, *holdout, *test]}
    eval_entities = holdout_entities([*hard, *test], snap)
    holdout_surface = holdout_entities(holdout, snap)
    holdout_anchors = set().union(*(_anchor(task) for task in holdout))
    holdout_targets = {task["entity"]["generic_name"] for task in holdout if task["type"] == "sponsor_hop"}

    def clean_eval(task):
        entities = task_entities(task, snap)
        return (
            task["id"] not in blocked_ids
            and not (entities.drugs & eval_entities.drugs)
            and not (entities.applications & eval_entities.applications)
            and not (entities.sponsors & eval_entities.sponsors)
        )

    def clean(task):
        return clean_eval(task) and not (task_entities(task, snap).applications & holdout_surface.applications)

    by_variant: dict[str, list[dict]] = {}
    for task in hard_train:
        variant = parse_question(task["question"])["variant"]
        task["_variant"] = variant
        key = variant if task["type"] == "sponsor_hop" else "compare:" + variant
        by_variant.setdefault(key, []).append(task)
    rng = random.Random(SEED)
    tokenizer = QwenTok()
    codec = QwenToolCodec(prompt_version="v4", tool_budget=8)
    limits = RolloutLimits(temperature=0.0, tool_budget=8, prompt_version="v4")
    pool = [*load_split("train"), *hard_train]
    labels, texts = sponsor_surface_caches(snap, [*hard_train, *hard, *holdout])

    def worst(sub):
        report = leak_report(sub, label_counts=labels, boxed_texts=texts, train_tasks=sub)
        scores = [row["z"] for rules in report.values() for row in rules.values() if row["n"] >= 5]
        max_z = max(scores) if scores else 0.0
        holdout_hits = sum(
            1 for task in sub if _anchor(task) & holdout_anchors or task["entity"]["generic_name"] in holdout_targets
        )
        return (len(clear_leaks(report)), int(max_z >= 2.0), holdout_hits, round(max_z, 3))

    def play(candidates, n, *, leak_search=False, require_clean=True):
        candidates = [task for task in candidates if (clean(task) if require_clean else clean_eval(task))]
        built = build_sft_rows(
            snap,
            pool,
            [task["id"] for task in candidates],
            tokenizer,
            codec,
            limits,
            max_length=8192,
            run_id="warmup-build",
        )
        good = {row.task_id: (row, example) for row, example in zip(built["rollouts"], built["rows"], strict=True)}
        ok = [
            task
            for task in candidates
            if task["id"] in good
            and (
                task["type"] != "sponsor_hop" or hop_target_searched(good[task["id"]][0].trajectory.get("steps"), task)
            )
        ]
        take = min(n, len(ok))
        if leak_search:
            best = None
            for _ in range(LEAK_TRIES):
                sub = rng.sample(ok, take)
                score = worst(sub)
                if best is None or score < best[0]:
                    best = (score, sub)
            print("  leak search over", len(ok), "worst (n_clear, max_z):", best[0])
            sub = best[1]
        else:
            sub = rng.sample(ok, take)
        return [(task, *good[task["id"]]) for task in sub], built["skipped"]

    chosen = []
    skipped_all = []
    for variant, count in N_PER_VARIANT.items():
        kept, skipped = play(by_variant[variant], count, leak_search=True, require_clean=False)
        chosen.extend(kept)
        skipped_all.extend(skipped)
        print(
            variant,
            "available",
            len(by_variant[variant]),
            "kept",
            len(kept),
            "clean",
            sum(clean(task) for task, _, _ in kept),
        )
    compare_keys = sorted(key for key in by_variant if key.startswith("compare:"))
    per = {key: 0 for key in compare_keys}
    for index in range(N_COMPARE):
        per[compare_keys[index % len(compare_keys)]] += 1
    for key in compare_keys:
        kept, skipped = play(by_variant[key], per[key])
        chosen.extend(kept)
        skipped_all.extend(skipped)
        print(
            key, "available", len(by_variant[key]), "kept", len(kept), "clean", sum(clean(task) for task, _, _ in kept)
        )

    ids = [task["id"] for task, _, _ in chosen]
    out.mkdir(parents=True, exist_ok=True)
    (out / "warmup_ids.txt").write_text("\n".join(ids) + "\n")
    with (out / "warmup_tasks.jsonl").open("w") as handle:
        for task, _, _ in chosen:
            handle.write(json.dumps({key: value for key, value in task.items() if key != "_variant"}) + "\n")
    with (out / "warmup_trajectories.jsonl").open("w") as handle:
        for task, rollout, row in chosen:
            text = tokenizer.decode(row["input_ids"])
            prompt = tokenizer.decode(rollout.prompt_ids)
            handle.write(
                json.dumps(
                    {
                        "task_id": task["id"],
                        "type": task["type"],
                        "variant": task["_variant"],
                        "reward": rollout.reward,
                        "n_tokens": len(row["input_ids"]),
                        "n_supervised": sum(1 for token in row["labels"] if token != IGNORE_INDEX),
                        "prompt": prompt,
                        "completion": text[len(prompt) :],
                    }
                )
                + "\n"
            )
    entities = {task["id"]: task_entities(task, snap) for task, _, _ in chosen}

    def union(pick):
        return set().union(*(pick(item) for item in entities.values()))

    digest = task_ids_digest(ids)
    audit = {
        "n": len(chosen),
        "by_variant": Counter(task["_variant"] for task, _, _ in chosen),
        "all_clean_vs_hard_test": all(clean_eval(task) for task, _, _ in chosen),
        "n_no_holdout_app_overlap": sum(clean(task) for task, _, _ in chosen),
        "id_overlap_hard": sorted(set(ids) & {task["id"] for task in hard}),
        "id_overlap_holdout": sorted(set(ids) & {task["id"] for task in holdout}),
        "id_overlap_test": sorted(set(ids) & {task["id"] for task in test}),
        "drug_overlap_hard_or_test": sorted(union(lambda item: item.drugs) & eval_entities.drugs),
        "app_overlap_hard_or_test": sorted(union(lambda item: item.applications) & eval_entities.applications),
        "sponsor_overlap_hard_or_test": sorted(union(lambda item: item.sponsors) & eval_entities.sponsors),
        "app_overlap_holdout": sorted(union(lambda item: item.applications) & holdout_surface.applications),
        "holdout_anchor_overlap": sorted(set().union(*(_anchor(task) for task, _, _ in chosen)) & holdout_anchors),
        "holdout_hop_target_overlap": sorted(
            {task["entity"]["generic_name"] for task, _, _ in chosen if task["type"] == "sponsor_hop"} & holdout_targets
        ),
        "all_split_hard_train": all(task["split"] == "hard_train" for task, _, _ in chosen),
        "min_reward": min(rollout.reward["total"] for _, rollout, _ in chosen),
        "hop_target_searched_all": all(
            hop_target_searched(rollout.trajectory.get("steps"), task)
            for task, rollout, _ in chosen
            if task["type"] == "sponsor_hop"
        ),
        "max_tokens": max(len(row["input_ids"]) for _, _, row in chosen),
        "ids_sha256": digest,
        "skipped": skipped_all[:20],
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=list))
    if COMMITTED_IDS.is_file():
        committed = [line.strip() for line in COMMITTED_IDS.read_text().splitlines() if line.strip()]
        print("matches committed ids:", task_ids_digest(committed) == digest and set(committed) == set(ids))
    print(json.dumps({key: value for key, value in audit.items() if key != "skipped"}, indent=1, default=list))
    return 0


def _anchor(task: dict) -> set[str]:
    return set(_APP.findall(task["question"]))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the multi-hop warm-up id list. Does not start Modal.")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json from the SFT v4 adapter")
    parser.add_argument("--out", default=str(ROOT / "results" / "warmup-mh1"))
    parser.add_argument("--snapshot", default="2026-10-15")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
