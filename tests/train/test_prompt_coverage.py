"""Gold citation ids survive into the prompts GRPO and Plan A eval actually render."""

from fdagym.agents.baseline import load_split
from fdagym.openfda import DiskSnapshot
from fdagym.train.context import assemble_documents
from fdagym.train.prompt import render_messages

# Question-only needles and a 6,000-character budget, snapshot 2026-10-15.
# Gold citation ids are not used to rank, so this is the measured floor, not a 95% target.
CITATION_FLOOR = {"train": (686, 720), "test": (202, 211)}


def _citation_ids(task: dict) -> list[str]:
    found = []
    for spec in task.get("gold_citations") or []:
        found.extend(cid for cid in spec.get("ids") or [] if isinstance(cid, str) and cid)
    return found


def _coverage(split: str, tasks: list[dict], snapshot) -> dict:
    id_hits = 0
    id_total = 0
    decoy_hits = 0
    for task in tasks:
        docs = assemble_documents(snapshot, tasks, task, n_distractors=2)
        user = render_messages(task, docs)[1]["content"]
        ids = _citation_ids(task)
        if ids:
            id_total += 1
            if any(cid in user for cid in ids):
                id_hits += 1
        decoys = [doc for doc in docs if doc["task_id"] != task["id"]]
        if decoys and any((doc.get("arguments") or {}).get("generic_name") in user for doc in decoys):
            decoy_hits += 1
    return {
        "split": split,
        "n": len(tasks),
        "id_hits": id_hits,
        "id_total": id_total,
        "decoy_hits": decoy_hits,
    }


def test_gold_citation_ids_and_decoys_cover_both_splits():
    snapshot = DiskSnapshot("2026-10-15")
    train = load_split("train")
    test = load_split("test")
    for tasks, split in ((train, "train"), (test, "test")):
        stats = _coverage(split, tasks, snapshot)
        hits, total = CITATION_FLOOR[split]
        assert stats["id_total"] == total
        assert stats["id_hits"] >= hits, stats
        assert stats["decoy_hits"] == stats["n"], stats
