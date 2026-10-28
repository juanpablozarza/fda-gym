"""SFT targets are gold JSON copied from the prompt."""

import json

from fdagym.train.config import TrainConfig
from fdagym.train.score import score_completion
from fdagym.train.sft import build_sft_rows
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot


def test_sft_target_is_verbatim_and_scores_one():
    snapshot = JsonSnapshot()
    rows, stats = build_sft_rows(snapshot, ALL_TASKS, TrainConfig(n_distractors=1))
    assert stats["n_kept"] >= 1
    row = next(item for item in rows if item["task_id"] == EXAMPLE_TASK["id"])
    user = row["messages"][1]["content"]
    payload = json.loads(row["completion"])
    assert payload["answer"]
    for citation in payload["citations"]:
        assert citation["id"] in user
        assert citation["quote"] in user
        assert citation["endpoint"] in {"drug/label", "drug/event", "drug/drugsfda"}
    reward = score_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, row["completion"], dense=True)
    strict = score_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, row["completion"])
    assert strict["total"] == 1.0
    assert reward["total"] == 1.0


def test_sft_skips_a_task_whose_answer_is_not_in_the_prompt():
    snapshot = JsonSnapshot()
    hidden = json.loads(json.dumps(EXAMPLE_TASK))
    hidden["id"] = "FG-0098"
    hidden["gold"] = {"kind": "exact", "value": "NOT-IN-THE-PROMPT-XYZ"}
    hidden["answerable"] = True
    rows, stats = build_sft_rows(snapshot, [hidden], TrainConfig(n_distractors=0))
    assert rows == []
    assert stats["skipped"]["answer_not_in_prompt"] == 1
