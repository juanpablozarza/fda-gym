import json

from fdagym.env import FDAEnv
from fdagym.train.context import pack_distractors, play_oracle
from fdagym.train.gold import gold_submit
from fdagym.train.prompt import render_messages
from fdagym.train.score import score_completion
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot


def test_gold_completion_scores_one():
    snapshot = JsonSnapshot()
    env = FDAEnv(snapshot, ALL_TASKS, run_id="t")
    steps = play_oracle(env, EXAMPLE_TASK)
    gold = gold_submit(EXAMPLE_TASK, steps)
    reward = score_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, json.dumps(gold))
    assert reward["total"] == 1.0
    assert reward["efficiency"] == 1.0


def test_junk_and_format_only():
    snapshot = JsonSnapshot()
    junk = score_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, "bleeding I guess")
    assert junk["total"] == 0.0
    assert junk["format"] == 0.0
    fmt = score_completion(
        snapshot,
        ALL_TASKS,
        EXAMPLE_TASK,
        json.dumps({"answer": "NOT_FOUND", "citations": []}),
    )
    assert fmt["total"] == 0.1
    assert fmt["correctness"] == 0.0


def test_distractor_id_is_not_a_valid_citation():
    snapshot = JsonSnapshot()
    extra = pack_distractors(snapshot, ALL_TASKS, EXAMPLE_TASK, k=2)
    assert extra
    assert "Question:" in render_messages(EXAMPLE_TASK, extra)[1]["content"]
    env = FDAEnv(snapshot, ALL_TASKS, run_id="t")
    gold = gold_submit(EXAMPLE_TASK, play_oracle(env, EXAMPLE_TASK))
    spoof = {
        "answer": gold["answer"],
        "citations": [
            {
                "endpoint": "drug/drugsfda",
                "id": "NDA209637",
                "field": "submissions",
                "quote": "APPROVAL",
            }
        ],
    }
    reward = score_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, json.dumps(spoof))
    assert reward["correctness"] == 0.0
    assert reward["citation"] == 0.0


def test_skipping_oracle_calls_kills_a_gold_answer(monkeypatch):
    snapshot = JsonSnapshot()
    env = FDAEnv(snapshot, ALL_TASKS, run_id="t")
    steps = play_oracle(env, EXAMPLE_TASK)
    gold = gold_submit(EXAMPLE_TASK, steps)

    def no_tools(env, task):
        env.reset(task["id"])
        return []

    monkeypatch.setattr("fdagym.train.score.play_oracle", no_tools)
    reward = score_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, json.dumps(gold))
    assert reward["correctness"] == 0.0
    assert reward["total"] < 0.4
