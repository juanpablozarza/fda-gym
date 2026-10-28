"""Prompt packing keeps the cited record, the decoys, and the question."""

import json

import pytest

from fdagym.train.config import TrainConfig
from fdagym.train.context import _clip, _needles, assemble_documents, render_task_prompt
from fdagym.train.dataset import build_rows
from fdagym.train.prompt_len import assert_prompts_fit
from tests.env.support import ALL_TASKS, JsonSnapshot


def _approval_task():
    return {
        "id": "FG-PACK",
        "type": "approval",
        "question": "What brand name is marketed under application NDA020977?",
        "entity": {"generic_name": "abacavir", "brand_name": "ZIAGEN"},
        "answerable": True,
        "gold": {"kind": "exact", "value": "ZIAGEN"},
        "gold_citations": [{"endpoint": "drug/drugsfda", "ids": ["NDA020977"], "fields": ["products"]}],
        "oracle_calls": [{"tool": "search_approvals", "arguments": {"generic_name": "abacavir"}}],
    }


def _record(application, brand, *, submissions=None, marker=None):
    row = {
        "application_number": application,
        "sponsor_name": "SOME SPONSOR",
        "brand": [brand],
        "products": [{"brand_name": brand, "active_ingredients": [{"name": "ABACAVIR SULFATE"}]}],
        "submissions": submissions
        or [
            {
                "submission_type": "SUPPL",
                "submission_number": "1",
                "submission_status_date": "20100101",
                "note": marker or ("x" * 40),
            }
        ],
    }
    return row


def test_needles_come_from_the_question_not_the_answer_key():
    task = _approval_task()
    ids, drugs = _needles(task)
    assert ids == {"nda020977"}
    assert drugs == set()
    task["question"] = "What is the most common reaction for abacavir?"
    task["gold"] = {"kind": "exact", "value": "ZIAGEN"}
    task["gold_citations"] = [{"endpoint": "drug/drugsfda", "ids": ["NDA999999"], "fields": ["products"]}]
    task["entity"] = {"generic_name": "abacavir", "brand_name": "ZIAGEN"}
    ids, drugs = _needles(task)
    assert ids == set()
    assert "nda999999" not in ids
    assert drugs == {"abacavir"}
    assert "ziagen" not in drugs
    task["question"] = "In what year did the FDA approve the original NDA for ZIAGEN (abacavir)?"
    ids, drugs = _needles(task)
    assert ids == set()
    assert drugs == {"ziagen", "abacavir"}


def test_ranking_never_reads_gold_fields():
    class Guard(dict):
        def __getitem__(self, key):
            if key in {"gold", "gold_citations"}:
                raise AssertionError(key)
            return dict.__getitem__(self, key)

        def get(self, key, default=None):
            if key in {"gold", "gold_citations"}:
                raise AssertionError(key)
            return dict.get(self, key, default)

    task = Guard(_approval_task())
    task["gold"] = {"kind": "exact", "value": "ZIAGEN"}
    task["gold_citations"] = [{"endpoint": "drug/drugsfda", "ids": ["NDA-NOT-IN-QUESTION"], "fields": ["products"]}]
    pack = {
        "tool": "search_approvals",
        "arguments": {"generic_name": "abacavir"},
        "output": {
            "id": "step:1",
            "results": [_record("NDA020977", "NOTTHEANSWER"), _record("ANDA000111", "ZIAGEN")],
            "meta": {"results": {"total": 2}},
        },
        "step_index": 1,
        "task_id": "FG-PACK",
    }
    text = json.dumps(_clip(pack, 6000, task)["output"], ensure_ascii=False)
    assert "NDA020977" in text


def test_survivors_keep_original_order():
    task = _approval_task()
    task["question"] = "What brand name is marketed under application NDA020977 for abacavir?"
    early = _record("ANDA000222", "COMBO", marker="EARLY-COMBO")
    early["products"] = [{"brand_name": "COMBO", "active_ingredients": [{"name": "ABACAVIR SULFATE"}]}]
    junk = _record("ANDA000333", "UNRELATED", marker="JUNK-RECORD")
    junk["products"] = [{"brand_name": "UNRELATED", "active_ingredients": [{"name": "OTHER"}]}]
    match = _record("NDA020977", "NOTTHEANSWER", marker="LATE-MATCH")
    results = [early, junk, match]
    pack = {
        "tool": "search_approvals",
        "arguments": {"generic_name": "abacavir"},
        "output": {"id": "step:1", "results": results, "meta": {"results": {"total": 3}}},
        "step_index": 1,
        "task_id": task["id"],
    }
    # The two matching records fit; the junk record in the middle does not.
    budget = len(json.dumps({"id": "step:1", "results": [early, match], "meta": {"results": {"total": 3}}, "truncated": True, "omitted": 1})) + 40
    text = json.dumps(_clip(pack, budget, task)["output"], ensure_ascii=False)
    assert text.index("EARLY-COMBO") < text.index("NDA020977")
    assert "JUNK-RECORD" not in text
    assert "ZIAGEN" not in text


def test_ranking_does_not_inject_the_gold_answer():
    """The answer string lives only on a non-matching record. A tight budget must drop it."""
    task = _approval_task()
    filler = _record("ANDA111111", "ZIAGEN", marker="LEAK-BRAND")
    filler["products"] = [{"brand_name": "ZIAGEN", "active_ingredients": [{"name": "OTHER"}]}]
    match = _record(
        "NDA020977",
        "NOTTHEANSWER",
        submissions=[
            {
                "submission_type": "ORIG",
                "submission_number": "1",
                "submission_status_date": "19981217",
            }
        ],
    )
    match["products"] = [{"brand_name": "NOTTHEANSWER", "active_ingredients": [{"name": "ABACAVIR SULFATE"}]}]
    results = [filler]
    for index in range(12):
        results.append(_record(f"ANDA{index:06d}", "OTHER", marker="pad-" + ("y" * 80)))
    results.append(match)
    pack = {
        "tool": "search_approvals",
        "arguments": {"generic_name": "abacavir"},
        "output": {"id": "step:1", "results": results, "meta": {"results": {"total": len(results)}}},
        "step_index": 1,
        "task_id": task["id"],
    }
    clipped = _clip(pack, 700, task)
    text = json.dumps(clipped["output"], ensure_ascii=False)
    assert "NDA020977" in text
    assert "ZIAGEN" not in text
    assert "LEAK-BRAND" not in text
    assert len(text) <= 700


def test_long_product_list_does_not_crowd_out_the_original_approval():
    """Mounjaro-shaped: products alone exceed the budget; the year lives only on ORIG."""
    task = _approval_task()
    task["question"] = "In what year did the FDA approve the original NDA for MOUNJARO (tirzepatide)?"
    task["entity"] = {"generic_name": "tirzepatide", "brand_name": "MOUNJARO"}
    task["gold"] = {"kind": "number", "value": 2022}
    task["gold_citations"] = [{"endpoint": "drug/drugsfda", "ids": ["NDA215866"], "fields": ["submissions"]}]
    products = [
        {
            "brand_name": "MOUNJARO",
            "active_ingredients": [{"name": "TIRZEPATIDE", "strength": "EQ " + ("9" * 80) + f" MG {index}"}],
        }
        for index in range(30)
    ]
    record = {
        "application_number": "NDA215866",
        "sponsor_name": "ELI LILLY",
        "brand": ["MOUNJARO"],
        "products": products,
        "submissions": [
            {
                "submission_type": "ORIG",
                "submission_number": "1",
                "submission_status_date": "20220513",
            }
        ],
    }
    pack = {
        "tool": "search_approvals",
        "arguments": {"generic_name": "tirzepatide"},
        "output": {"id": "step:1", "results": [record], "meta": {"results": {"total": 1}}},
        "step_index": 1,
        "task_id": task["id"],
    }
    clipped = _clip(pack, 6000, task)
    text = json.dumps(clipped["output"], ensure_ascii=False)
    assert "NDA215866" in text
    assert "20220513" in text
    assert len(text) <= 6000


def test_oversized_record_keeps_orig_submission_and_drops_the_tail():
    task = _approval_task()
    task["gold"] = {"kind": "number", "value": 1998}
    submissions = [
        {
            "submission_type": "SUPPL",
            "submission_number": str(index),
            "submission_status_date": "20100101",
            "note": f"SUPPLEMENT-ONLY-{index}-" + ("z" * 200),
        }
        for index in range(40)
    ]
    submissions.append(
        {
            "submission_type": "ORIG",
            "submission_number": "1",
            "submission_status_date": "19981217",
            "note": "ORIGINAL-APPROVAL",
        }
    )
    match = _record("NDA020977", "NOTTHEANSWER", submissions=submissions)
    pack = {
        "tool": "search_approvals",
        "arguments": {"generic_name": "abacavir"},
        "output": {"id": "step:1", "results": [match], "meta": {"results": {"total": 1}}},
        "step_index": 1,
        "task_id": task["id"],
    }
    clipped = _clip(pack, 1200, task)
    text = json.dumps(clipped["output"], ensure_ascii=False)
    assert "NDA020977" in text
    assert "19981217" in text
    assert "ORIGINAL-APPROVAL" in text
    assert "SUPPLEMENT-ONLY-39" not in text
    assert len(text) <= 1200


def test_fixture_prompts_keep_decoys_and_ignore_pool_order():
    snapshot = JsonSnapshot()
    task = ALL_TASKS[0]
    forward = assemble_documents(snapshot, ALL_TASKS, task, n_distractors=1)
    backward = assemble_documents(snapshot, list(reversed(ALL_TASKS)), task, n_distractors=1)
    assert [doc["task_id"] for doc in forward] == [doc["task_id"] for doc in backward]
    decoys = [doc for doc in forward if doc["task_id"] != task["id"]]
    assert decoys
    prompt = render_task_prompt(snapshot, ALL_TASKS, task, n_distractors=1)
    user = prompt[1]["content"]
    assert decoys[0]["arguments"]["generic_name"] in user
    assert "Question:" in user


def test_dataset_rows_match_the_shared_renderer():
    snapshot = JsonSnapshot()
    config = TrainConfig(n_distractors=1, stage=1)
    rows, _mix = build_rows(snapshot, ALL_TASKS, config)
    assert rows
    for row in rows:
        task = next(item for item in ALL_TASKS if item["id"] == row["task_id"])
        again = render_task_prompt(
            snapshot,
            ALL_TASKS,
            task,
            n_distractors=config.n_distractors,
            max_chars=config.max_pack_chars,
        )
        assert again == row["prompt"]


def test_eval_split_uses_the_shared_renderer(tmp_path, monkeypatch):
    from fdagym.train.eval_local import eval_split

    seen = {}

    def fake(_snapshot, _tasks, task, **_kwargs):
        seen["id"] = task["id"]
        return [{"role": "system", "content": "s"}, {"role": "user", "content": "Question:\n" + task["question"]}]

    monkeypatch.setattr("fdagym.train.eval_local.render_task_prompt", fake)
    eval_split(
        generate=lambda _messages: '{"answer": "NOT_FOUND", "citations": []}',
        snapshot=JsonSnapshot(),
        tasks=ALL_TASKS[:1],
        model_name="local",
        run_id="pack-eval",
        results_dir=tmp_path,
        split="test",
    )
    assert seen["id"] == ALL_TASKS[0]["id"]


def test_launch_rejects_a_prompt_over_the_budget():
    config = TrainConfig()
    short = {"task_id": "FG-ok", "prompt": [{"role": "user", "content": "Question: hi"}]}
    assert_prompts_fit([short], config)
    huge = {"task_id": "FG-long", "prompt": [{"role": "user", "content": "x" * 30000}]}
    with pytest.raises(RuntimeError, match="FG-long"):
        assert_prompts_fit([short, huge], config)


def test_none_prompt_length_still_respects_the_vllm_window():
    config = TrainConfig(max_prompt_length=None, vllm_max_model_len=128, max_completion_length=16)
    huge = {"task_id": "FG-vllm", "prompt": [{"role": "user", "content": "x" * 5000}]}
    with pytest.raises(RuntimeError, match="FG-vllm"):
        assert_prompts_fit([huge], config)
