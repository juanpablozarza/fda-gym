from fdagym.train.config import TrainConfig
from fdagym.train.dataset import build_rows
from fdagym.train.dry import run_dry
from tests.env.support import ALL_TASKS, JsonSnapshot


def test_dry_run_gold_is_one_and_junk_is_zero():
    report = run_dry(JsonSnapshot(), ALL_TASKS, TrainConfig(stage=1, n_distractors=1))
    assert report["gold"]["total"] == 1.0
    assert report["junk"]["total"] == 0.0
    assert report["format_only"]["total"] == 0.1
    assert report["advantages_equal"] == [0.0, 0.0, 0.0, 0.0]
    assert report["n_prompts"] >= 1
    length = report["prompt_length"]
    assert length["fits"] is True
    assert length["vllm_max_model_len"] <= 8192
    assert length["vllm_max_model_len"] >= length["measured_prompt_tokens"] + 256
    assert length["max_pack_chars"] == 6000
    assert "worst_case_prompt_tokens" not in length
    assert report["vllm_max_model_len"] == length["vllm_max_model_len"]


def test_rows_are_chat_prompts_with_task_id():
    rows, mix = build_rows(JsonSnapshot(), ALL_TASKS, TrainConfig(n_distractors=1))
    assert rows
    assert mix["easy"] + mix["medium"] + mix["hard"] == len(rows)
    row = rows[0]
    assert row["task_id"].startswith("FG-")
    assert row["prompt"][0]["role"] == "system"
    assert "Question:" in row["prompt"][1]["content"]
    assert "### Document 1" in row["prompt"][1]["content"]
