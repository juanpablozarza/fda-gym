"""CPU checks for tool-call GRPO, the SFT mask, and the Modal entrypoints."""

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

from fdagym.rewards import score
from fdagym.train.config import QWEN_3B, TrainConfig
from fdagym.train.eval_tools import eval_tool_tasks, tool_eval_model
from fdagym.train.grpo_tools import ToolGRPOTrainer, build_tool_rows, resolve_tool_start, unused_reward
from fdagym.train.rollout import ASSISTANT_TURN, QwenToolCodec, RolloutLimits
from fdagym.train.rollout_dry import CharTokenizer, OracleGenerator
from fdagym.train.sft_config import SFTConfig
from fdagym.train.sft_tools import IGNORE_INDEX, MIN_SFT_REWARD, build_sft_rows, split_sft_by_drug
from fdagym.train.tool_config import ToolTrainConfig, tool_train_config
from fdagym.train.tool_score import episode_scores, pad_rows, training_reward, trl_group_advantages
from tests.env.support import ALL_TASKS, JsonSnapshot, task_nausea
from tests.train.test_rollout import _run

ROOT = Path(__file__).resolve().parents[2]


def test_approved_limits_and_plan_a_defaults_stay():
    limits = RolloutLimits()
    assert limits.max_turns == 9
    assert limits.tool_budget == 8
    assert limits.prompt_version == "v4"
    assert limits.max_new_tokens == 256
    assert limits.max_tool_chars == 3500
    assert limits.max_context_tokens == 8192
    assert limits.force_submit is False
    tool = tool_train_config()
    assert tool.num_generations == 8
    assert tool.per_device_train_batch_size == 8
    assert tool.gradient_accumulation_steps == 1
    assert tool.gradient_checkpointing is True
    assert tool.vllm_gpu_memory_utilization == 0.30
    assert tool.logprob_chunk_size == 256
    assert "logprob_chunk_size" not in tool.grpo_kwargs("/tmp/out")
    assert tool.max_prompt_length == 2048
    assert tool.max_completion_length == 6144
    assert tool.max_prompt_length + tool.max_completion_length == 8192
    assert tool.max_steps == 150
    assert tool.dense_reward is True
    assert tool_train_config(dense_reward=False).dense_reward is False
    assert tool.grpo_kwargs("/tmp/out")["log_completions"] is False
    assert tool.grpo_kwargs("/tmp/out")["max_steps"] == 150
    plan_a = TrainConfig()
    assert plan_a.max_prompt_length == 8192
    assert plan_a.max_pack_chars == 6000
    assert plan_a.vllm_max_model_len == 8192
    assert plan_a.dense_reward is True
    assert plan_a.max_completion_length == 256
    assert plan_a.max_steps == 400
    assert plan_a.gradient_checkpointing is False
    wrapped = ToolTrainConfig()
    assert wrapped.tool_parse_bonus == 0.05
    assert wrapped.tool_budget == 8
    assert wrapped.prompt_version == "v4"
    assert wrapped.limits.tool_budget == 8
    assert wrapped.limits.prompt_version == "v4"
    assert wrapped.limits.max_turns == 9
    assert wrapped.limits.force_submit is False


def test_sft_config_matches_the_lora_and_the_epoch():
    cfg = SFTConfig()
    assert cfg.learning_rate == 1e-4
    assert cfg.num_train_epochs == 1.0
    assert cfg.per_device_train_batch_size == 4
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.max_length == 8192
    assert cfg.gradient_checkpointing is True
    assert cfg.min_reward == MIN_SFT_REWARD == 0.97
    assert cfg.val_drug_fraction == 0.05
    assert cfg.eval_steps == 50
    assert cfg.lora_kwargs()["r"] == 16
    assert cfg.lora_kwargs()["lora_alpha"] == 32
    kwargs = cfg.trainer_kwargs("/tmp/out")
    assert kwargs["num_train_epochs"] == 1.0
    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["eval_steps"] == 50
    assert kwargs["prediction_loss_only"] is True
    assert cfg.trainer_kwargs("/tmp/out", eval_split=False).get("eval_strategy") is None


def test_parse_bonus_is_once_per_episode_and_stays_out_of_the_trajectory():
    assert training_reward(1.0, 2, 0.05) == pytest.approx(1.05)
    assert training_reward(1.0, 1, 0.05) == pytest.approx(1.05)
    assert training_reward(0.1, 0, 0.05) == pytest.approx(0.1)
    assert training_reward(1.0, 3, 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        training_reward(1.0, 1, -0.01)
    rollouts, _generator, _tokenizer, _codec = _run(["FG-0001", "FG-0002"], codec=QwenToolCodec())
    by_id = {rollout.task_id: rollout for rollout in rollouts}
    assert by_id["FG-0001"].parsed_tool_calls == 2
    assert by_id["FG-0002"].parsed_tool_calls == 1
    assert by_id["FG-0001"].reward["total"] == 1.0
    assert by_id["FG-0001"].trajectory["reward"]["total"] == 1.0
    task = next(task for task in ALL_TASKS if task["id"] == "FG-0001")
    strict, base, shaped, hops = episode_scores(by_id["FG-0001"], task, JsonSnapshot(), dense=True, bonus=0.05)
    assert strict == 1.0
    assert base == 1.0
    assert shaped == pytest.approx(1.05)
    assert hops == 0.0
    assert by_id["FG-0001"].trajectory["reward"]["total"] == 1.0
    assert by_id["FG-0001"].trajectory["usage"]["prompt_tokens"] == len(by_id["FG-0001"].prompt_ids)
    assert by_id["FG-0001"].trajectory["usage"]["completion_tokens"] == len(by_id["FG-0001"].completion_ids)


def test_dense_flag_lifts_a_near_miss_and_the_bonus_stacks_once():
    from types import SimpleNamespace

    from tests.rewards.test_dense import _approval_task, _steps, _submit

    task = _approval_task()
    final = _submit(19981217, quote="nope")
    trajectory = {"steps": _steps(), "final_answer": final}
    strict_reward = score(trajectory, task)
    rollout = SimpleNamespace(reward=dict(strict_reward), trajectory=trajectory, parsed_tool_calls=1)
    strict, base, shaped, hops = episode_scores(rollout, task, None, dense=True, bonus=0.05)
    assert strict == strict_reward["total"] == 0.1
    assert base > strict
    assert shaped == pytest.approx(base + 0.05)
    assert hops == 0.0
    _off, off_base, off_shaped, off_hops = episode_scores(rollout, task, None, dense=False, bonus=0.05)
    assert off_hops == 0.0
    assert off_base == 0.1
    assert off_shaped == pytest.approx(0.15)
    assert rollout.reward["total"] == 0.1


def test_qwen_assistant_header_is_part_of_the_masked_tool_span():
    rollouts, _generator, _tokenizer, _codec = _run(["FG-0002"], codec=QwenToolCodec())
    rollout = rollouts[0]
    tool_spans = [span for span in rollout.spans if span.role == "tool"]
    assert tool_spans
    cursor = 0
    for span in rollout.spans:
        chunk = rollout.loss_mask[cursor : cursor + len(span.token_ids)]
        if span.role == "tool":
            assert span.text.endswith(ASSISTANT_TURN)
            assert "<|im_start|>assistant\n" in span.text
            assert chunk == tuple(0 for _ in span.token_ids)
        cursor += len(span.token_ids)
    model_text = "".join(span.text for span in rollout.spans if span.role == "model")
    assert ASSISTANT_TURN not in model_text
    prompt = QwenToolCodec().prompt(task_nausea()["question"])
    assert prompt.endswith(ASSISTANT_TURN)
    assert "<tool_response>" not in prompt


def test_group_advantages_use_the_unbiased_std():
    std = (1 / 3) ** 0.5
    advantages = trl_group_advantages([1.0, 0.0, 1.0, 0.0], 4, scale=True)
    assert advantages[0] == pytest.approx(0.5 / (std + 1e-4))
    assert advantages[1] == pytest.approx(-0.5 / (std + 1e-4))
    assert trl_group_advantages([1.0, 0.0], 2, scale=False) == [0.5, -0.5]
    assert trl_group_advantages([1.0, 1.0, 0.0, 0.0], 2, scale=True)[:2] == [0.0, 0.0]
    with pytest.raises(ValueError):
        trl_group_advantages([1.0, 0.0, 1.0], 2)
    ids, mask = pad_rows([[1, 2], [3]], pad_value=0, side="left")
    assert ids == [[1, 2], [0, 3]]
    assert mask == [[1, 1], [0, 1]]
    ids, mask = pad_rows([[1], [2, 3]], pad_value=9, side="right")
    assert ids == [[1, 9], [2, 3]]
    assert mask == [[1, 0], [1, 1]]


def test_sft_rows_mask_the_prompt_and_the_tool_spans():
    snapshot = JsonSnapshot()
    built = build_sft_rows(snapshot, ALL_TASKS, ["FG-0001", "FG-0002"], CharTokenizer(), max_length=8192)
    assert built["n_kept"] == 2
    assert built["n_skipped"] == 0
    for row, rollout in zip(built["rows"], built["rollouts"], strict=True):
        assert row["labels"][: len(rollout.prompt_ids)] == [IGNORE_INDEX] * len(rollout.prompt_ids)
        completion_labels = row["labels"][len(rollout.prompt_ids) :]
        for token_id, bit, label in zip(rollout.completion_ids, rollout.loss_mask, completion_labels, strict=True):
            if bit:
                assert label == token_id
            else:
                assert label == IGNORE_INDEX
        assert sum(1 for label in row["labels"] if label != IGNORE_INDEX) == sum(rollout.loss_mask)
        assert "<tool_response>" in "".join(span.text for span in rollout.spans if span.role == "tool")


def test_sft_keeps_a_row_at_the_efficiency_floor_and_drops_below_it():
    easy = json.loads(json.dumps(next(task for task in ALL_TASKS if task["id"] == "FG-0001")))
    easy["id"] = "FG-EASY"
    easy["difficulty_features"] = dict(easy["difficulty_features"])
    easy["difficulty_features"]["min_tool_calls"] = 1
    tasks = [*ALL_TASKS, easy]
    kept = build_sft_rows(JsonSnapshot(), tasks, ["FG-EASY"], CharTokenizer(), max_length=8192)
    assert kept["n_kept"] == 1
    assert kept["rollouts"][0].reward["total"] == pytest.approx(0.975)
    assert kept["rollouts"][0].reward["efficiency"] == pytest.approx(0.5)
    dropped = build_sft_rows(JsonSnapshot(), tasks, ["FG-EASY"], CharTokenizer(), max_length=8192, min_reward=1.0)
    assert dropped["n_kept"] == 0
    assert dropped["skipped"][0]["reason"] == "reward"


def test_sft_validation_holds_out_drugs_not_tasks():
    from fdagym.agents.baseline import load_split

    pool = [task for task in load_split("train") if task["type"] != "label_open"]
    drugs = sorted({task["entity"]["generic_name"] for task in pool})
    train, val = split_sft_by_drug(pool, fraction=0.05, seed=0)
    val_drugs = {task["entity"]["generic_name"] for task in val}
    train_drugs = {task["entity"]["generic_name"] for task in train}
    assert not (val_drugs & train_drugs)
    assert len(val_drugs) == round(0.05 * len(drugs))
    assert {task["id"] for task in train} | {task["id"] for task in val} == {task["id"] for task in pool}
    again, _ = split_sft_by_drug(pool, fraction=0.05, seed=0)
    assert [task["id"] for task in again] == [task["id"] for task in train]
    other, _ = split_sft_by_drug(pool, fraction=0.05, seed=1)
    assert [task["id"] for task in other] != [task["id"] for task in train]
    for drug in val_drugs:
        assert all(task["entity"]["generic_name"] == drug for task in val if task["entity"]["generic_name"] == drug)
        assert drug not in train_drugs


def test_sft_launch_wires_the_drug_holdout_as_eval():
    source = (ROOT / "fdagym" / "train" / "sft_launch.py").read_text()
    assert "split_sft_by_drug" in source
    assert "eval_dataset" in source
    assert "teacher_forced_tool_match" in source
    assert "eval_first_tool_exact" in source
    assert "eval_tool_seq_exact" in source
    assert "config.num_train_epochs" in source


def test_sft_skips_a_row_that_does_not_score_one():
    built = build_sft_rows(
        JsonSnapshot(),
        ALL_TASKS,
        ["FG-0002"],
        CharTokenizer(),
        limits=RolloutLimits(max_turns=1, temperature=0.0),
        max_length=8192,
    )
    assert built["n_kept"] == 0
    assert built["skipped"][0]["reason"] == "reward"


def test_tool_prompts_have_no_packed_documents():
    rows, mix = build_tool_rows(ALL_TASKS, tool_train_config(), QwenToolCodec())
    assert [row["task_id"] for row in rows] == ["FG-0001", "FG-0002", "FG-0003"]
    assert sum(mix.values()) == 3
    for row in rows:
        assert row["prompt"].endswith(ASSISTANT_TURN)
        assert "<tool_response>" not in row["prompt"]
        assert "0cbce382-9c88-4f58-ae0f-532a841e8f95" not in row["prompt"]


def test_missing_sft_adapter_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not fall back"):
        resolve_tool_start(tmp_path, tmp_path / "missing", resume=True)
    with pytest.raises(RuntimeError, match="must not run"):
        unused_reward()


def test_hub_adapter_is_a_valid_grpo_start(tmp_path):
    hub = "example/fdagym-adapter"
    start = resolve_tool_start(tmp_path, hub, resume=True)
    assert start == {"mode": "sft", "checkpoint": None, "sft_adapter": hub}
    with pytest.raises(FileNotFoundError, match="does not fall back"):
        resolve_tool_start(tmp_path, "runs/missing/adapter", resume=True)


def test_checkpoint_wins_and_resume_false_uses_sft(tmp_path):
    (tmp_path / "checkpoint-50").mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    resumed = resolve_tool_start(tmp_path, adapter, resume=True)
    assert resumed["mode"] == "resume"
    assert resumed["checkpoint"].endswith("checkpoint-50")
    fresh = resolve_tool_start(tmp_path, adapter, resume=False)
    assert fresh == {"mode": "sft", "checkpoint": None, "sft_adapter": str(adapter)}


def test_eval_dry_run_writes_a_schema_valid_trajectory(tmp_path):
    snapshot = JsonSnapshot()
    tasks = [task for task in ALL_TASKS if task["id"] in {"FG-0001", "FG-0002"}]
    tokenizer = CharTokenizer()
    codec = QwenToolCodec()
    summary = eval_tool_tasks(
        snapshot=snapshot,
        tasks=tasks,
        tokenizer=tokenizer,
        generator=OracleGenerator(tokenizer, codec, snapshot, tasks),
        model={"name": "oracle-dry", "kind": "trained", "checkpoint_step": None},
        run_id="eval-tools-dry",
        results_dir=tmp_path,
        codec=codec,
        limits=RolloutLimits(temperature=0.0),
        split="fixture",
    )
    assert summary["protocol"] == "tool_rollout"
    assert summary["eval_temperature"] == 0.0
    assert summary["models"]["oracle-dry"]["mean_reward"] == 1.0
    assert summary["dense_reward"]["n"] == 2
    assert summary["dense_reward"]["mean_reward"] == 1.0
    assert "strict" in summary["dense_reward"]["note"]
    lines = (tmp_path / "eval-tools-dry.jsonl").read_text().splitlines()
    assert len(lines) == 2
    import json

    row = json.loads(lines[0])
    assert row["reward"]["total"] == 1.0
    assert row["usage"]["prompt_tokens"] > 0
    assert row["usage"]["completion_tokens"] > 0
    assert row["model"]["kind"] == "trained"


def test_base_eval_is_the_raw_model_and_adapter_eval_is_trained():
    base = tool_eval_model(None)
    assert base == {"name": QWEN_3B, "kind": "baseline", "checkpoint_step": None}
    trained = tool_eval_model("/vol/runs/grpo-qwen3b-tools/adapter")
    assert trained["kind"] == "trained"
    assert trained["name"] == f"{QWEN_3B}+tools"
    assert tool_eval_model("")["kind"] == "baseline"


def test_loss_uses_loss_mask_and_attention_uses_completion_mask():
    source = inspect.getsource(ToolGRPOTrainer._compute_loss)
    assert 'inputs["loss_mask"]' in source
    assert "per_token_loss * token_mask" in source
    assert "per_token_kl * token_mask" in source
    assert "torch.cat([prompt_mask, completion_mask], dim=1)" in source
    assert "per_token_loss * completion_mask" not in source
    generate = inspect.getsource(ToolGRPOTrainer._generate_and_score_completions)
    assert "completion_mask" in generate
    assert "loss_mask" in generate
    assert "_sample_groups" in generate
    assert "rewards/hops/mean" in generate
    scored = inspect.getsource(ToolGRPOTrainer._rollout_scored)
    assert "episode_scores" in scored
    assert "self.dense_reward" in scored
    launch = inspect.getsource(inspect.getmodule(ToolGRPOTrainer).run_tool_grpo)
    assert "dense_reward=config.dense_reward" in launch
    env_call = (ROOT / "fdagym" / "env.py").read_text().split("self._reward = score(", 1)[1].split(")", 1)[0]
    assert "dense" not in env_call
    assert "use_liger_loss" in inspect.getsource(ToolGRPOTrainer.compute_loss)
    text = (ROOT / "fdagym" / "train" / "grpo_tools.py").read_text()
    assert "chunked_token_logps" in text
    assert "logprob_chunk_size=config.logprob_chunk_size" in text
    assert "_get_per_token_logps(" not in text
    assert "_get_per_token_logps_and_entropies" not in text
    assert "hf_smoke" in text
    assert "_generate_and_score_hf" in text


def test_tool_modules_import_without_torch():
    import importlib

    grpo_tools = importlib.import_module("fdagym.train.grpo_tools")
    hf_rollout = importlib.import_module("fdagym.train.hf_rollout")
    sft_launch = importlib.import_module("fdagym.train.sft_launch")
    assert grpo_tools.ToolGRPOTrainer is ToolGRPOTrainer
    assert hf_rollout.HFRolloutGenerator.__name__ == "HFRolloutGenerator"
    assert sft_launch.run_sft.__name__ == "run_sft"
    assert "torch" not in sys.modules
    assert "trl" not in sys.modules
    for relative in (
        "fdagym/train/grpo_tools.py",
        "fdagym/train/sft_launch.py",
        "fdagym/train/hf_rollout.py",
        "fdagym/train/sft_tools.py",
        "fdagym/train/eval_tools.py",
        "fdagym/train/jobs.py",
        "fdagym/train/gpu_stack.py",
    ):
        tree = ast.parse((ROOT / relative).read_text())
        imported = _top_level_imports(tree)
        assert "torch" not in imported
        assert "vllm" not in imported


def test_modal_tool_actions_are_a100_and_scripts_stay_offline():
    text = (ROOT / "modal_train.py").read_text()
    assert 'TOOL_GPU = "A100-80GB"' in text
    assert "def sft(" in text
    assert "def train_tools(" in text
    assert "def eval_tools(" in text
    assert "train-tools" in text
    assert "eval-tools-base" in text
    assert "baseline-qwen3b-tools" in text
    assert "adapter=None" in text
    assert "sft-qwen3b-tools" in text
    assert "gpu = TOOL_GPU" in text
    assert "eval-base" in text
    for name in ("sft_warmup.py", "grpo_tools.py", "eval_tools.py", "tool_rollout_dry.py"):
        script = (ROOT / "scripts" / name).read_text()
        assert "--dry-run" in script or name == "tool_rollout_dry.py"
        assert "import modal" not in script
    runbook = (ROOT / "docs" / "train.md").read_text()
    assert "modal run modal_train.py --action sft" in runbook
    assert "modal run modal_train.py --action train-tools" in runbook
    assert "modal run modal_train.py --action eval-tools-base" in runbook
    assert "modal run modal_train.py --action eval-tools" in runbook
    assert "150 steps" in runbook
    assert "$0.021" in runbook
    assert "$3.15" in runbook
    assert runbook.index("modal volume get") < runbook.index("compare_runs.py")


def _top_level_imports(tree: ast.AST) -> set[str]:
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    return imported
