"""Hard-RL Modal flags, SFT pool guard, and the unused hop-reward hook."""

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.rewards.combine import WEIGHTS
from fdagym.train.config import TrainConfig
from fdagym.train.gold import citations_from_outputs
from fdagym.train.grpo_tools import build_tool_rows, run_tool_grpo
from fdagym.train.hard_mix import (
    format_pool_exclusion_log,
    hard_mix_weights,
    pool_exclusion_proof,
    prove_training_pool,
    rollout_tasks,
)
from fdagym.train.rollout import QwenToolCodec
from fdagym.train.sft_config import SFTConfig
from fdagym.train.sft_select import assert_sft_rows_allowed
from fdagym.train.spawn_plan import spawn_plan
from fdagym.train.tool_score import episode_scores, hop_reward, training_reward

ROOT = Path(__file__).resolve().parents[2]


def test_hard_mix_weight_zero_is_the_old_path():
    assert hard_mix_weights(0) is None
    assert hard_mix_weights(0.0) is None
    weights = hard_mix_weights(0.6)
    assert weights["hard"] == pytest.approx(0.6)
    assert weights["current"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        hard_mix_weights(1)
    with pytest.raises(TypeError):
        hard_mix_weights(True)


def test_real_pool_excludes_holdout_and_hard_eval():
    proof = prove_training_pool(load_split("train"), load_split("hard_train"))
    assert proof["n_holdout"] == 34
    hard_present = (Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl").exists()
    assert proof["n_hard_eval"] == (100 if hard_present else len(load_split("public")))
    assert proof["leaked"] == []
    assert set(proof["holdout_ids"]).isdisjoint(proof["hard_eval_ids"])
    log = format_pool_exclusion_log(proof)
    assert '"n_holdout": 34' in log
    assert f'"n_hard_eval": {100 if hard_present else len(load_split("public"))}' in log
    assert proof["holdout_sha256"] in log
    assert proof["hard_eval_sha256"] in log
    assert proof["holdout_ids"][0] in log
    if hard_present:
        assert "FG-8001" in proof["hard_eval_ids"]


def test_mixed_grpo_env_owns_hard_train_ids():
    """Prompts can sample hard_train; FDAEnv.reset looks those ids up on the trainer."""
    current = load_split("train")
    hard = load_split("hard_train")
    rows, _mix = build_tool_rows(
        current,
        TrainConfig(),
        QwenToolCodec(prompt_version="v4", tool_budget=8),
        hard_train_mix=hard_mix_weights(0.6),
        hard_tasks=hard,
        mix_rows=20,
        mix_seed=0,
    )
    prompt_ids = {row["task_id"] for row in rows}
    hard_ids = {task["id"] for task in hard}
    sampled_hard = prompt_ids & hard_ids
    assert sampled_hard
    train_ids = {task["id"] for task in current}
    assert sampled_hard.isdisjoint(train_ids)
    env_ids = {task["id"] for task in rollout_tasks(current, hard)}
    assert prompt_ids <= env_ids
    assert hard_ids <= env_ids
    hard_file = Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl"
    if hard_file.exists():
        assert env_ids.isdisjoint(task["id"] for task in load_split("hard"))
    assert env_ids.isdisjoint(task["id"] for task in load_split("hard_train_holdout"))
    assert {task["id"] for task in rollout_tasks(current, None)} == train_ids
    source = inspect.getsource(run_tool_grpo)
    assert "rollout_tasks(" in source
    assert "tasks=env_tasks" in source


def test_pool_proof_fails_when_an_eval_id_is_in_the_input():
    holdout = [{"id": "FG-H", "split": "hard_train_holdout"}]
    hard_eval = [{"id": "FG-E", "split": "hard"}]
    current = [{"id": "FG-H", "split": "train", "type": "approval", "entity": {"generic_name": "x"}}]
    hard_train = [{"id": "FG-T", "split": "hard_train", "type": "compare"}]
    with pytest.raises(RuntimeError, match="training pool"):
        pool_exclusion_proof(current, hard_train, holdout, hard_eval)
    with pytest.raises(RuntimeError, match="empty"):
        pool_exclusion_proof(current, hard_train, [], hard_eval)


def test_sft_never_takes_holdout_or_hard_eval():
    assert SFTConfig().include_hard_train is False
    assert SFTConfig().min_reward == 0.97
    assert SFTConfig().tool_budget == 8
    assert SFTConfig().prompt_version == "v4"
    assert SFTConfig().model == "Qwen/Qwen2.5-3B-Instruct"
    assert SFTConfig().learning_rate == 1e-4
    assert SFTConfig().num_train_epochs == 1.0
    assert SFTConfig().per_device_train_batch_size == 4
    assert SFTConfig().gradient_accumulation_steps == 2
    holdout_id = load_split("hard_train_holdout")[0]["id"]
    with pytest.raises(RuntimeError, match="holdout"):
        assert_sft_rows_allowed(
            [{"id": holdout_id, "split": "train", "type": "approval"}],
            include_hard_train=True,
        )
    public_id = load_split("public")[0]["id"]
    with pytest.raises(RuntimeError, match="hard eval"):
        assert_sft_rows_allowed([{"id": public_id, "split": "train", "type": "approval"}], include_hard_train=True)
    hard_id = load_split("hard_train")[0]["id"]
    row = [{"id": hard_id, "split": "hard_train", "type": "compare"}]
    with pytest.raises(RuntimeError, match="hard_train"):
        assert_sft_rows_allowed(row, include_hard_train=False)
    assert_sft_rows_allowed(row, include_hard_train=True)


def test_compare_citations_use_the_answer_drug_not_the_first_call():
    steps = [
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "adalimumab"},
            "output": {"id": "step:1", "meta": {"results": {"total": 1}}, "results": []},
        },
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "oxycodone"},
            "output": {"id": "step:2", "meta": {"results": {"total": 9}}, "results": []},
        },
        {
            "tool": "query_adverse_events",
            "arguments": {"generic_name": "oxycodone"},
            "output": {"id": "step:3", "results": [{"term": "DRUG DEPENDENCE", "count": 4}]},
        },
    ]
    task = {
        "id": "T",
        "entity": {"generic_name": "oxycodone"},
        "gold_citations": [
            {"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]},
            {"endpoint": "drug/event", "ids": [], "fields": ["results"]},
        ],
    }
    cites = citations_from_outputs(steps, task)
    assert [cite["id"] for cite in cites] == ["step:2", "step:3"]
    single_task = {
        **task,
        "entity": {"generic_name": "adalimumab"},
        "gold_citations": [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}],
    }
    single = citations_from_outputs(steps[:1], single_task)
    assert [cite["id"] for cite in single] == ["step:1"]


def test_hop_reward_is_not_added():
    assert WEIGHTS == {"correctness": 0.60, "citation": 0.25, "format": 0.10, "efficiency": 0.05}
    source = inspect.getsource(episode_scores)
    assert "hop_reward" in source
    assert "hop_reward(" not in source
    assert hop_reward({}, {}, None) == 0.0
    assert training_reward(0.8, 1, 0.05) == pytest.approx(0.85)
    assert training_reward(0.8, 0, 0.05) == pytest.approx(0.8)


def test_spawn_plan_is_the_new_hard_run_and_does_not_import_modal():
    sft = spawn_plan("sft")
    assert sft["function"] == "sft"
    assert sft["kwargs"] == {
        "run_id": "sft-qwen3b-tools-v4",
        "tool_budget": 8,
        "prompt_version": "v4",
        "include_hard_train": False,
    }
    evaluation = spawn_plan("eval-hard")
    assert evaluation["function"] == "eval_tools"
    assert evaluation["kwargs"]["split"] == "hard"
    assert evaluation["kwargs"]["tool_budget"] == 8
    assert evaluation["kwargs"]["prompt_version"] == "v4"
    assert evaluation["kwargs"]["adapter"] == "/vol/runs/sft-qwen3b-tools-v4/adapter"
    dry = spawn_plan("grpo-dry")
    assert dry["kwargs"]["hard_train_mix"] == 0.6
    assert dry["kwargs"]["max_steps"] == 3
    assert dry["kwargs"]["tool_budget"] == 8
    assert dry["kwargs"]["prompt_version"] == "v4"
    assert dry["kwargs"]["memorization_every"] == 10
    with pytest.raises(ValueError):
        spawn_plan("train")
    script = (ROOT / "scripts" / "spawn_modal.py").read_text()
    tree = ast.parse(script)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "modal" not in imported
    assert "fn.spawn" in script


def test_modal_hard_path_logs_the_pool_and_wires_memorization():
    modal = (ROOT / "modal_train.py").read_text()
    jobs = (ROOT / "fdagym" / "train" / "jobs.py").read_text()
    grpo = (ROOT / "fdagym" / "train" / "grpo_tools.py").read_text()
    assert "format_pool_exclusion_log" in jobs
    assert "memorization=weights is not None" in jobs
    assert "type_mix=tool.type_mix" in jobs
    assert "run_train_tools(" in modal
    assert "include_hard_train: bool = False" in modal
    assert "memorization: bool = False" in grpo
    assert 'if memorization and measure_box["fn"] is None' in grpo
    assert "live_memorization_measure" in grpo
    assert "fdagym.swapped" not in grpo
    assert "SwappedFactsEnv" not in modal
    launcher = (ROOT / "scripts" / "launch_pipeline.py").read_text()
    spec = importlib.util.spec_from_file_location("launch_pipeline", ROOT / "scripts" / "launch_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parser().parse_args(["tools", "--init-adapter", "adapter", "--hard-train-mix", "0.6"])
    assert module._kwargs(args) == {"init_adapter": "adapter", "force": False, "hard_train_mix": 0.6}
    assert "modal" not in {alias.name.split(".")[0] for node in ast.parse(launcher).body if isinstance(node, ast.Import) for alias in node.names}


def test_v4_sft_audit_keeps_the_v3_counts():
    data = json.loads((ROOT / "results" / "sft-v4-oracle.json").read_text())
    assert data["prompt_version"] == "v4"
    assert data["tool_budget"] == 8
    assert data["min_reward"] == 0.97
    assert data["include_hard_train"] is False
    assert data["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert data["fit"]["n_kept"] == 1106
    assert data["fit"]["n_script_failures"] == 0
    assert data["fit"]["n_reward_dropped"] == 0
    assert data["drug_holdout"]["n_kept"] == 54
    assert data["drug_holdout"]["n_script_failures"] == 0
    assert data["bpe"]["default_rows_over_8192_tokens"] == 0
    assert data["bpe"]["default_max_tokens"] == 2967
    assert data["hard_train_opt_in"]["fit"]["n_reward_dropped"] == 0
    assert data["hard_train_opt_in"]["drug_holdout"]["n_reward_dropped"] == 0
    assert data["bpe"]["hard_train_rows_over_8192_tokens"] == 26
    example = data["example_row"]
    assert example["task_id"] == "FG-0343"
    assert example["tool_budget"] == 8
    assert example["prompt_version"] == "v4"
    assert example["reward_total"] == 1.0
    assert "- You get at most 8 tool calls, then you must submit_answer." in example["prompt"]
    assert "mode is top_reactions or count" in example["prompt"]
    assert example["tools"] == ["search_approvals", "submit_answer"]
