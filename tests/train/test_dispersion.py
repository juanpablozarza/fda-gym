"""CPU checks for the reward-dispersion probe. No GPU and no Modal calls."""

import ast
import json
import statistics
from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.rewards import score
from fdagym.train import dispersion
from fdagym.train.dispersion import ids_sha256, probe_task_ids, run_probe, sampling_param_kwargs
from fdagym.train.group_sample import tied_group_flags
from fdagym.train.hard_mix import hard_mix_weights
from fdagym.train.rollout import QwenToolCodec, TurnGeneration, generate_with_vllm
from fdagym.train.rollout_dry import CharTokenizer, OracleGenerator
from fdagym.train.tool_config import configure_tool_run
from fdagym.train.tool_score import frac_zero_std, training_reward
from tests.env.support import ALL_TASKS, JsonSnapshot

ROOT = Path(__file__).resolve().parents[2]

# Draw order from the hard_train_mix=0.6 pool (seed 20261022, n=40).
# sha256("\n".join(ids)) with no trailing newline.
PROBE_IDS = [
    "FG-8405",
    "FG-8339",
    "FG-8109",
    "FG-8363",
    "FG-0943",
    "FG-0861",
    "FG-8184",
    "FG-0928",
    "FG-1206",
    "FG-1234",
    "FG-8149",
    "FG-8168",
    "FG-8446",
    "FG-1195",
    "FG-8301",
    "FG-8370",
    "FG-0864",
    "FG-1377",
    "FG-8333",
    "FG-8293",
    "FG-8315",
    "FG-0900",
    "FG-8447",
    "FG-8147",
    "FG-1080",
    "FG-1013",
    "FG-0875",
    "FG-8129",
    "FG-8180",
    "FG-8142",
    "FG-8453",
    "FG-0912",
    "FG-8258",
    "FG-8336",
    "FG-8276",
    "FG-1089",
    "FG-8247",
    "FG-8113",
    "FG-1306",
    "FG-1005",
]
PROBE_SHA = "51d68d3770e85bb49990468012f96b0c68f9b7e7a7d22408c37f993c416edfcf"
# Composite rows from that draw. They are no longer in the GRPO pool.
COMPOSITE_PROBE_IDS = ("FG-0943", "FG-0928")
# Easy train approval the hard-run filter drops, so it is not a pool row.
OUTSIDE_POOL = "FG-0343"


def test_probe_sha_matches_the_published_draw_order():
    assert ids_sha256(PROBE_IDS) == PROBE_SHA
    assert ids_sha256(list(reversed(PROBE_IDS))) != PROBE_SHA


def test_probe_task_ids_rejects_a_sha_mismatch():
    with pytest.raises(ValueError, match="sha256"):
        probe_task_ids(PROBE_IDS, "0" * 64)


def test_probe_task_ids_rejects_an_id_outside_the_pool():
    assert OUTSIDE_POOL not in PROBE_IDS
    with pytest.raises(ValueError, match="not in the GRPO training pool"):
        probe_task_ids([OUTSIDE_POOL], ids_sha256([OUTSIDE_POOL]))


def test_probe_task_ids_rejects_a_hard_eval_id():
    if not (Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl").exists():
        pytest.skip("data/tasks/hard.jsonl stays in the private repo")
    task_id = load_split("hard")[0]["id"]
    with pytest.raises(ValueError, match="eval, holdout, or test"):
        probe_task_ids([task_id], ids_sha256([task_id]))


def test_probe_task_ids_rejects_a_holdout_id():
    task_id = load_split("hard_train_holdout")[0]["id"]
    with pytest.raises(ValueError, match="eval, holdout, or test"):
        probe_task_ids([task_id], ids_sha256([task_id]))


def test_probe_task_ids_accepts_ids_from_the_current_pool():
    from fdagym.train.grpo_tools import build_tool_rows
    from fdagym.train.hard_mix import hard_mix_weights
    from fdagym.train.prompt_filter import blocked_eval_ids, hard_prompt_filter
    from fdagym.train.rollout import QwenToolCodec
    from fdagym.train.tool_config import tool_train_config

    rows, mix = build_tool_rows(
        load_split("train"),
        tool_train_config(),
        QwenToolCodec(),
        hard_train_mix=hard_mix_weights(0.6),
        hard_tasks=load_split("hard_train"),
        prompt_filter=hard_prompt_filter(),
        blocked_ids=blocked_eval_ids(),
        mix_rows=600,
        mix_seed=0,
    )
    assert "composite" not in mix["by_type"]["rows"]
    sample = [row["task_id"] for row in rows[:8]]
    tool, env_tasks, ids = probe_task_ids(sample, ids_sha256(sample))
    assert ids == sample
    assert tool.train.num_generations == 8
    assert tool.train.dense_reward is True
    assert tool.tool_parse_bonus == 0.05
    assert tool.limits.temperature == 0.65
    assert tool.limits.max_new_tokens == 256
    assert tool.limits.max_turns == 9
    assert tool.limits.max_tool_chars == 3500
    assert tool.limits.max_context_tokens == 8192
    assert tool.limits.tool_budget == 8
    assert tool.limits.prompt_version == "v4"
    assert tool.prompt_filter.enabled is True
    assert tool.pool_types == ("boxed_warning", "faers_top_reaction", "approval", "faers_count")
    env_ids = {task["id"] for task in env_tasks}
    assert set(sample) <= env_ids
    root = Path(__file__).resolve().parents[2] / "data" / "tasks"
    blocked = {
        task["id"]
        for name in ("hard", "public", "hard_train_holdout", "test")
        if (root / f"{name}.jsonl").exists()
        for task in load_split(name)
    }
    assert env_ids.isdisjoint(blocked)
    assert all(task.get("split") in {"train", "hard_train"} for task in env_tasks)


def test_probe_task_ids_rejects_composite():
    composite = COMPOSITE_PROBE_IDS[0]
    with pytest.raises(ValueError, match="not in the GRPO training pool"):
        probe_task_ids([composite], ids_sha256([composite]))
    with pytest.raises(ValueError, match="not in the GRPO training pool"):
        probe_task_ids(PROBE_IDS, PROBE_SHA)


def _tool():
    return configure_tool_run(
        hard=True,
        tool_budget=8,
        prompt_version="v4",
        hard_train_mix=hard_mix_weights(0.6),
    )


def _oracle(task_ids):
    snapshot = JsonSnapshot()
    tokenizer = CharTokenizer()
    codec = QwenToolCodec(prompt_version="v4", tool_budget=8)
    chosen = [task for task in ALL_TASKS if task["id"] in set(task_ids)]
    generator = OracleGenerator(tokenizer, codec, snapshot, chosen)
    return snapshot, tokenizer, generator


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_probe_writes_one_group_per_question_and_scores_with_episode_scores(tmp_path, monkeypatch):
    task_ids = ["FG-0001", "FG-0002"]
    snapshot, tokenizer, generator = _oracle(task_ids)
    seen = []
    batches = []
    real_scores = dispersion.episode_scores
    real_batch = dispersion.rollout_batch

    def spy_scores(rollout, task, snap, *, dense, bonus):
        seen.append((dense, bonus, task["id"]))
        return real_scores(rollout, task, snap, dense=dense, bonus=bonus)

    def spy_batch(*args, **kwargs):
        batches.append({"task_ids": args[2], "limits": args[6], "seed": kwargs["seed"]})
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(dispersion, "episode_scores", spy_scores)
    monkeypatch.setattr(dispersion, "rollout_batch", spy_batch)
    destination = tmp_path / "rollouts.jsonl"
    summary = run_probe(
        snapshot,
        ALL_TASKS,
        task_ids,
        tokenizer,
        generator,
        _tool(),
        k=2,
        seed=7,
        out_path=destination,
        chunk_groups=1,
        run_id="probe-test",
    )
    rows = _rows(destination)
    assert [row["task_id"] for row in rows] == ["FG-0001", "FG-0001", "FG-0002", "FG-0002"]
    assert [row["sample_index"] for row in rows] == [0, 1, 0, 1]
    assert [row["group_index"] for row in rows] == [0, 0, 1, 1]
    assert [row["seed"] for row in rows] == [7, 7, 7 + 10007, 7 + 10007]
    assert batches[0]["task_ids"] == ["FG-0001", "FG-0001"]
    assert batches[1]["task_ids"] == ["FG-0002", "FG-0002"]
    assert batches[0]["limits"].temperature == 0.65
    assert batches[0]["limits"].prompt_version == "v4"
    assert batches[0]["limits"].tool_budget == 8
    assert [batch["seed"] for batch in batches] == [7, 7 + 10007]
    assert seen and all(dense is True and bonus == 0.05 for dense, bonus, _task_id in seen)
    by_id = {task["id"]: task for task in ALL_TASKS}
    for row in rows:
        comp = score(
            {"steps": row["trajectory"]["steps"], "final_answer": row["trajectory"]["final_answer"]},
            by_id[row["task_id"]],
            dense=True,
            snapshot=snapshot,
        )
        assert row["shaped"] == training_reward(float(comp["total"]), int(row["parsed_tool_calls"]), 0.05)
        assert row["dense"] == float(comp["total"])
        assert row["strict"] == 1.0
        assert row["shaped"] == 1.05
        assert "notes" not in row["components"]
        assert row["components"]["total"] == comp["total"]
        assert row["components"]["correctness"] == comp["correctness"]
    assert summary == {
        "n_groups": 2,
        "n_episodes": 4,
        "frac_mixed_shaped": 0.0,
        "mean_within_std_shaped": 0.0,
    }
    shaped = [row["shaped"] for row in rows]
    assert summary["frac_mixed_shaped"] == 1.0 - frac_zero_std(shaped, 2)
    assert tied_group_flags(shaped, 2) == [True, True]
    assert "training_rollout(" not in (ROOT / "fdagym" / "train" / "dispersion.py").read_text(encoding="utf-8")


def test_timeout_mid_run_leaves_whole_groups(tmp_path, monkeypatch):
    task_ids = ["FG-0001", "FG-0002"]
    snapshot, tokenizer, generator = _oracle(task_ids)
    real_batch = dispersion.rollout_batch
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("simulated timeout")
        return real_batch(*args, **kwargs)

    commits = {"n": 0}
    monkeypatch.setattr(dispersion, "rollout_batch", wrapped)
    destination = tmp_path / "rollouts.jsonl"
    with pytest.raises(TimeoutError, match="simulated timeout"):
        run_probe(
            snapshot,
            ALL_TASKS,
            task_ids,
            tokenizer,
            generator,
            _tool(),
            k=2,
            seed=0,
            out_path=destination,
            chunk_groups=1,
            on_chunk=lambda: commits.__setitem__("n", commits["n"] + 1),
        )
    assert commits["n"] == 1
    rows = _rows(destination)
    assert len(rows) == 2
    assert len(rows) % 2 == 0
    assert {row["task_id"] for row in rows} == {"FG-0001"}
    assert [row["group_index"] for row in rows] == [0, 0]
    assert [row["sample_index"] for row in rows] == [0, 1]


class _SplitGenerator:
    """First episode follows the script. The other submits a wrong answer immediately."""

    def __init__(self, inner, tokenizer, codec):
        self.inner = inner
        self.tokenizer = tokenizer
        self.codec = codec
        self._opened = False

    def generate(self, prompt_token_ids, *, max_tokens, temperature):
        turns = self.inner.generate(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)
        if not self._opened and len(turns) == 2:
            self._opened = True
            bad = {"tool": "submit_answer", "arguments": {"answer": "NOPE", "citations": []}}
            body = self.tokenizer.encode(self.codec.action_text(bad))
            tokens = tuple(body[: max(max_tokens - 1, 0)] + [self.tokenizer.eos_token_id])
            turns[1] = TurnGeneration(tokens, tuple(-0.2 for _ in tokens))
        return turns


def test_run_probe_mixed_group_uses_the_tie_helpers(tmp_path):
    task_ids = ["FG-0001"]
    snapshot, tokenizer, oracle = _oracle(task_ids)
    codec = QwenToolCodec(prompt_version="v4", tool_budget=8)
    generator = _SplitGenerator(oracle, tokenizer, codec)
    destination = tmp_path / "rollouts.jsonl"
    summary = run_probe(
        snapshot,
        ALL_TASKS,
        task_ids,
        tokenizer,
        generator,
        _tool(),
        k=2,
        seed=0,
        out_path=destination,
        chunk_groups=10,
    )
    rows = _rows(destination)
    shaped = [row["shaped"] for row in rows]
    assert len(set(shaped)) == 2
    assert summary["n_groups"] == 1
    assert summary["n_episodes"] == 2
    assert summary["frac_mixed_shaped"] == 1.0
    assert summary["frac_mixed_shaped"] == 1.0 - frac_zero_std(shaped, 2)
    assert tied_group_flags(shaped, 2) == [False]
    assert summary["mean_within_std_shaped"] == statistics.pstdev(shaped)


def test_recorded_sampling_kwargs_match_generate_with_vllm():
    source = Path(generate_with_vllm.__code__.co_filename).read_text(encoding="utf-8")
    assert "sampling_params_cls(n=1, temperature=temperature, max_tokens=max_tokens, logprobs=1)" in source
    assert sampling_param_kwargs(0.8, 256) == {"n": 1, "temperature": 0.8, "max_tokens": 256, "logprobs": 1}
    assert "top_p" not in sampling_param_kwargs(0.8, 256)
    assert "repetition_penalty" not in sampling_param_kwargs(0.8, 256)


def test_probe_dispersion_uses_vllm_generator_and_the_spawn_signature():
    source = (ROOT / "modal_train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "probe_dispersion")
    names = [arg.arg for arg in function.args.args]
    assert names == [
        "adapter",
        "task_ids",
        "task_ids_sha256",
        "run_id",
        "k",
        "seed",
        "hard_train_mix",
        "tool_budget",
        "prompt_version",
        "temperature",
        "pool_types",
    ]
    defaults = [ast.literal_eval(default) for default in function.args.defaults]
    assert defaults == [8, 0, 0.6, 8, "v4", 0.65, "boxed_warning,faers_top_reaction,approval,faers_count"]
    decorator = ast.get_source_segment(source, function.decorator_list[0])
    assert decorator is not None
    assert "gpu=TOOL_GPU" in decorator
    assert "timeout=10 * 60" in decorator
    assert "memory=32768" in decorator
    assert "retries=0" in decorator
    assert "secrets" not in decorator
    assert "WANDB" not in decorator
    body = ast.get_source_segment(source, function)
    assert body is not None
    assert "run_probe_dispersion(" in body
    assert "HFRolloutGenerator" not in body
    assert "load_hf_rollout" not in body
    assert "llm.generate" not in body
    assert "start_wandb" not in body
    assert "hf_token" not in body
    job_src = (ROOT / "fdagym" / "train" / "jobs.py").read_text(encoding="utf-8")
    job_tree = ast.parse(job_src)
    job = next(node for node in job_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_probe_dispersion")
    job_body = ast.get_source_segment(job_src, job)
    assert job_body is not None
    assert "VLLMGenerator(" in job_body
    assert "SamplingParams(" in job_body
    assert "HFRolloutGenerator" not in job_body
    assert "load_hf_rollout" not in job_body
    assert "llm.generate" not in job_body
    assert "start_wandb" not in job_body
    assert "hf_token" not in job_body
    cli = (ROOT / "scripts" / "run_gpu.py").read_text(encoding="utf-8")
    assert "run_probe_dispersion(" in cli
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_src = ast.get_source_segment(source, main)
    assert main_src is not None
    assert "probe_dispersion" not in main_src
    module = (ROOT / "fdagym" / "train" / "dispersion.py").read_text(encoding="utf-8")
    tree = ast.parse(module)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported
    assert "vllm" not in imported
    assert "modal" not in imported
    assert "wandb" not in imported
