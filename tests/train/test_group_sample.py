"""Zero-variance groups, the 4×8 hard batch, and the current-prompt filter."""

import random
from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.train.curriculum import parse_pool_types
from fdagym.train.group_sample import (
    GROUP_COUNT_METRICS,
    HARD_MAX_RESAMPLE_ATTEMPTS,
    HARD_PROMPTS_PER_STEP,
    PromptPool,
    collapse_group_counts,
    example_mask_for,
    fill_zero_std_groups,
    grouped_prompt_ids,
    optimizer_batch,
    resolve_hard_signal,
    tied_group_flags,
    weighted_mean,
)
from fdagym.train.grpo_tools import build_tool_rows
from fdagym.train.prompt_filter import (
    PromptFilter,
    blocked_eval_ids,
    filter_current_tasks,
    hard_prompt_filter,
)
from fdagym.train.rollout import QwenToolCodec
from fdagym.train.tool_config import ToolTrainConfig, configure_tool_run, tool_train_config

ROOT_METRICS = (
    "groups/dropped",
    "groups/resampled",
    "groups/resample_rounds",
    "groups/masked",
    "groups/trained",
    "groups/initial_frac_zero_std",
)


def test_pool_types_allowlist_defaults_without_composite():
    assert parse_pool_types(None) == ("boxed_warning", "faers_top_reaction", "approval", "faers_count")
    assert parse_pool_types("boxed_warning,composite") == ("boxed_warning", "composite")
    with pytest.raises(ValueError, match="at least one"):
        parse_pool_types("")
    hot = configure_tool_run(hard=True, temperature=0.8)
    assert hot.train.temperature == 0.8
    assert hot.limits.temperature == 0.8


def test_hard_batch_is_four_groups_accumulated_not_one_wide_forward():
    batch = optimizer_batch(8, 4)
    assert batch == {
        "num_generations": 8,
        "prompts_per_step": 4,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "generation_batch_size": 8,
    }
    assert HARD_PROMPTS_PER_STEP == 4
    assert HARD_MAX_RESAMPLE_ATTEMPTS == 3
    with pytest.raises(ValueError):
        optimizer_batch(8, 0)
    with pytest.raises(ValueError):
        optimizer_batch(1, 4)
    plain = configure_tool_run(hard=False)
    assert plain.prompts_per_step == 1
    assert plain.dynamic_sampling is False
    assert plain.prompt_filter.enabled is False
    assert plain.train.gradient_accumulation_steps == 1
    assert plain.train.per_device_train_batch_size == 8
    hard = configure_tool_run(hard=True, hard_train_mix={"hard": 0.6, "current": 0.4})
    assert hard.prompts_per_step == 4
    assert hard.dynamic_sampling is True
    assert hard.max_resample_attempts == 3
    assert hard.prompt_filter.enabled is True
    assert hard.prompt_filter.protect_types == ("boxed_warning",)
    assert hard.pool_types == ("boxed_warning", "faers_top_reaction", "approval", "faers_count")
    kwargs = hard.train.grpo_kwargs("/tmp/out")
    assert kwargs["num_generations"] == 8
    assert kwargs["per_device_train_batch_size"] == 8
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["generation_batch_size"] == 8
    assert hard.train.logprob_chunk_size == 256
    assert hard.train.vllm_gpu_memory_utilization == 0.30
    assert ToolTrainConfig().train.gradient_accumulation_steps == tool_train_config().gradient_accumulation_steps == 1
    forced = configure_tool_run(hard=True, prompts_per_step=1, dynamic_sampling=False, difficulty_filter=False)
    assert forced.prompts_per_step == 1
    assert forced.dynamic_sampling is False
    assert forced.prompt_filter.enabled is False
    assert forced.train.gradient_accumulation_steps == 1


def test_signal_defaults_follow_the_hard_mix():
    auto = resolve_hard_signal(
        True,
        prompts_per_step=None,
        dynamic_sampling=None,
        difficulty_filter=None,
        max_resample_attempts=None,
    )
    assert auto == {
        "prompts_per_step": 4,
        "dynamic_sampling": True,
        "difficulty_filter": True,
        "max_resample_attempts": 3,
    }
    off = resolve_hard_signal(
        False,
        prompts_per_step=None,
        dynamic_sampling=None,
        difficulty_filter=None,
        max_resample_attempts=None,
    )
    assert off["prompts_per_step"] == 1
    assert off["dynamic_sampling"] is False
    assert off["difficulty_filter"] is False
    overridden = resolve_hard_signal(
        True,
        prompts_per_step=2,
        dynamic_sampling=False,
        difficulty_filter=None,
        max_resample_attempts=0,
    )
    assert overridden["prompts_per_step"] == 2
    assert overridden["dynamic_sampling"] is False
    assert overridden["difficulty_filter"] is True
    assert overridden["max_resample_attempts"] == 0
    with pytest.raises(ValueError):
        resolve_hard_signal(
            True,
            prompts_per_step=True,
            dynamic_sampling=None,
            difficulty_filter=None,
            max_resample_attempts=None,
        )


def test_tied_flags_and_the_loss_mean_skip_a_masked_group():
    assert tied_group_flags([1.05] * 8, 8) == [True]
    assert tied_group_flags([1.05] * 8 + [1.05, 0.4] * 4, 8) == [True, False]
    assert example_mask_for([True, False], 2, enabled=True) == [0, 0, 1, 1]
    assert example_mask_for([True], 2, enabled=False) == [1, 1]
    # A zero-weight group must not pull the mean from 1.0 down to 2/3.
    assert weighted_mean([1.0, 1.0, 0.0], [1.0, 1.0, 0.0]) == 1.0
    assert weighted_mean([0.2, 0.4], [1.0, 1.0]) == pytest.approx(0.3)
    assert weighted_mean([1.0, 1.0], [0.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        grouped_prompt_ids(["A", "B"], 2)


def _rollout(patterns):
    """patterns[task_id] is a list of per-member rewards, cycled if the group is longer."""

    calls = []

    def rollout_fn(ids, round_index):
        calls.append((list(ids), round_index))
        rewards = []
        for index, task_id in enumerate(ids):
            pattern = patterns[task_id]
            rewards.append(pattern[index % len(pattern)])
        return list(ids), rewards

    return rollout_fn, calls


def test_dynamic_sampling_replaces_a_tied_group_and_keeps_a_varied_one():
    rollout_fn, calls = _rollout({"A": [1.0, 1.0], "B": [1.0, 0.0], "C": [0.2, 0.9]})
    pool = PromptPool(["C"], blocked=set())
    result = fill_zero_std_groups(
        ["A", "A", "B", "B"],
        2,
        rollout_fn,
        pool,
        enabled=True,
        max_attempts=3,
        rng=random.Random(0),
    )
    assert calls[0] == (["A", "A", "B", "B"], 0)
    assert calls[1][0] == ["C", "C"]
    assert calls[1][1] == 1
    assert result.task_ids == ["C", "C", "B", "B"]
    assert result.example_mask == [1, 1, 1, 1]
    assert result.stats["dropped"] == 1
    assert result.stats["resampled"] == 1
    assert result.stats["resample_rounds"] == 1
    assert result.stats["masked"] == 0
    assert result.stats["trained"] == 2
    assert result.stats["initial_zero_std"] == 1


def test_cap_masks_a_group_that_stays_tied_and_off_leaves_it_alone():
    rollout_fn, calls = _rollout({"A": [1.0, 1.0]})
    pool = PromptPool(["A"], blocked=set())
    capped = fill_zero_std_groups(
        ["A", "A"],
        2,
        rollout_fn,
        pool,
        enabled=True,
        max_attempts=2,
        rng=random.Random(0),
    )
    assert len(calls) == 3
    assert [round_index for _ids, round_index in calls] == [0, 1, 2]
    assert capped.stats == {
        "initial_zero_std": 1,
        "dropped": 1,
        "resampled": 2,
        "resample_rounds": 2,
        "masked": 1,
        "trained": 0,
        "n_groups": 1,
    }
    assert capped.example_mask == [0, 0]
    rollout_fn, calls = _rollout({"A": [1.0, 1.0], "B": [0.0, 1.0]})
    left = fill_zero_std_groups(
        ["A", "A"],
        2,
        rollout_fn,
        PromptPool(["B"], blocked=set()),
        enabled=False,
        max_attempts=3,
        rng=random.Random(0),
    )
    assert calls == [(["A", "A"], 0)]
    assert left.stats["dropped"] == 0
    assert left.stats["resampled"] == 0
    assert left.stats["masked"] == 0
    assert left.example_mask == [1, 1]
    assert left.task_ids == ["A", "A"]


def test_resample_pool_rejects_eval_holdout_and_test_ids():
    blocked = blocked_eval_ids()
    public_id = load_split("public")[0]["id"]
    assert public_id in blocked
    holdout = load_split("hard_train_holdout")[0]["id"]
    test_id = load_split("test")[0]["id"]
    assert holdout in blocked and test_id in blocked
    with pytest.raises(RuntimeError, match="eval"):
        PromptPool([public_id, holdout, test_id], blocked)
    pool = PromptPool(["FG-0343"], blocked)
    assert pool.draw(random.Random(0), avoid=set()) == "FG-0343"


def test_collapse_sums_group_counts_for_one_optimizer_step():
    metrics = {
        "groups/dropped": [1, 0, 1, 0],
        "groups/resampled": [1, 0, 2, 0],
        "groups/resample_rounds": [1, 0, 2, 0],
        "groups/masked": [0, 0, 1, 0],
        "groups/trained": [1, 1, 0, 1],
        "frac_reward_zero_std": [1.0, 0.0, 0.0, 0.0],
    }
    collapse_group_counts(metrics)
    assert metrics["groups/dropped"] == [2]
    assert metrics["groups/resampled"] == [3]
    assert metrics["groups/masked"] == [1]
    assert metrics["groups/trained"] == [3]
    assert metrics["frac_reward_zero_std"] == [1.0, 0.0, 0.0, 0.0]
    assert set(GROUP_COUNT_METRICS) == set(ROOT_METRICS) - {"groups/initial_frac_zero_std"}


def _task(task_id, kind, difficulty, split="train", rate=None):
    return {
        "id": task_id,
        "split": split,
        "type": kind,
        "difficulty": difficulty,
        "base_pass_rate": rate,
    }


def test_filter_drops_solved_or_failed_and_keeps_protected_types():
    tasks = [
        _task("A", "approval", "medium", rate=1.0),
        _task("B", "boxed_warning", "easy", rate=1.0),
        _task("C", "composite", "hard", rate=0.0),
        _task("D", "faers_count", "hard", rate=0.0),
        _task("E", "approval", "medium", rate=0.5),
        _task("F", "approval", "easy"),
        _task("G", "approval", "easy", rate=0.4),
    ]
    kept, report = filter_current_tasks(tasks, hard_prompt_filter(), blocked_ids=set())
    assert [task["id"] for task in kept] == ["B", "E", "G"]
    assert report["dropped_by_type"] == {"approval": 2, "composite": 1, "faers_count": 1}
    assert report["kept_by_type"]["boxed_warning"] == 1
    assert "composite" not in report["kept_by_type"]
    with pytest.raises(ValueError, match="protected floors"):
        PromptFilter(protect_types=("boxed_warning", "composite"), protected_floor=0.6)


def test_filter_refuses_eval_holdout_test_and_hard_train_rows():
    if not (Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl").exists():
        pytest.skip("data/tasks/hard.jsonl stays in the private repo")
    config = hard_prompt_filter()
    blocked = blocked_eval_ids()
    with pytest.raises(RuntimeError, match="FG-8001"):
        filter_current_tasks([_task("FG-8001", "approval", "medium")], config, blocked_ids=blocked)
    for split in ("test", "hard", "hard_train", "hard_train_holdout"):
        with pytest.raises(RuntimeError, match="refuses"):
            filter_current_tasks([_task("X", "approval", "medium", split=split)], config, blocked_ids=set())


def test_real_hard_prompts_keep_the_mix_and_the_regressed_types():
    if not (Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl").exists():
        pytest.skip("data/tasks/hard.jsonl stays in the private repo")
    blocked = blocked_eval_ids()
    rows, mix = build_tool_rows(
        load_split("train"),
        tool_train_config(),
        QwenToolCodec(),
        hard_train_mix={"hard": 0.6, "current": 0.4},
        hard_tasks=load_split("hard_train"),
        prompt_filter=hard_prompt_filter(),
        blocked_ids=blocked,
        mix_rows=600,
        mix_seed=0,
    )
    assert mix["rows"] == {"hard": 360, "current": 240}
    assert mix["filter"]["n_dropped"] == 432
    assert mix["filter"]["dropped_by_type"] == {"approval": 432}
    assert "composite" not in mix["filter"]["kept_by_type"]
    assert mix["filter"]["kept_by_type"]["boxed_warning"] == 106
    assert mix["pool_types"]["allowlist"] == ["boxed_warning", "faers_top_reaction", "approval", "faers_count"]
    assert mix["pool_types"]["dropped_by_type"]["composite"] == 69
    assert mix["floor"]["target"] == {"boxed_warning": 60}
    assert mix["floor"]["short"] == {"boxed_warning": 0}
    assert mix["by_type"]["rows"] == {
        "approval": 11,
        "boxed_warning": 60,
        "compare": 222,
        "faers_count": 39,
        "faers_top_reaction": 130,
        "sponsor_hop": 138,
    }
    assert "composite" not in mix["by_type"]["rows"]
    assert mix["by_type"]["splits"] == {"hard_train": 360, "train": 240}
    by_id = {task["id"]: task for task in [*load_split("train"), *load_split("hard_train")]}
    prompt_ids = [row["task_id"] for row in rows]
    assert set(prompt_ids).isdisjoint(blocked)
    hard_file = Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl"
    if hard_file.exists():
        assert set(prompt_ids).isdisjoint(task["id"] for task in load_split("hard"))
    assert set(prompt_ids).isdisjoint(task["id"] for task in load_split("hard_train_holdout"))
    assert set(prompt_ids).isdisjoint(task["id"] for task in load_split("test"))
    for task_id in prompt_ids:
        task = by_id[task_id]
        if task["split"] == "train" and task["type"] != "boxed_warning":
            assert task["difficulty"] != "easy"
        assert task["type"] != "composite"
    kinds = mix["by_type"]["rows"]
    assert kinds["boxed_warning"] >= 60
    assert kinds["compare"] + kinds["sponsor_hop"] == 360


def test_trainer_logs_dropped_and_resampled_groups():
    root = Path(__file__).resolve().parents[2]
    source = (root / "fdagym" / "train" / "grpo_tools.py").read_text(encoding="utf-8")
    counters = (root / "fdagym" / "train" / "group_sample.py").read_text(encoding="utf-8")
    for key in ROOT_METRICS:
        assert key in counters or key in source
    assert "fill_zero_std_groups(" in source
    assert "collapse_group_counts(" in source
    assert "example_mask" in source
    jobs = (root / "fdagym" / "train" / "jobs.py").read_text(encoding="utf-8")
    modal = (root / "modal_train.py").read_text(encoding="utf-8")
    assert "configure_tool_run(" in jobs
    assert "run_train_tools(" in modal
    assert "prompts_per_step: int | None = None" in modal
    assert "dynamic_sampling: bool | None = None" in modal
    assert "difficulty_filter: bool | None = None" in modal
