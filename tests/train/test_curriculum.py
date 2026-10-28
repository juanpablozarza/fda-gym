from copy import deepcopy

from fdagym.train.curriculum import STAGE_MIX, filter_pool, mix_counts, stage_pool
from tests.env.support import ALL_TASKS, EXAMPLE_TASK


def test_skips_label_open_and_test_split():
    extra = deepcopy(EXAMPLE_TASK)
    extra["id"] = "FG-open"
    extra["type"] = "label_open"
    extra["split"] = "train"
    pool = filter_pool([*ALL_TASKS, extra], stage=1)
    ids = {task["id"] for task in pool}
    assert "FG-open" not in ids
    assert "FG-0004" not in ids
    assert "FG-0005" not in ids
    assert "FG-0001" in ids


def test_drops_already_solved_when_pass_rates_exist():
    rows = []
    for task, rate in zip(ALL_TASKS, [1.0, 0.5, 0.0, None, 0.25]):
        row = deepcopy(task)
        row["base_pass_rate"] = rate
        row["split"] = "train"
        rows.append(row)
    stage1 = {task["id"] for task in filter_pool(rows, 1)}
    assert "FG-0001" not in stage1
    assert "FG-0002" in stage1
    assert "FG-0003" not in stage1
    stage3 = {task["id"] for task in filter_pool(rows, 3)}
    assert "FG-0003" in stage3
    assert "FG-0001" not in stage3


def test_stage_one_mix_fills_from_harder_tiers():
    selected, actual = stage_pool(ALL_TASKS, 1)
    assert actual["easy"] + actual["medium"] + actual["hard"] == len(selected)
    assert all(task.get("split") != "test" for task in selected)
    counts = mix_counts(10, STAGE_MIX[1])
    assert counts["easy"] + counts["medium"] + counts["hard"] == 10
    assert counts["easy"] >= counts["medium"] >= counts["hard"]
