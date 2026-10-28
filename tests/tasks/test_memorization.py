"""Known-drug holdout and memorization metrics. Rewards and the training pool stay put."""

import ast
import inspect
from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.env import FDAEnv
from fdagym.memorization import (
    MemorizationThresholds,
    attach_memorization,
    looked,
    memorization_report,
    record_training_memorization,
)
from fdagym.swapped import SWAP_N, SwappedFactsEnv, SwappedFactsError, fact_swap, select_fact_swaps
from fdagym.tasks.hard_solver import parse_question
from fdagym.tasks.holdout import assert_question_holdout, carve_known_holdout
from fdagym.train.curriculum import in_training_pool
from fdagym.train.hard_mix import mix_hard_train
from fdagym.train.rollout import training_rollout
from fdagym.train.tool_config import ToolTrainConfig

ROOT = Path(__file__).resolve().parents[2]


def _traj(correct: bool, total: float, steps: list[dict], answer):
    return {
        "reward": {"total": total, "correctness": 1.0 if correct else 0.0},
        "steps": steps,
        "final_answer": {"answer": answer, "citations": []},
    }


def _task(kind: str, value, citations=None):
    return {"id": "FG-0001", "gold": {"kind": kind, "value": value}, "gold_citations": citations or []}


def test_answered_without_looking_uses_hidden_gold_and_citation_ids():
    number = _task("number", 1621, [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}])
    counted = {"tool": "query_adverse_events", "output": {"meta": {"results": {"total": 1621}}, "results": []}}
    year_only = {"tool": "search_approvals", "output": {"results": [{"submission_status_date": "2023-01-02"}]}}
    assert looked(number, _traj(True, 1, [counted], 1621))
    assert not looked(number, _traj(True, 1, [year_only], 1621))

    label = _task(
        "exact",
        "divalproex",
        [{"endpoint": "drug/label", "ids": ["abc-set"], "fields": ["boxed_warning"]}],
    )
    record = {"tool": "search_drug_label", "output": {"results": [{"set_id": "abc-set", "sections": ["boxed_warning"]}]}}
    assert looked(label, _traj(True, 1, [record], "divalproex"))
    assert not looked(label, _traj(True, 1, [{"tool": "submit_answer", "output": {"set_id": "abc-set"}}], "divalproex"))

    boolean = _task("boolean", False, [{"endpoint": "drug/label", "ids": ["abc-set"], "fields": ["sections"]}])
    noise = {"tool": "search_drug_label", "output": {"error": "false", "note": "no boxed warning"}}
    assert not looked(boolean, _traj(True, 1, [noise], "no"))
    assert looked(boolean, _traj(True, 1, [record], "no"))

    faers = _task("exact", "amoxicillin", [{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}])
    faers["entity"] = {"generic_name": "amoxicillin"}
    own = {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": "amoxicillin", "mode": "count"},
        "output": {"meta": {"results": {"total": 12}}, "results": []},
    }
    other = {
        "tool": "query_adverse_events",
        "arguments": {"generic_name": "simvastatin", "mode": "count"},
        "output": {"meta": {"results": {"total": 99}}, "results": []},
    }
    assert looked(faers, _traj(True, 1, [own], "amoxicillin"))
    assert not looked(faers, _traj(True, 1, [other], "amoxicillin"))

    reaction = _task("exact", "FATIGUE", [{"endpoint": "drug/event", "ids": [], "fields": ["results"]}])
    term = {"tool": "query_adverse_events", "output": {"results": [{"term": "FATIGUE", "count": 4}]}}
    assert looked(reaction, _traj(True, 1, [term], "FATIGUE"))
    short = _task("exact", "no")
    assert not looked(short, _traj(True, 1, [noise], "no"))


def test_thresholds_warn_and_the_problem_flag_does_not_stop():
    known_task = _task("exact", "kept")
    unseen_task = _task("exact", "other")
    known = [(known_task, _traj(True, 1.0, [], "kept"))]
    unseen = [(unseen_task, _traj(True, 0.8, [], "other"))]
    report = memorization_report(known=known, unseen=unseen, swapped=[], thresholds=MemorizationThresholds())
    assert report["gap"]["strict"] == pytest.approx(0.2)
    assert report["answered_without_looking"]["rate"] == 1.0
    assert report["flags"]["gap"] == "warn"
    assert report["flags"]["answered_without_looking"] == "warn"
    assert report["flags"]["level"] == "stop and review"
    assert report["auto_stop"] is False
    assert report["mean_tool_calls"] == {"known": 0.0, "unseen": 0.0}

    followed = _task("exact", "HEADACHE")
    swapped = [(followed, _traj(False, 0.0, [], "NAUSEA"), "NAUSEA")]
    mild = memorization_report(
        known=[(known_task, _traj(True, 1.0, [{"tool": "query_adverse_events", "output": {"note": "kept"}}], "kept"))],
        unseen=[(unseen_task, _traj(True, 1.0, [{"tool": "query_adverse_events", "output": {"note": "other"}}], "other"))],
        swapped=swapped,
    )
    assert mild["gap"]["strict"] == 0.0
    assert mild["flags"]["level"] == "ok"
    assert mild["swapped_facts"]["follow_rate"] == 1.0

    half = memorization_report(
        known=[(known_task, _traj(True, 1.0, [{"tool": "x", "output": {"note": "kept"}}], "kept"))],
        unseen=[(unseen_task, _traj(True, 1.0, [{"tool": "x", "output": {"note": "other"}}], "other"))],
        swapped=[
            (followed, _traj(False, 0.0, [], "NAUSEA"), "NAUSEA"),
            (followed, _traj(True, 1.0, [], "HEADACHE"), "NAUSEA"),
        ],
    )
    assert half["swapped_facts"]["follow_rate"] == 0.5
    assert half["swapped_facts"]["original_rate"] == 0.5
    assert half["flags"]["swapped_facts"] == "warn"
    assert half["flags"]["level"] == "warn"
    assert half["auto_stop"] is False


def test_training_callback_logs_every_10_steps_and_does_not_stop():
    from fdagym.train.grpo_tools import ToolGRPOTrainer
    from fdagym.train.wandb_log import attach_grpo_train_logs, merge_memorization

    seen = {}

    def measure(step):
        seen["step"] = step
        seen["calls"] = seen.get("calls", 0) + 1
        return {"flags": {"level": "stop and review"}, "auto_stop": False}

    row = {"step": 10, "mean_reward": 0.4, "stage": 1}
    assert record_training_memorization(dict(row), 9, measure, every=10) == row
    assert "step" not in seen
    logged = record_training_memorization(dict(row), 10, measure, every=10)
    assert logged["memorization"]["flags"]["level"] == "stop and review"
    assert logged["memorization"]["auto_stop"] is False
    assert seen["step"] == 10
    again = record_training_memorization({"step": 20}, 20, measure, every=10)
    assert again["memorization"]["auto_stop"] is False
    assert seen["step"] == 20
    assert "should_training_stop" not in inspect.getsource(record_training_memorization)
    assert "should_training_stop" not in inspect.getsource(merge_memorization)

    calls_before = seen["calls"]
    off = {"reward": 0.4}
    attach_grpo_train_logs(off, step=9, measure=measure, every=10)
    assert seen["calls"] == calls_before
    assert "memorization/flags/level_name" not in off
    assert "memorization/auto_stop" not in off
    control = type("Control", (), {"should_training_stop": False})()
    on = {"reward": 0.4}
    attach_grpo_train_logs(on, step=10, measure=measure, every=10, control=control)
    assert on["memorization/flags/level_name"] == "stop and review"
    assert on["memorization/flags/level"] == 2
    assert on["memorization/auto_stop"] is False
    assert control.should_training_stop is False

    grpo = (ROOT / "fdagym" / "train" / "grpo_tools.py").read_text()
    wandb = (ROOT / "fdagym" / "train" / "wandb_log.py").read_text()
    assert "record_training_memorization" in wandb
    assert "attach_grpo_train_logs" in grpo
    assert "memorization_every: int = 10" in grpo
    log_source = inspect.getsource(ToolGRPOTrainer.log)
    assert log_source.index("attach_grpo_train_logs") < log_source.index("super().log")
    assert ToolTrainConfig().memorization.every_n_steps == 10
    assert ToolTrainConfig().memorization.gap_warn == 0.15
    assert ToolTrainConfig().memorization.no_look_warn == 0.20
    assert ToolTrainConfig().memorization.follow_warn == 0.70


def test_summary_keeps_existing_fields(tmp_path):
    from fdagym.train.eval_tools import eval_tool_tasks
    from fdagym.train.rollout import QwenToolCodec, RolloutLimits
    from fdagym.train.rollout_dry import CharTokenizer, OracleGenerator
    from tests.env.support import ALL_TASKS, JsonSnapshot

    tasks = [task for task in ALL_TASKS if task["id"] in {"FG-0001", "FG-0002"}]
    limits = RolloutLimits(temperature=0.0, tool_budget=6, prompt_version="v3")
    codec = QwenToolCodec(prompt_version="v3", tool_budget=6)
    report = {"flags": {"level": "ok"}, "auto_stop": False, "gap": {"strict": 0.0}}
    summary = eval_tool_tasks(
        snapshot=JsonSnapshot(),
        tasks=tasks,
        tokenizer=CharTokenizer(),
        generator=OracleGenerator(CharTokenizer(), codec, JsonSnapshot(), tasks),
        model={"name": "oracle-dry", "kind": "trained", "checkpoint_step": None},
        run_id="memo-summary",
        results_dir=tmp_path,
        codec=codec,
        limits=limits,
        split="fixture",
        memorization=report,
    )
    assert summary["models"]["oracle-dry"]["mean_reward"] == 1.0
    assert summary["protocol"] == "tool_rollout"
    assert summary["memorization"] == report
    stored = (tmp_path / "memo-summary" / "summary.json").read_text()
    assert '"memorization"' in stored
    assert '"mean_reward"' in stored
    copied = attach_memorization({"models": summary["models"]}, report)
    assert "memorization" not in summary or summary["memorization"] == report
    assert copied["models"] == summary["models"]


def test_holdout_questions_never_enter_the_mix(disk):
    train = load_split("hard_train")
    holdout = load_split("hard_train_holdout")
    known = assert_question_holdout(train, holdout, disk)
    assert known["holdout"] == 34
    assert known["holdout_by_variant"]["sponsor_count"] == 5
    assert known["train_drugs"] == 60
    assert known["holdout_drugs"] == 47
    assert known["holdout_sponsors"] <= known["train_sponsors"]
    _again_train, again_hold = carve_known_holdout([*train, *holdout], snapshot=disk)
    assert [task["id"] for task in again_hold] == [task["id"] for task in holdout]
    assert all(not in_training_pool(task) for task in holdout)
    rewritten = dict(holdout[0])
    rewritten["split"] = "train"
    rewritten["type"] = "approval"
    assert not in_training_pool(rewritten)
    with pytest.raises(RuntimeError, match="eval split"):
        mix_hard_train(load_split("train"), holdout, n_rows=10)
    relabeled = dict(holdout[0])
    relabeled["split"] = "hard_train"
    relabeled["type"] = "compare"
    with pytest.raises(RuntimeError, match="eval split"):
        mix_hard_train(load_split("train"), [relabeled, *train[:3]], n_rows=10)
    rows, _report = mix_hard_train(load_split("train"), train, n_rows=20, seed=1)
    assert {task["id"] for task in rows}.isdisjoint(task["id"] for task in holdout)


def test_swapped_facts_env_cannot_run_inside_training(disk):
    task = next(row for row in load_split("hard_train_holdout") if parse_question(row["question"])["variant"] == "sponsor_count")
    plan = fact_swap(task)
    assert plan.swapped == plan.original + 17
    plans = select_fact_swaps(load_split("hard_train_holdout"))
    assert len(plans) == SWAP_N
    with training_rollout():
        with pytest.raises(SwappedFactsError, match="eval-only"):
            SwappedFactsEnv(disk, [task], {task["id"]: plan}, tool_budget=8)
        env = FDAEnv(disk, [task], tool_budget=8)
        env.reset(task["id"])
    for relative in (
        "fdagym/train/grpo_tools.py",
        "fdagym/train/rollout.py",
        "fdagym/train/curriculum.py",
        "fdagym/train/hard_mix.py",
        "modal_train.py",
    ):
        text = (ROOT / relative).read_text()
        assert "SwappedFactsEnv" not in text
        assert "fdagym.swapped" not in text
    rollout = (ROOT / "fdagym" / "train" / "rollout.py").read_text()
    assert "with training_rollout()" in rollout
    assert "FDAEnv(" in rollout
    tree = ast.parse((ROOT / "fdagym" / "train" / "grpo_tools.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "fdagym.swapped" not in imported
    checker = (ROOT / "scripts" / "eval_tools.py").read_text()
    assert "--memorization" in checker
    assert 'summary["memorization"] = report' in checker


@pytest.fixture(scope="module")
def disk():
    from fdagym.openfda import DiskSnapshot

    return DiskSnapshot("2026-10-15")


def test_scripted_oracle_on_a_slice_follows_tools(disk):
    from fdagym.swapped import scripted_oracle_report

    holdout = load_split("hard_train_holdout")
    known = [
        next(task for task in holdout if parse_question(task["question"])["variant"] == "sponsor_count"),
        next(task for task in holdout if parse_question(task["question"])["variant"] == "year_reaction"),
    ]
    if not (Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl").exists():
        pytest.skip("data/tasks/hard.jsonl stays in the private repo")
    unseen = load_split("hard")[:2]
    report = scripted_oracle_report(disk, known, unseen, tool_budget=8)
    assert report["known"]["n"] == 2
    assert report["unseen"]["n"] == 2
    assert report["known"]["strict"] == 1.0
    assert report["unseen"]["strict"] == 1.0
    assert report["gap"]["strict"] == 0.0
    assert report["answered_without_looking"]["rate"] == 0.0
    assert report["swapped_facts"]["n"] == 2
    assert report["swapped_facts"]["follow_rate"] == 1.0
    assert report["swapped_facts"]["original"] == 0
    assert report["flags"]["level"] == "ok"
    assert report["auto_stop"] is False
    assert report["mean_tool_calls"]["known"] >= 3
    assert report["mean_tool_calls"]["unseen"] >= 3
