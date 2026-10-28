import json

from fdagym.train.compare import compare_summaries, format_compare
from fdagym.train.eval_local import eval_split
from fdagym.train.gold import gold_submit
from fdagym.train.score import run_completion
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot
from tests.test_schemas import validator


def test_eval_split_gold_generate_scores_one(tmp_path):
    snapshot = JsonSnapshot()

    def generate(_messages):
        from fdagym.env import FDAEnv
        from fdagym.train.context import play_oracle

        env = FDAEnv(snapshot, ALL_TASKS, run_id="gold")
        steps = play_oracle(env, EXAMPLE_TASK)
        return json.dumps(gold_submit(EXAMPLE_TASK, steps))

    summary = eval_split(
        generate=generate,
        snapshot=snapshot,
        tasks=[EXAMPLE_TASK],
        model_name="Qwen/Qwen2.5-3B-Instruct",
        run_id="baseline-qwen3b",
        results_dir=tmp_path,
        split="test",
        kind="baseline",
        limit=1,
    )
    traj = json.loads((tmp_path / "baseline-qwen3b.jsonl").read_text().splitlines()[0])
    validator("trajectory.schema.json").validate(traj)
    assert traj["reward"]["total"] == 1.0
    assert traj["id"].startswith("baseline-qwen3b/")
    assert summary["models"]["Qwen/Qwen2.5-3B-Instruct"]["mean_reward"] == 1.0
    assert summary["dense_reward"]["mean_reward"] == 1.0
    assert summary["protocol"] == "plan_a_single_turn"
    board = json.loads((tmp_path / "baseline-qwen3b" / "summary.json").read_text())
    models = board["models"]
    if isinstance(models, dict):
        assert models["Qwen/Qwen2.5-3B-Instruct"]["mean_reward"] == 1.0
    else:
        assert models[0]["mean_reward"] == 1.0


def test_run_completion_junk_is_zero():
    snapshot = JsonSnapshot()
    traj = run_completion(snapshot, ALL_TASKS, EXAMPLE_TASK, "bleeding I guess", run_id="junk")
    assert traj["reward"]["total"] == 0.0
    assert traj["terminated_by"] in {"submit", "error"}


def test_compare_detects_a_win(tmp_path):
    baseline = {
        "run_id": "baseline-qwen3b",
        "models": [{"name": "3b", "mean_reward": 0.21, "pass_rate": 0.12}],
    }
    trained = {
        "run_id": "grpo-qwen3b-eval",
        "models": [{"name": "3b+lora", "mean_reward": 0.34, "pass_rate": 0.22}],
    }
    row = compare_summaries(baseline, trained)
    assert row["beats_baseline"] is True
    assert row["delta_mean_reward"] == 0.13
    assert "BEATS baseline" in format_compare(row)
    tied = compare_summaries(baseline, {"run_id": "flat", "models": [{"name": "x", "mean_reward": 0.21, "pass_rate": 0.12}]})
    assert tied["beats_baseline"] is False
    noise = compare_summaries(
        baseline,
        {"run_id": "tiny", "models": [{"name": "x", "mean_reward": 0.22, "pass_rate": 0.12}]},
    )
    assert noise["delta_mean_reward"] == 0.01
    assert noise["beats_baseline"] is False


def test_eval_overwrites_existing_jsonl(tmp_path):
    snapshot = JsonSnapshot()

    def generate(_messages, seed=0):
        return json.dumps({"answer": "NOT_FOUND", "citations": []})

    for _ in range(2):
        eval_split(
            generate=generate,
            snapshot=snapshot,
            tasks=[EXAMPLE_TASK],
            model_name="local",
            run_id="baseline-qwen3b",
            results_dir=tmp_path,
            split="test",
            kind="baseline",
            limit=1,
        )
    lines = [line for line in (tmp_path / "baseline-qwen3b.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == 1
