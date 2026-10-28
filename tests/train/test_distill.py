"""Teacher-trajectory distill SFT: filter, nested subsets, assistant-only loss."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from fdagym.train.distill import (
    DistillConfig,
    cap_attempts,
    data_hash,
    dedupe_trajectories,
    filter_trajectories,
    fixture_path,
    load_teacher_jsonl,
    nested_subsets,
    parse_sizes,
    prepare_distill,
    proportional_counts,
    size_run_id,
)
from fdagym.train.distill_curve import collect_curve, write_curve
from fdagym.train.distill_sft import (
    assert_assistant_only_labels,
    build_distill_rows,
    encode_messages,
    messages_from_rollout,
)
from fdagym.train.jobs import run_distill_sft
from fdagym.train.rollout import QwenToolCodec, RolloutLimits
from fdagym.train.rollout_dry import CharTokenizer
from fdagym.train.sft_config import SFTConfig
from fdagym.train.sft_tools import IGNORE_INDEX, build_sft_rows, sft_example
from tests.env.support import ALL_TASKS, JsonSnapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = fixture_path()
CLI_PATH = ROOT / "scripts" / "run_gpu.py"
SONNET = "openrouter/anthropic/claude-sonnet-4.5"
HAIKU = "openrouter/anthropic/claude-haiku-4.5"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_gpu_distill_under_test", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic(
    *,
    question_id: str,
    template: str,
    teacher: str = "t",
    attempt: int = 0,
    passed: bool = True,
    variant: str = "",
    traj_id: str | None = None,
) -> dict:
    from fdagym.train.distill import parse_teacher_trajectory

    reward = {
        "total": 1.0 if passed else 0.1,
        "correctness": 1.0 if passed else 0.0,
        "citation": 1.0 if passed else 0.0,
        "format": 1.0,
        "efficiency": 1.0 if passed else 0.0,
        "citations_valid": 1 if passed else 0,
        "citations_total": 1,
        "notes": [],
    }
    messages = [
        {"role": "system", "content": "stub"},
        {"role": "user", "content": f"question {question_id}"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "x", "citations": []}}\n</tool_call>',
        },
    ]
    return parse_teacher_trajectory(
        {
            "id": traj_id or f"{teacher}/{question_id}/a{attempt}",
            "question_id": question_id,
            "template": template,
            "variant": variant,
            "teacher_model": teacher,
            "attempt_index": attempt,
            "prompt_version": "v4",
            "tool_budget": 8,
            "messages": messages,
            "reward": reward,
            "passed": passed,
        }
    )


def test_distill_modules_do_not_import_rewards_or_disjoint():
    import ast

    for rel in (
        "fdagym/train/distill.py",
        "fdagym/train/distill_sft.py",
        "fdagym/train/distill_launch.py",
        "fdagym/train/distill_curve.py",
        "scripts/sft_distill.py",
        "scripts/distill_curve.py",
    ):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        assert all(not name.startswith("fdagym.rewards") for name in names)
        assert all("disjoint" not in name for name in names)


def test_fixture_validates_and_has_oracle_conversations():
    rows = load_teacher_jsonl(FIXTURE)
    assert len(rows) >= 12
    templates = {row.template for row in rows}
    assert {"boxed_warning", "faers_top_reaction", "faers_count", "approval"} <= templates
    assert any(not row.passed for row in rows)
    assert any(row.attempt_index > 0 for row in rows)
    teachers = {row.teacher_model for row in rows}
    assert SONNET in teachers and HAIKU in teachers
    first = next(row for row in rows if row.question_id == "FG-0001" and row.attempt_index == 0 and row.teacher_model == SONNET)
    roles = [message["role"] for message in first.messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "assistant" in roles and "tool" in roles
    assert "<tool_call>" in first.messages[2]["content"]
    assert "<|im_start|>" not in first.messages[2]["content"]


def test_filter_keeps_passing_rows_and_dedupes():
    rows = load_teacher_jsonl(FIXTURE)
    kept, stats = filter_trajectories(rows)
    assert stats["n_skipped"]["failed"] >= 1
    assert stats["n_skipped"]["duplicate"] >= 1
    assert all(row.passed for row in kept)
    keys = [row.dedupe_key for row in kept]
    assert len(keys) == len(set(keys))
    sonnet_only, sonnet_stats = filter_trajectories(rows, teachers=[SONNET])
    assert all(row.teacher_model == SONNET for row in sonnet_only)
    assert sonnet_stats["n_kept"] < stats["n_kept"]
    boxed, _ = filter_trajectories(rows, templates=["boxed_warning"])
    assert boxed and all(row.template == "boxed_warning" for row in boxed)
    capped, cap_stats = filter_trajectories(rows, max_attempts_per_question=1)
    assert cap_stats["n_skipped"]["max_attempts"] >= 1
    counts = {}
    for row in capped:
        counts[row.question_id] = counts.get(row.question_id, 0) + 1
    assert max(counts.values()) == 1


def test_dedupe_and_cap_are_order_stable():
    a = _synthetic(question_id="Q", template="t", attempt=0, traj_id="first")
    b = _synthetic(question_id="Q", template="t", attempt=0, traj_id="second")
    unique, n = dedupe_trajectories([a, b, a])
    assert n == 2
    assert unique == [a]
    extra = _synthetic(question_id="Q", template="t", attempt=1, teacher="other")
    kept, dropped = cap_attempts([a, extra], 1)
    assert dropped == 1
    assert kept == [a]


def test_nested_subsets_are_contained_and_stratified():
    rows = []
    for template, n in (("A", 10), ("B", 6), ("C", 4)):
        for i in range(n):
            rows.append(_synthetic(question_id=f"{template}-{i}", template=template, attempt=0))
            if i < 2:
                rows.append(
                    _synthetic(question_id=f"{template}-{i}", template=template, attempt=1, teacher="other")
                )
    subsets = nested_subsets(rows, "5,10,all", seed=0)
    small = subsets["5"]
    mid = subsets["10"]
    large = subsets["all"]
    def ids(group):
        return {row.id for row in group}

    assert ids(small) < ids(mid) <= ids(large)
    assert len(small) == 5
    assert len(mid) == 10
    assert len(large) == len(rows)
    # every template appears in the smallest set that can hold one of each
    assert {row.template for row in small} == {"A", "B", "C"}
    for name, chosen in subsets.items():
        counts = {template: sum(1 for row in chosen if row.template == template) for template in ("A", "B", "C")}
        total = sum(counts.values())
        # Largest remainder: shares stay within 1 of the rounded proportion of the parent.
        parent = large if name == "all" else (mid if name == "10" else None)
        if parent is None:
            parent = mid
        parent_counts = {template: sum(1 for row in parent if row.template == template) for template in ("A", "B", "C")}
        expected = proportional_counts(parent_counts, total)
        assert counts == expected


def test_unique_questions_are_preferred_over_repeats():
    rows = []
    for i in range(6):
        rows.append(_synthetic(question_id=f"Q{i}", template="A", attempt=0))
        rows.append(_synthetic(question_id=f"Q{i}", template="A", attempt=1, teacher="other"))
    subsets = nested_subsets(rows, "6,12", seed=0)
    six = subsets["6"]
    assert len({row.question_id for row in six}) == 6
    assert all(row.attempt_index == 0 for row in six)
    twelve = subsets["12"]
    assert len(twelve) == 12
    assert {row.id for row in six} <= {row.id for row in twelve}


def test_same_seed_is_deterministic():
    rows = load_teacher_jsonl(FIXTURE)
    kept, _ = filter_trajectories(rows)
    a = nested_subsets(kept, "3,5,all", seed=0)
    b = nested_subsets(kept, "3,5,all", seed=0)
    c = nested_subsets(kept, "3,5,all", seed=1)
    assert [row.id for row in a["5"]] == [row.id for row in b["5"]]
    assert [row.id for row in a["5"]] != [row.id for row in c["5"]] or len(kept) < 2
    assert data_hash(a["5"]) == data_hash(list(reversed(a["5"])))


def test_parse_sizes_and_run_id():
    assert parse_sizes("100,250,all,100") == ["100", "250", "all"]
    with pytest.raises(ValueError, match="positive"):
        parse_sizes("0")
    assert size_run_id("distill-sft-qwen3b", "100") == "distill-sft-qwen3b-n100"


def test_distill_config_matches_warmup_sft_defaults():
    distill = DistillConfig()
    warmup = SFTConfig()
    assert distill.learning_rate == warmup.learning_rate == 1e-4
    assert distill.num_train_epochs == warmup.num_train_epochs == 1.0
    assert distill.per_device_train_batch_size == warmup.per_device_train_batch_size == 4
    assert distill.gradient_accumulation_steps == warmup.gradient_accumulation_steps == 2
    assert distill.max_length == warmup.max_length == 8192
    assert distill.lora_kwargs() if False else distill.sft_config().lora_kwargs() == warmup.lora_kwargs()
    lora = distill.sft_config().lora_kwargs()
    assert lora["r"] == 16 and lora["lora_alpha"] == 32
    assert distill.prompt_version == "v4"
    assert distill.tool_budget == 8
    assert distill.sft_config().val_drug_fraction == 0.0
    from fdagym.train.distill_launch import trainer_kwargs

    args = trainer_kwargs(DistillConfig(max_steps=2), "/tmp/out")
    assert args["max_steps"] == 2
    assert "eval_strategy" not in args


def test_assistant_only_loss_matches_oracle_sft_rows():
    snapshot = JsonSnapshot()
    tokenizer = CharTokenizer()
    codec = QwenToolCodec()
    limits = RolloutLimits(temperature=0.0)
    oracle = build_sft_rows(snapshot, ALL_TASKS, ["FG-0001", "FG-0002"], tokenizer, codec, limits, max_length=8192)
    assert oracle["n_kept"] == 2
    for rollout in oracle["rollouts"]:
        messages = messages_from_rollout(rollout, tokenizer)
        encoded = encode_messages(messages, tokenizer, codec, limits)
        row = sft_example(encoded)
        assert encoded.prompt_ids == rollout.prompt_ids
        assert encoded.completion_ids == rollout.completion_ids
        assert encoded.loss_mask == rollout.loss_mask
        assert row["labels"] == sft_example(rollout)["labels"]
        assert_assistant_only_labels(row, encoded)
        assert row["labels"][: len(rollout.prompt_ids)] == [IGNORE_INDEX] * len(rollout.prompt_ids)
        prompt_text = tokenizer.decode(rollout.prompt_ids)
        assert prompt_text.endswith("<|im_start|>assistant\n")
        supervised = tokenizer.decode(
            [token for token, bit in zip(encoded.completion_ids, encoded.loss_mask) if bit]
        )
        assert "<tool_call>" in supervised
        assert "<tool_response>" not in supervised
        masked = tokenizer.decode(
            [token for token, bit in zip(encoded.completion_ids, encoded.loss_mask) if not bit]
        )
        assert "<tool_response>" in masked


def test_fixture_rows_mask_tool_tokens():
    rows = load_teacher_jsonl(FIXTURE)
    passing = [row for row in rows if row.passed and row.question_id.startswith("FG-")]
    tokenizer = CharTokenizer()
    built = build_distill_rows(passing, tokenizer, max_length=8192)
    assert built["n_kept"] == len(passing)
    assert built["n_skipped"] == 0
    for row, encoded in zip(built["rows"], built["encoded"], strict=True):
        assert_assistant_only_labels(row, encoded)
        assert sum(1 for label in row["labels"] if label != IGNORE_INDEX) == sum(encoded.loss_mask)


def test_json_assistant_content_renders_through_the_codec():
    tokenizer = CharTokenizer()
    codec = QwenToolCodec()
    messages = [
        {"role": "user", "content": "Does warfarin have a boxed warning?"},
        {
            "role": "assistant",
            "content": json.dumps({"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}),
        },
    ]
    encoded = encode_messages(messages, tokenizer, codec)
    text = tokenizer.decode(encoded.completion_ids)
    assert "<tool_call>" in text
    assert '"name": "search_drug_label"' in text or '"name":"search_drug_label"' in text
    assert all(encoded.loss_mask)


def test_prepare_and_jobs_dry_run(tmp_path):
    config = DistillConfig(sizes=("3", "5", "all"), seed=0, run_id="distill-dry")
    prepared = prepare_distill(FIXTURE, config)
    assert prepared["filter"]["n_kept"] >= 8
    assert {row.id for row in prepared["subsets"]["3"]} <= {row.id for row in prepared["subsets"]["5"]}
    result = run_distill_sft(
        runs_dir=tmp_path / "runs",
        trajectories=FIXTURE,
        run_id="distill-dry",
        sizes="3,5,all",
        seed=0,
        dry_run=True,
    )
    assert result["dry_run"] is True
    parent = tmp_path / "runs" / "distill-dry"
    assert (parent / "DONE").is_file()
    assert (parent / "dry-run.json").is_file()
    payload = json.loads((parent / "dry-run.json").read_text())
    assert payload["sizes"]["3"]["n"] == 3
    assert payload["sizes"]["5"]["n"] == 5
    assert payload["sizes"]["all"]["n"] == result["filter"]["n_kept"]
    assert payload["sizes"]["3"]["n_tokenized"] == 3
    assert all(token_count > 0 for token_count in payload["sizes"]["all"]["supervised_tokens"])


def test_cli_distill_sft_forwards_dry_run(monkeypatch, tmp_path):
    cli = _load_cli()
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"run_id": kwargs["run_id"], "dry_run": kwargs["dry_run"], "filter": {"n_kept": 0}}

    monkeypatch.setattr("fdagym.train.jobs.run_distill_sft", fake)
    assert (
        cli.main(
            [
                "distill-sft",
                "--dry-run",
                "--runs-dir",
                str(tmp_path),
                "--trajectories",
                str(FIXTURE),
                "--sizes",
                "3,5,all",
                "--seed",
                "0",
                "--teacher",
                SONNET,
                "--init-adapter",
                "/workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter",
                "--eval-split",
                "hard",
            ]
        )
        == 0
    )
    assert seen["dry_run"] is True
    assert seen["eval_after"] is False
    assert seen["sizes"] == "3,5,all"
    assert seen["teachers"] == [SONNET]
    assert seen["eval_split"] == "hard"
    assert seen["init_adapter"].endswith("warmup-mh1/adapter")
    assert seen["on_save"] is None
    parsed = cli._parser().parse_args(["distill-sft", "--eval", "--eval-split", "hard"])
    assert parsed.eval is True
    assert parsed.eval_split == "hard"
    assert parsed.dry_run is False
    text = CLI_PATH.read_text(encoding="utf-8")
    assert "WANDB_API_KEY=" not in text
    assert "HF_TOKEN=" not in text


def test_curve_csv_and_svg(tmp_path):
    runs = tmp_path / "runs"
    prefix = "distill-sft-qwen3b"
    for size, reward, extra in (("3", 0.2, True), ("5", 0.4, True), ("all", 0.5, False)):
        run_id = f"{prefix}-n{size}-hard"
        directory = runs / run_id
        directory.mkdir(parents=True)
        model = "Qwen/Qwen2.5-3B-Instruct+tools"
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "split": "hard",
                    "models": {
                        model: {
                            "n": 4,
                            "mean_reward": reward,
                            "pass_rate": reward,
                            "mean_citation": 0.5,
                            "mean_tools": 3.0,
                            "cost_usd": 0,
                        }
                    },
                    "leaderboard": [{"name": model, "mean_reward": reward, "pass": reward}],
                }
            )
        )
        train_dir = runs / f"{prefix}-n{size}"
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "manifest.json").write_text(json.dumps({"n": int(size) if size != "all" else 14}))
        if extra:
            tasks = [
                {"id": "FG-8001", "type": "compare", "question": "Of a, b, and c, which drug has a boxed warning on its label? Answer no if none of them do."},
                {"id": "FG-8050", "type": "sponsor_hop", "question": "Who is the sponsor of the original application for acoramidis?"},
            ]
            (directory / "tasks.json").write_text(json.dumps(tasks))
            lines = [
                json.dumps(
                    {
                        "task_id": "FG-8001",
                        "reward": {"total": reward, "correctness": 1.0 if reward > 0.3 else 0.0},
                    }
                ),
                json.dumps(
                    {
                        "task_id": "FG-8050",
                        "reward": {"total": 0.1, "correctness": 0.0},
                    }
                ),
            ]
            (runs / f"{run_id}.jsonl").write_text("\n".join(lines) + "\n")
    records = collect_curve(runs, prefix, "3,5,all", split="hard")
    overall = [row for row in records if row.get("variant") == ""]
    assert [row["mean_reward"] for row in overall] == [0.2, 0.4, 0.5]
    variants = {row["variant"] for row in records if row.get("variant")}
    assert "boxed_none" in variants
    assert "sponsor_hop" in variants
    written = write_curve(records, tmp_path / "curve")
    csv_text = Path(written["csv"]).read_text()
    assert "mean_reward" in csv_text
    assert "0.2" in csv_text
    svg = Path(written["svg"]).read_text()
    assert "<svg" in svg
    assert "overall" in svg
