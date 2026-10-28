"""CLI and Modal wrappers call the same job functions. CPU only. No Modal and no GPU."""

import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest

from fdagym.train.dispersion import ids_sha256
from fdagym.train.jobs import prepare_runtime, run_eval_tools, run_probe_dispersion, run_train_tools
from tests.train.test_dispersion import PROBE_IDS, PROBE_SHA

ROOT = Path(__file__).resolve().parents[2]
MODAL = ROOT / "modal_train.py"
CLI_PATH = ROOT / "scripts" / "run_gpu.py"


@pytest.fixture(autouse=True)
def _restore_runtime_env():
    keys = ("HF_HOME", "HF_HUB_CACHE", "FDAGYM_GIT_SHA", "FDAGYM_EXECUTOR")
    saved = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_gpu_under_test", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _resolved_defaults(path: Path, name: str) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assigned[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    function = _function(path, name)
    args = function.args.args
    defaults = function.args.defaults
    offset = len(args) - len(defaults)
    resolved = {}
    for arg, default in zip(args[offset:], defaults, strict=True):
        if isinstance(default, ast.Constant):
            resolved[arg.arg] = default.value
        elif isinstance(default, ast.Name) and default.id in assigned:
            resolved[arg.arg] = assigned[default.id]
        else:
            resolved[arg.arg] = ast.literal_eval(default)
    return resolved


def _call_keywords(path: Path, function: str, callee: str) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    node = _function(path, function)
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    found = None
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == callee:
            found = child
            break
    assert found is not None, callee
    keywords = {}
    for keyword in found.keywords:
        assert keyword.arg is not None
        keywords[keyword.arg] = ast.get_source_segment(source, keyword.value)
    return keywords


def test_published_probe_file_is_the_pr27_draw():
    ids = (ROOT / "data" / "probe" / "dispersion-warmup-mh1.txt").read_text(encoding="utf-8").split()
    assert ids == PROBE_IDS
    assert ids_sha256(ids) == PROBE_SHA
    assert len(ids) == 40


def test_job_defaults_match_the_modal_signatures():
    train = _resolved_defaults(MODAL, "train_tools")
    signature = inspect.signature(run_train_tools)
    for name, modal_default in train.items():
        assert signature.parameters[name].default == modal_default
    assert signature.parameters["runs_dir"].default is inspect.Parameter.empty
    assert signature.parameters["on_save"].default is None
    assert "hf_token" not in signature.parameters

    evaluate = _resolved_defaults(MODAL, "eval_tools")
    eval_signature = inspect.signature(run_eval_tools)
    for name, modal_default in evaluate.items():
        if name == "hf_token":
            continue
        assert eval_signature.parameters[name].default == modal_default
    assert "hf_token" not in eval_signature.parameters

    probe = _resolved_defaults(MODAL, "probe_dispersion")
    probe_signature = inspect.signature(run_probe_dispersion)
    for name, modal_default in probe.items():
        assert probe_signature.parameters[name].default == modal_default
    assert probe["k"] == 8
    assert probe["hard_train_mix"] == 0.6
    assert probe["tool_budget"] == 8
    assert probe["prompt_version"] == "v4"


def test_modal_wrappers_forward_the_same_arguments_the_cli_does():
    train_modal = _call_keywords(MODAL, "train_tools", "run_train_tools")
    train_cli = _call_keywords(CLI_PATH, "_train", "run_train_tools")
    assert set(train_modal) == set(train_cli)
    modal_params = [arg.arg for arg in _function(MODAL, "train_tools").args.args]
    for name in modal_params:
        assert train_modal[name] == name
        assert train_cli[name] == f"args.{name}"
    assert train_modal["on_save"] == "_commit"
    assert train_cli["on_save"] == "None"
    assert "VOL_MOUNT" in train_modal["runs_dir"]
    assert train_cli["runs_dir"] == "args.runs_dir"

    eval_modal = _call_keywords(MODAL, "eval_tools", "run_eval_tools")
    eval_cli = _call_keywords(CLI_PATH, "_eval", "run_eval_tools")
    assert set(eval_modal) == set(eval_cli)
    assert "hf_token" not in eval_modal
    eval_params = [arg.arg for arg in _function(MODAL, "eval_tools").args.args]
    for name in eval_params:
        if name == "hf_token":
            continue
        assert eval_modal[name] == name
        assert eval_cli[name] == f"args.{name}"
    wrapper = ast.get_source_segment(MODAL.read_text(encoding="utf-8"), _function(MODAL, "eval_tools"))
    assert wrapper is not None
    assert 'os.environ["HF_TOKEN"] = hf_token' in wrapper

    probe_modal = _call_keywords(MODAL, "probe_dispersion", "run_probe_dispersion")
    probe_cli = _call_keywords(CLI_PATH, "_probe", "run_probe_dispersion")
    assert set(probe_modal) == set(probe_cli)
    probe_params = [arg.arg for arg in _function(MODAL, "probe_dispersion").args.args]
    for name in probe_params:
        assert probe_modal[name] == name
        if name == "task_ids":
            assert probe_cli[name] == "_task_ids(args)"
        else:
            assert probe_cli[name] == f"args.{name}"


def test_cli_defaults_match_and_secrets_are_not_arguments():
    cli = _load_cli()
    train = cli._parser().parse_args(["train-tools"])
    assert train.runs_dir == Path("runs")
    assert train.resume is True
    assert train.run_id == "grpo-qwen3b-tools"
    assert train.sft_run_id == "sft-qwen3b-tools"
    assert train.init_adapter == ""
    assert train.tool_budget == 8
    assert train.prompt_version == "v4"
    assert train.hard_train_mix == 0.0
    assert train.max_steps == 0
    assert train.memorization_every == 10
    assert train.save_steps == 50
    assert train.prompts_per_step is None
    assert train.dynamic_sampling is None
    assert train.difficulty_filter is None
    assert train.max_resample_attempts is None
    assert train.temperature == 0.65
    assert train.pool_types == "boxed_warning,faers_top_reaction,approval,faers_count"
    off = cli._parser().parse_args(["train-tools", "--no-dynamic-sampling", "--no-difficulty-filter", "--no-resume"])
    assert off.dynamic_sampling is False
    assert off.difficulty_filter is False
    assert off.resume is False
    on = cli._parser().parse_args(["train-tools", "--dynamic-sampling", "--difficulty-filter"])
    assert on.dynamic_sampling is True
    assert on.difficulty_filter is True
    probe = cli._parser().parse_args(
        ["probe", "--adapter", "a", "--task-ids-sha256", "b", "--run-id", "c", "--task-id", "FG-1"]
    )
    assert probe.k == 8
    assert probe.seed == 0
    assert probe.executor == "runpod"
    assert probe.hard_train_mix == 0.6
    assert probe.tool_budget == 8
    assert probe.prompt_version == "v4"
    assert probe.temperature == 0.65
    assert probe.pool_types == "boxed_warning,faers_top_reaction,approval,faers_count"
    text = CLI_PATH.read_text(encoding="utf-8")
    assert "WANDB_API_KEY=" not in text
    assert "HF_TOKEN=" not in text
    with pytest.raises(SystemExit):
        cli.main(["train-tools", "--wandb-api-key", "nope"])
    with pytest.raises(SystemExit):
        cli.main(["eval-tools", "--hf-token", "nope"])


def test_cli_forwards_the_hard_run_and_the_probe_file(monkeypatch, tmp_path, capsys):
    cli = _load_cli()
    seen = {}

    def fake_train(**kwargs):
        seen["train"] = kwargs
        return {"run_id": kwargs["run_id"]}

    def fake_probe(**kwargs):
        seen["probe"] = kwargs
        return {"n_groups": 0}

    monkeypatch.setattr("fdagym.train.jobs.run_train_tools", fake_train)
    monkeypatch.setattr("fdagym.train.jobs.run_probe_dispersion", fake_probe)
    runs = tmp_path / "runs"
    assert (
        cli.main(
            [
                "train-tools",
                "--runs-dir",
                str(runs),
                "--run-id",
                "grpo-qwen3b-tools-hard-mh1",
                "--init-adapter",
                "/workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter",
                "--tool-budget",
                "8",
                "--prompt-version",
                "v4",
                "--hard-train-mix",
                "0.6",
                "--max-steps",
                "60",
                "--memorization-every",
                "10",
                "--save-steps",
                "10",
            ]
        )
        == 0
    )
    forwarded = seen["train"]
    assert forwarded["runs_dir"] == runs
    assert forwarded["on_save"] is None
    assert forwarded["run_id"] == "grpo-qwen3b-tools-hard-mh1"
    assert forwarded["init_adapter"].endswith("sft-qwen3b-tools-v4-warmup-mh1/adapter")
    assert forwarded["tool_budget"] == 8
    assert forwarded["prompt_version"] == "v4"
    assert forwarded["hard_train_mix"] == 0.6
    assert forwarded["max_steps"] == 60
    assert forwarded["memorization_every"] == 10
    assert forwarded["save_steps"] == 10
    assert forwarded["resume"] is True
    assert forwarded["temperature"] == 0.65
    assert forwarded["pool_types"] == "boxed_warning,faers_top_reaction,approval,faers_count"
    assert forwarded["prompts_per_step"] is None
    assert forwarded["dynamic_sampling"] is None
    dry = cli._parser().parse_args(
        [
            "train-tools",
            "--no-resume",
            "--run-id",
            "grpo-qwen3b-tools-hard-mh1-dry",
            "--max-steps",
            "2",
            "--save-steps",
            "10",
            "--hard-train-mix",
            "0.6",
        ]
    )
    assert dry.resume is False
    assert dry.max_steps == 2
    assert dry.save_steps == 10
    assert (
        cli.main(
            [
                "probe",
                "--runs-dir",
                str(runs),
                "--adapter",
                "/workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter",
                "--task-ids-file",
                str(ROOT / "data" / "probe" / "dispersion-warmup-mh1.txt"),
                "--task-ids-sha256",
                PROBE_SHA,
                "--run-id",
                "probe-dispersion-warmup-mh1",
            ]
        )
        == 0
    )
    probed = seen["probe"]
    assert probed["task_ids"] == PROBE_IDS
    assert probed["task_ids_sha256"] == PROBE_SHA
    assert probed["k"] == 8
    assert probed["seed"] == 0
    assert probed["hard_train_mix"] == 0.6
    assert probed["temperature"] == 0.65
    assert probed["pool_types"] == "boxed_warning,faers_top_reaction,approval,faers_count"
    assert probed["on_save"] is None
    assert "secret" not in capsys.readouterr().out
    assert (
        cli.main(
            [
                "train-tools",
                "--runs-dir",
                str(runs),
                "--run-id",
                "grpo-qwen3b-distill-sft500",
                "--init-adapter",
                "example/fdagym-adapter",
                "--temperature",
                "0.8",
                "--pool-types",
                "boxed_warning,faers_top_reaction,approval,faers_count,composite",
            ]
        )
        == 0
    )
    override = seen["train"]
    assert override["init_adapter"] == "example/fdagym-adapter"
    assert override["temperature"] == 0.8
    assert override["pool_types"] == "boxed_warning,faers_top_reaction,approval,faers_count,composite"


def test_train_job_builds_the_same_hard_mix_config(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        (Path(kwargs["results_dir"]) / kwargs["run_id"]).mkdir(parents=True, exist_ok=True)
        return {"run_id": kwargs["run_id"], "adapter": "adapter", "n_prompts": 4, "resumed": False}

    monkeypatch.setattr("fdagym.train.grpo_tools.run_tool_grpo", fake_run)
    commits = []
    runs = tmp_path / "runs"
    result = run_train_tools(
        runs_dir=runs,
        on_save=lambda state=None: commits.append(state),
        run_id="grpo-qwen3b-tools-hard-mh1",
        init_adapter=str(tmp_path / "adapter"),
        tool_budget=8,
        prompt_version="v4",
        hard_train_mix=0.6,
        max_steps=60,
        memorization_every=10,
        save_steps=10,
    )
    config = captured["args"][2]
    limits = captured["kwargs"]["limits"]
    assert config.max_steps == 60
    assert config.save_steps == 10
    assert config.num_generations == 8
    assert config.dense_reward is True
    assert config.temperature == 0.65
    assert config.gradient_accumulation_steps == 4
    assert limits.temperature == 0.65
    assert limits.tool_budget == 8
    assert limits.prompt_version == "v4"
    assert limits.max_new_tokens == 256
    assert captured["kwargs"]["tool_parse_bonus"] == 0.05
    assert captured["kwargs"]["prompts_per_step"] == 4
    assert captured["kwargs"]["dynamic_sampling"] is True
    assert captured["kwargs"]["memorization"] is True
    assert captured["kwargs"]["memorization_every"] == 10
    weights = captured["kwargs"]["hard_train_mix"]
    assert weights["hard"] == pytest.approx(0.6)
    assert weights["current"] == pytest.approx(0.4)
    assert captured["kwargs"]["prompt_filter"].enabled is True
    assert captured["kwargs"]["pool_types"] == ("boxed_warning", "faers_top_reaction", "approval", "faers_count")
    assert captured["kwargs"]["sft_adapter"] == str(tmp_path / "adapter")
    assert captured["kwargs"]["results_dir"] == runs
    assert captured["args"][0].version == "2026-10-15"
    assert result["pool_exclusion"]["n_holdout"] == 34
    hard_n = 100 if (Path(__file__).resolve().parents[2] / "data" / "tasks" / "hard.jsonl").exists() else 93
    assert result["pool_exclusion"]["n_hard_eval"] == hard_n
    assert commits == [None, None]
    assert (runs / "grpo-qwen3b-tools-hard-mh1" / "DONE").is_file()
    manifest = json.loads((runs / "grpo-qwen3b-tools-hard-mh1" / "run.json").read_text())
    assert manifest["job"] == "tool-train"
    assert manifest["config"]["rollout"]["tool_budget"] == 8
    plain_commits = []

    def fake_plain(*args, **kwargs):
        captured["plain"] = kwargs
        (Path(kwargs["results_dir"]) / kwargs["run_id"]).mkdir(parents=True, exist_ok=True)
        return {"run_id": kwargs["run_id"], "adapter": "adapter", "n_prompts": 1, "resumed": False}

    monkeypatch.setattr("fdagym.train.grpo_tools.run_tool_grpo", fake_plain)
    plain = run_train_tools(runs_dir=runs, on_save=lambda state=None: plain_commits.append(state))
    assert plain["run_id"] == "grpo-qwen3b-tools"
    assert "pool_exclusion" not in plain
    assert captured["plain"]["memorization"] is False
    assert captured["plain"]["hard_train_mix"] is None
    assert captured["plain"]["prompts_per_step"] == 1
    assert captured["plain"]["dynamic_sampling"] is False
    assert captured["plain"]["max_resample_attempts"] == 3
    assert captured["plain"]["prompt_filter"].enabled is False


def test_unflagged_train_keeps_one_group_and_a_two_step_dry_run_keeps_save_steps(monkeypatch, tmp_path):
    seen = {}

    def fake_run(*args, **kwargs):
        seen["config"] = args[2]
        seen["kwargs"] = kwargs
        (Path(kwargs["results_dir"]) / kwargs["run_id"]).mkdir(parents=True, exist_ok=True)
        return {"run_id": kwargs["run_id"], "adapter": "a", "n_prompts": 1, "resumed": None}

    monkeypatch.setattr("fdagym.train.grpo_tools.run_tool_grpo", fake_run)
    run_train_tools(runs_dir=tmp_path / "runs", on_save=None)
    assert seen["config"].max_steps == 150
    assert seen["config"].save_steps == 50
    assert seen["config"].num_generations == 8
    assert seen["kwargs"]["prompts_per_step"] == 1
    assert seen["kwargs"]["dynamic_sampling"] is False
    assert seen["kwargs"]["prompt_filter"].enabled is False
    run_train_tools(
        runs_dir=tmp_path / "runs2",
        on_save=None,
        run_id="grpo-qwen3b-tools-hard-mh1-dry",
        max_steps=2,
        save_steps=10,
        hard_train_mix=0.6,
        resume=False,
    )
    assert seen["config"].max_steps == 2
    assert seen["config"].save_steps == 10
    assert seen["config"].num_generations == 8
    assert seen["kwargs"]["prompts_per_step"] == 4
    assert seen["kwargs"]["resume"] is False


def test_eval_job_is_greedy_and_keeps_the_split(monkeypatch, tmp_path):
    captured = {}

    def fake_load(model, adapter=None):
        captured["model"] = model
        captured["adapter"] = adapter
        return "generator", "tokenizer"

    def fake_eval(**kwargs):
        captured["eval"] = kwargs
        (Path(kwargs["results_dir"]) / kwargs["run_id"]).mkdir(parents=True, exist_ok=True)
        return {
            "run_id": kwargs["run_id"],
            "models": {"m": {}},
            "protocol": "tool_rollout",
            "dense_reward": {"mean": 0.1},
            "prompt_version": kwargs["limits"].prompt_version,
            "tool_budget": kwargs["limits"].tool_budget,
        }

    monkeypatch.setattr("fdagym.train.hf_rollout.load_hf_rollout", fake_load)
    monkeypatch.setattr("fdagym.train.eval_tools.eval_tools_from_disk", fake_eval)
    commits = []
    runs = tmp_path / "runs"
    result = run_eval_tools(
        runs_dir=runs,
        on_save=lambda: commits.append("commit"),
        adapter="/workspace/runs/grpo-qwen3b-tools-hard-mh1/adapter",
        run_id="grpo-qwen3b-tools-hard-mh1-hard",
        split="hard",
        tool_budget=8,
        prompt_version="v4",
    )
    assert result["run_id"] == "grpo-qwen3b-tools-hard-mh1-hard"
    assert result["protocol"] == "tool_rollout"
    limits = captured["eval"]["limits"]
    assert limits.temperature == 0.0
    assert limits.tool_budget == 8
    assert limits.prompt_version == "v4"
    assert captured["eval"]["split"] == "hard"
    assert captured["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert commits == ["commit", "commit"]
    assert (runs / "grpo-qwen3b-tools-hard-mh1-hard" / "DONE").is_file()
    manifest = json.loads((runs / "grpo-qwen3b-tools-hard-mh1-hard" / "run.json").read_text())
    assert manifest["job"] == "tool-eval"
    assert manifest["config"]["split"] == "hard"


def test_probe_refuses_a_bad_k_a_bad_run_id_and_an_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="k must be 8"):
        run_probe_dispersion(
            runs_dir=tmp_path,
            adapter="/nope",
            task_ids=["FG-1"],
            task_ids_sha256="0" * 64,
            run_id="probe",
            k=2,
        )
    with pytest.raises(ValueError, match="path segment"):
        run_probe_dispersion(
            runs_dir=tmp_path,
            adapter="/nope",
            task_ids=PROBE_IDS,
            task_ids_sha256=PROBE_SHA,
            run_id="a/b",
        )
    run_dir = tmp_path / "runs" / "probe-dispersion-warmup-mh1"
    run_dir.mkdir(parents=True)
    pool_id = "FG-8405"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_probe_dispersion(
            runs_dir=tmp_path / "runs",
            adapter="/no/such/adapter",
            task_ids=[pool_id],
            task_ids_sha256=ids_sha256([pool_id]),
            run_id="probe-dispersion-warmup-mh1",
        )
    assert not (run_dir / "DONE").exists()
    assert not (run_dir / "FAILED").exists()


def test_prepare_runtime_matches_modal_when_the_env_is_already_set(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("FDAGYM_GIT_SHA", raising=False)
    runs = tmp_path / "runs"
    prepare_runtime(runs)
    assert os.environ["HF_HOME"] == str(tmp_path / "hf-cache")
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "hf-cache")
    assert os.environ["FDAGYM_GIT_SHA"]
    monkeypatch.setenv("HF_HOME", "/vol/hf-cache")
    monkeypatch.setenv("HF_HUB_CACHE", "/vol/hf-cache")
    monkeypatch.setenv("FDAGYM_GIT_SHA", "baked")
    prepare_runtime(Path("/vol/runs"))
    assert os.environ["HF_HOME"] == "/vol/hf-cache"
    assert os.environ["FDAGYM_GIT_SHA"] == "baked"


def test_failed_marker_redacts_tokens_and_a_training_error_writes_it(monkeypatch, tmp_path):
    from fdagym.train.jobs import _mark

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    run = tmp_path / "run"
    run.mkdir()
    _mark(run, "failed", RuntimeError("boom secret-token wandb-secret"))
    text = (run / "FAILED").read_text(encoding="utf-8")
    assert "secret-token" not in text
    assert "wandb-secret" not in text
    assert "***" in text

    def explode(*_args, **_kwargs):
        raise RuntimeError("step failed secret-token")

    monkeypatch.setattr("fdagym.train.grpo_tools.run_tool_grpo", explode)
    runs = tmp_path / "runs"
    # The directory is created inside run_tool_grpo. The mock raises first, so create it
    # the way a failure after the first checkpoint would.
    (runs / "grpo-qwen3b-tools").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="step failed"):
        run_train_tools(runs_dir=runs, on_save=None, hard_train_mix=0.0)
    failed = (runs / "grpo-qwen3b-tools" / "FAILED").read_text(encoding="utf-8")
    assert "secret-token" not in failed


def test_jobs_module_does_not_import_torch_or_define_a_reward():
    source = (ROOT / "fdagym" / "train" / "jobs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"torch", "vllm", "modal", "wandb", "peft", "transformers"})
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert names.isdisjoint({"score", "episode_scores", "training_reward"})
    assert "fdagym.rewards" not in source
