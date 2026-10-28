"""Warm-up SFT: continue an adapter, train an explicit id list, expose GRPO save_steps."""

import ast
import hashlib
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from fdagym.agents.baseline import load_split
from fdagym.tasks.hard_solver import parse_question
from fdagym.train.sft_config import SFTConfig
from fdagym.train.sft_launch import _blank, adapter_sha256, is_hub_adapter, load_sft_model
from fdagym.train.sft_select import sft_sources, task_ids_digest
from fdagym.train.sft_tools import VAL_DRUG_FRACTION, split_sft_by_drug
from fdagym.train.tool_config import configure_tool_run

ROOT = Path(__file__).resolve().parents[2]
WARMUP_SHA = "1686e70ac36e0ab67376bbe34eac457bb5a48d801ed633d31fa3375d3e212692"
MODAL = ROOT / "modal_train.py"


def test_task_ids_digest_is_sorted_ids_joined_by_newlines():
    assert task_ids_digest(["b", "a"]) == hashlib.sha256(b"a\nb").hexdigest()
    assert task_ids_digest(("b", "a")) == task_ids_digest(["a", "b"])


def test_task_ids_pool_is_exact_and_rejects_blocked_rows():
    train = [
        {"id": "T1", "split": "train", "type": "approval", "entity": {"generic_name": "a"}},
        {"id": "FG-8001", "split": "train", "type": "approval", "entity": {"generic_name": "b"}},
    ]
    hard = [
        {"id": "H1", "split": "hard_train", "type": "sponsor_hop", "entity": {"generic_name": "c"}},
    ]
    holdout_id = load_split("hard_train_holdout")[0]["id"]
    holdout = {
        "id": holdout_id,
        "split": "hard_train_holdout",
        "type": "compare",
        "entity": {"generic_name": "d"},
    }
    selected, mix = sft_sources(train, SFTConfig(task_ids=["H1", "T1"]), hard)
    assert [task["id"] for task in selected] == ["H1", "T1"]
    assert mix == {"task_ids": 2, "by_split": {"hard_train": 1, "train": 1}}
    digest = task_ids_digest(["T1"])
    again, _mix = sft_sources(train, SFTConfig(task_ids=("T1",), task_ids_sha256=digest.upper()), None)
    assert [task["id"] for task in again] == ["T1"]
    with pytest.raises(RuntimeError, match="unknown task ids"):
        sft_sources(train, SFTConfig(task_ids=["NOPE"]), hard)
    blocked_id = "FG-8001" if (ROOT / "data" / "tasks" / "hard.jsonl").exists() else load_split("public")[0]["id"]
    with pytest.raises(RuntimeError, match="hard eval|unknown task ids"):
        sft_sources(train, SFTConfig(task_ids=[blocked_id]), hard)
    with pytest.raises(RuntimeError, match="holdout"):
        sft_sources([*train, holdout], SFTConfig(task_ids=[holdout_id]), hard)
    with pytest.raises(RuntimeError, match="unknown task ids"):
        sft_sources(train, SFTConfig(task_ids=[holdout_id]), hard)
    with pytest.raises(RuntimeError, match="sha256"):
        sft_sources(train, SFTConfig(task_ids=["T1"], task_ids_sha256="0" * 64), None)
    with pytest.raises(RuntimeError, match="duplicate"):
        sft_sources(train, SFTConfig(task_ids=["T1", "T1"]), None)
    with pytest.raises(RuntimeError, match="empty"):
        sft_sources(train, SFTConfig(task_ids=[]), None)
    with pytest.raises(RuntimeError, match="both train and hard_train"):
        sft_sources(train, SFTConfig(task_ids=["T1"]), [{"id": "T1", "split": "hard_train"}])
    with pytest.raises(ValueError, match="include_hard_train"):
        SFTConfig(include_hard_train=True, task_ids=["H1"])
    with pytest.raises(ValueError, match="task_ids_sha256 requires"):
        SFTConfig(task_ids_sha256=digest)


def test_committed_warmup_ids_are_the_hard_train_subset():
    ids = (ROOT / "data" / "warmup" / "mh1_ids.txt").read_text().split()
    assert len(ids) == 71
    assert len(set(ids)) == 71
    assert task_ids_digest(ids) == WARMUP_SHA
    config = SFTConfig(
        task_ids=ids,
        task_ids_sha256=WARMUP_SHA,
        num_train_epochs=3,
        learning_rate=3e-5,
        save_steps=9,
        val_drug_fraction=0.0,
        warmup_ratio=0.1,
    )
    selected, mix = sft_sources(load_split("train"), config, load_split("hard_train"))
    assert [task["id"] for task in selected] == ids
    assert mix["by_split"] == {"hard_train": 71}
    variants = Counter((task["type"], parse_question(task["question"])["variant"]) for task in selected)
    assert variants[("sponsor_hop", "sponsor_drug")] == 16
    assert variants[("sponsor_hop", "sponsor_faers")] == 16
    assert variants[("sponsor_hop", "sponsor_count")] == 16
    assert variants[("compare", "earliest_boxed")] == 4
    assert variants[("compare", "earliest_reaction")] == 4
    assert variants[("compare", "faers_reaction")] == 4
    assert variants[("compare", "faers_year")] == 4
    assert variants[("compare", "year_reaction")] == 4
    assert variants[("compare", "boxed_none")] == 3
    for name in ("hard", "public", "hard_train_holdout", "test", "train"):
        path = ROOT / "data" / "tasks" / f"{name}.jsonl"
        if path.exists():
            assert set(ids).isdisjoint(task["id"] for task in load_split(name))
    train_rows, val_rows = split_sft_by_drug(selected, fraction=0.0, seed=0)
    assert val_rows == []
    assert [task["id"] for task in train_rows] == ids
    kwargs = config.trainer_kwargs("/tmp/out", eval_split=False)
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["learning_rate"] == 3e-5
    assert kwargs["save_steps"] == 9
    assert kwargs["warmup_ratio"] == 0.1
    assert "eval_strategy" not in kwargs


def test_sft_modal_defaults_match_sft_config_and_are_forwarded():
    defaults = _signature_defaults(MODAL, "sft")
    cfg = SFTConfig()
    assert defaults["num_train_epochs"] == cfg.num_train_epochs == 1.0
    assert defaults["learning_rate"] == cfg.learning_rate == 1e-4
    assert defaults["save_steps"] == cfg.save_steps == 50
    assert defaults["val_drug_fraction"] == cfg.val_drug_fraction == VAL_DRUG_FRACTION == 0.05
    assert defaults["warmup_ratio"] == cfg.warmup_ratio == 0.03
    assert defaults["init_adapter"] is None
    assert defaults["task_ids"] is None
    assert defaults["task_ids_sha256"] is None
    assert defaults["include_hard_train"] is False
    assert cfg.task_ids is None
    body = _function_text(MODAL, "sft", "train_tools")
    for name in (
        "init_adapter",
        "task_ids",
        "task_ids_sha256",
        "num_train_epochs",
        "learning_rate",
        "save_steps",
        "val_drug_fraction",
        "warmup_ratio",
        "include_hard_train",
        "tool_budget",
        "prompt_version",
    ):
        assert f"{name}={name}" in body
    assert 'load_split("hard_train") if config.include_hard_train or config.task_ids is not None' in body
    forwarded = SFTConfig(
        num_train_epochs=defaults["num_train_epochs"],
        learning_rate=defaults["learning_rate"],
        save_steps=defaults["save_steps"],
        val_drug_fraction=defaults["val_drug_fraction"],
        warmup_ratio=defaults["warmup_ratio"],
    )
    assert forwarded == SFTConfig()
    assert forwarded.trainer_kwargs("/tmp/out") == SFTConfig().trainer_kwargs("/tmp/out")


def test_zero_val_fraction_and_bad_sft_knobs():
    tasks = [
        {"id": "A", "entity": {"generic_name": "a"}},
        {"id": "B", "entity": {"generic_name": "a"}},
        {"id": "C", "entity": {"generic_name": "b"}},
    ]
    train_rows, val_rows = split_sft_by_drug(tasks, fraction=0.0, seed=0)
    assert val_rows == []
    assert train_rows == tasks
    assert SFTConfig(val_drug_fraction=0.0).trainer_kwargs("/tmp/out", eval_split=False).get("eval_strategy") is None
    with pytest.raises(ValueError, match="val_drug_fraction"):
        SFTConfig(val_drug_fraction=1)
    with pytest.raises(ValueError, match="save_steps"):
        SFTConfig(save_steps=0)
    with pytest.raises(ValueError, match="warmup_ratio"):
        SFTConfig(warmup_ratio=-0.1)


def test_init_adapter_loads_the_existing_lora_as_trainable(monkeypatch):
    calls = []

    class Base:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=True)
            self.checkpointed = False
            self.input_grads = False

        def gradient_checkpointing_enable(self):
            self.checkpointed = True

        def enable_input_require_grads(self):
            self.input_grads = True

    class Auto:
        @staticmethod
        def from_pretrained(name, torch_dtype=None, trust_remote_code=None):
            calls.append(("base", name, torch_dtype, trust_remote_code))
            return Base()

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path, is_trainable=False):
            calls.append(("peft", path, is_trainable, model.checkpointed, model.input_grads, model.config.use_cache))
            return "wrapped"

    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def get_peft_model(model, cfg):
        calls.append(("fresh", cfg.kwargs, model.checkpointed))
        return "fresh-model"

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bf16"
    torch_mod.float32 = "fp32"
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = Auto
    peft = types.ModuleType("peft")
    peft.PeftModel = PeftModel
    peft.LoraConfig = LoraConfig
    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    wrapped = load_sft_model(SFTConfig(gradient_checkpointing=True, bf16=True), "/vol/runs/sft/adapter")
    assert wrapped == "wrapped"
    assert calls[0][:3] == ("base", "Qwen/Qwen2.5-3B-Instruct", "fp32")
    assert calls[1] == ("peft", "/vol/runs/sft/adapter", True, True, True, False)
    assert not any(call[0] == "fresh" for call in calls)

    calls.clear()
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
    fresh = load_sft_model(SFTConfig(gradient_checkpointing=False), None)
    assert fresh == "fresh-model"
    assert calls[0][2] == "bf16"
    assert calls[1][0] == "fresh"
    assert calls[1][1]["r"] == 16
    assert calls[1][1]["lora_alpha"] == 32
    assert calls[1][2] is False
    assert not any(call[0] == "peft" for call in calls)
    launch = (ROOT / "fdagym" / "train" / "sft_launch.py").read_text()
    assert "load_sft_model(config, init_adapter)" in launch
    assert launch.index("_blank(init_adapter)") < launch.index("adapter_sha256(init_adapter)")
    assert "init_adapter_sha256" in launch
    assert "is_trainable=True" in launch
    assert _blank(None) is None
    assert _blank("") is None
    assert _blank("  /vol/runs/sft/adapter  ") == "/vol/runs/sft/adapter"


def test_adapter_sha256_hashes_the_safetensors_file(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    payload = b"not a real adapter"
    (adapter / "adapter_model.safetensors").write_bytes(payload)
    assert adapter_sha256(adapter) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        adapter_sha256(tmp_path / "missing")
    hub = "example/fdagym-adapter"
    assert is_hub_adapter(hub) is True
    assert is_hub_adapter(adapter) is False
    assert is_hub_adapter("/vol/runs/missing/adapter") is False
    assert is_hub_adapter("runs/missing/adapter") is False
    assert adapter_sha256(hub) == hashlib.sha256(f"hub:{hub}".encode()).hexdigest()


def test_grpo_save_steps_reaches_the_train_config():
    plain = configure_tool_run(hard=False)
    assert plain.train.save_steps == 50
    assert plain.train.grpo_kwargs("/tmp/out")["save_steps"] == 50
    assert plain.train.max_steps == 150
    hard = configure_tool_run(
        hard=True,
        hard_train_mix={"hard": 0.6, "current": 0.4},
        max_steps=60,
        save_steps=10,
    )
    assert hard.train.save_steps == 10
    assert hard.train.max_steps == 60
    assert hard.train.grpo_kwargs("/tmp/out")["save_steps"] == 10
    assert hard.prompts_per_step == 4
    assert hard.train.gradient_accumulation_steps == 4
    dry = configure_tool_run(hard=True, max_steps=2, save_steps=2, hard_train_mix={"hard": 0.6, "current": 0.4})
    assert dry.train.save_steps == 2
    assert dry.train.max_steps == 2
    with pytest.raises(ValueError, match="save_steps"):
        configure_tool_run(hard=False, save_steps=0)
    defaults = _signature_defaults(MODAL, "train_tools")
    assert defaults["save_steps"] == 50
    body = _function_text(MODAL, "train_tools", "eval_tools")
    assert "save_steps=save_steps" in body


def test_build_warmup_script_keeps_the_quotas_and_does_not_import_modal():
    import importlib.util

    path = ROOT / "scripts" / "build_warmup.py"
    text = path.read_text()
    assert "import modal" not in text
    assert "task_ids_digest" in text
    assert "refusing to write into data/tasks" in text
    spec = importlib.util.spec_from_file_location("build_warmup_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    assert module.N_PER_VARIANT == {"sponsor_drug": 16, "sponsor_faers": 16, "sponsor_count": 16}
    assert module.N_COMPARE == 24
    assert module.SEED == 0
    assert module.COMMITTED_IDS == ROOT / "data" / "warmup" / "mh1_ids.txt"


def test_plan_a_run_sft_is_still_the_plan_a_entrypoint():
    """The legacy single-turn fitter stays. Tool SFT is sft_launch.run_sft."""
    text = (ROOT / "fdagym" / "train" / "sft.py").read_text()
    modal = MODAL.read_text()
    assert "def run_sft(" in text
    assert "from fdagym.train.sft import run_sft" in modal
    assert "from fdagym.train.sft_launch import run_sft" in modal


def _signature_defaults(path: Path, name: str) -> dict:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args.args
            defaults = node.args.defaults
            offset = len(args) - len(defaults)
            return {arg.arg: _literal(default) for arg, default in zip(args[offset:], defaults, strict=True)}
    raise AssertionError(name)


def _literal(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return ast.literal_eval(node)


def _function_text(path: Path, name: str, follower: str) -> str:
    text = path.read_text()
    start = text.index(f"def {name}(")
    end = text.index(f"\ndef {follower}(")
    return text[start:end]
