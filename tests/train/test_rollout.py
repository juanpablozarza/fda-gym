"""Offline rollouts. A scripted generator stands in for vLLM."""

import ast
from pathlib import Path

import pytest

from fdagym.env import FDAEnv
from fdagym.train.context import play_oracle
from fdagym.train.gold import gold_submit
from fdagym.train.rollout import (
    JsonActionCodec,
    QwenToolCodec,
    RolloutLimits,
    TurnGeneration,
    chosen_logprobs,
    generate_with_vllm,
    rollout_batch,
)
from fdagym.train.rollout_dry import CharTokenizer, OracleGenerator, resolve_task_ids, run_rollout_dry
from tests.env.support import ALL_TASKS, EXAMPLE_TASK, JsonSnapshot, task_nausea

ROOT = Path(__file__).resolve().parents[2]


def _run(task_ids, generator=None, codec=None, limits=None, tasks=None):
    tasks = tasks or ALL_TASKS
    codec = codec or JsonActionCodec()
    snapshot = JsonSnapshot()
    tokenizer = CharTokenizer()
    if generator is None:
        chosen = [task for task in tasks if task["id"] in set(task_ids)]
        generator = OracleGenerator(tokenizer, codec, snapshot, chosen)
    rollouts = rollout_batch(
        snapshot,
        tasks,
        task_ids,
        tokenizer,
        generator,
        codec,
        limits,
        run_id="t",
        seed=0,
    )
    return rollouts, generator, tokenizer, codec


def test_two_tasks_score_one_and_the_batch_shrinks():
    rollouts, generator, _tokenizer, _codec = _run(["FG-0001", "FG-0002"])
    by_id = {rollout.task_id: rollout for rollout in rollouts}
    assert by_id["FG-0001"].reward["total"] == 1.0
    assert by_id["FG-0001"].reward["efficiency"] == 1.0
    assert by_id["FG-0001"].n_turns == 3
    assert by_id["FG-0002"].reward["total"] == 1.0
    assert by_id["FG-0002"].n_turns == 2
    assert generator.batch_sizes == [2, 2, 1]
    assert generator.temperatures == [0.7, 0.7, 0.7]
    assert by_id["FG-0001"].trajectory["seed"] == 0
    assert by_id["FG-0002"].trajectory["seed"] == 1


def test_loss_mask_is_one_only_on_model_tokens():
    rollouts, _generator, tokenizer, _codec = _run(["FG-0001"])
    rollout = rollouts[0]
    assert len(rollout.loss_mask) == len(rollout.completion_ids) == len(rollout.logprobs)
    assert rollout.attention_mask == tuple(1 for _ in rollout.completion_ids)
    assert sum(rollout.loss_mask) < len(rollout.loss_mask)
    cursor = 0
    for span in rollout.spans:
        chunk = rollout.loss_mask[cursor : cursor + len(span.token_ids)]
        if span.role == "model":
            assert chunk == tuple(1 for _ in span.token_ids)
            assert span.text
            assert rollout.logprobs[cursor] == -0.2
        else:
            assert span.role == "tool"
            assert chunk == tuple(0 for _ in span.token_ids)
            assert "<tool_response>" in span.text
            assert all(value is None for value in rollout.logprobs[cursor : cursor + len(span.token_ids)])
        assert tokenizer.decode(span.token_ids) == span.text
        cursor += len(span.token_ids)
    assert any(span.role == "tool" for span in rollout.spans)
    assert not any(span.role == "injected" for span in rollout.spans)


def test_qwen_codec_scores_one_on_the_same_runner():
    rollouts, _generator, _tokenizer, codec = _run(["FG-0001"], codec=QwenToolCodec())
    rollout = rollouts[0]
    assert codec.name == "qwen"
    assert rollout.reward["total"] == 1.0
    assert rollout.n_turns == 3
    model = "".join(span.text for span in rollout.spans if span.role == "model")
    assert "<tool_call>" in model
    assert "search_drug_label" in model
    assert "<tool_response>" not in codec.prompt(EXAMPLE_TASK["question"])


def test_submit_without_a_tool_call_is_format_only():
    snapshot = JsonSnapshot()
    env = FDAEnv(snapshot, ALL_TASKS, run_id="gold")
    play_oracle(env, EXAMPLE_TASK)
    gold = gold_submit(EXAMPLE_TASK, env.steps)
    codec = JsonActionCodec()
    tokenizer = CharTokenizer()

    class SubmitNow:
        def generate(self, prompt_token_ids, *, max_tokens, temperature):
            text = codec.action_text({"tool": "submit_answer", "arguments": gold})
            tokens = tuple(tokenizer.encode(text)[: max_tokens - 1] + [tokenizer.eos_token_id])
            return [TurnGeneration(tokens) for _ in prompt_token_ids]

    rollouts, _generator, _tokenizer, _codec = _run(["FG-0001"], generator=SubmitNow())
    reward = rollouts[0].reward
    assert reward["format"] == 1.0
    assert reward["citation"] == 0.0
    assert reward["correctness"] == 0.0
    assert reward["total"] == 0.1
    assert rollouts[0].n_turns == 1
    assert all(span.role == "model" for span in rollouts[0].spans)


def test_garbage_text_scores_zero():
    tokenizer = CharTokenizer()

    class Garbage:
        def generate(self, prompt_token_ids, *, max_tokens, temperature):
            tokens = tuple(tokenizer.encode("not json at all") + [tokenizer.eos_token_id])
            return [TurnGeneration(tokens) for _ in prompt_token_ids]

    rollouts, _generator, _tokenizer, _codec = _run(["FG-0002"], generator=Garbage(), limits=RolloutLimits(max_turns=4))
    rollout = rollouts[0]
    assert rollout.reward["total"] == 0.0
    assert rollout.terminated_by == "error"
    assert rollout.n_turns == 1
    assert rollout.loss_mask == tuple(1 for _ in rollout.completion_ids)


def test_force_submit_masks_the_injected_tokens():
    tokenizer = CharTokenizer()
    codec = JsonActionCodec()

    class SearchOnce:
        def generate(self, prompt_token_ids, *, max_tokens, temperature):
            text = codec.action_text(
                {"tool": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}}
            )
            tokens = tuple(tokenizer.encode(text) + [tokenizer.eos_token_id])
            return [TurnGeneration(tokens) for _ in prompt_token_ids]

    limits = RolloutLimits(max_turns=1, force_submit=True)
    rollouts, _generator, _tokenizer, _codec = _run(["FG-0001"], generator=SearchOnce(), limits=limits)
    rollout = rollouts[0]
    roles = [span.role for span in rollout.spans]
    assert roles == ["model", "tool", "injected"]
    assert rollout.terminated_by == "submit"
    assert rollout.reward["total"] == 0.1
    assert rollout.trajectory["final_answer"]["answer"] == "NOT_FOUND"
    injected = rollout.spans[-1]
    start = len(rollout.completion_ids) - len(injected.token_ids)
    assert rollout.loss_mask[start:] == tuple(0 for _ in injected.token_ids)
    assert "submit_answer" in injected.text


def test_force_submit_leaves_a_real_submit_alone():
    limits = RolloutLimits(force_submit=True)
    rollouts, _generator, _tokenizer, _codec = _run(["FG-0002"], limits=limits)
    rollout = rollouts[0]
    assert rollout.reward["total"] == 1.0
    assert not any(span.role == "injected" for span in rollout.spans)


def test_unfinished_episode_adds_no_injected_tokens():
    tokenizer = CharTokenizer()
    codec = JsonActionCodec()

    class SearchOnce:
        def generate(self, prompt_token_ids, *, max_tokens, temperature):
            text = codec.action_text(
                {"tool": "query_adverse_events", "arguments": {"generic_name": "metformin", "mode": "count"}}
            )
            tokens = tuple(tokenizer.encode(text) + [tokenizer.eos_token_id])
            return [TurnGeneration(tokens) for _ in prompt_token_ids]

    rollouts, _generator, _tokenizer, _codec = _run(
        ["FG-0002"],
        generator=SearchOnce(),
        limits=RolloutLimits(max_turns=1, force_submit=False),
    )
    rollout = rollouts[0]
    assert [span.role for span in rollout.spans] == ["model", "tool"]
    assert rollout.terminated_by == "error"
    assert rollout.reward["format"] == 0.0
    assert rollout.reward["total"] == 0.0


def test_context_limit_stops_the_episode():
    tokenizer = CharTokenizer()
    codec = JsonActionCodec()
    prompt = codec.prompt(task_nausea()["question"])
    room = 8
    limits = RolloutLimits(max_turns=3, max_new_tokens=100, max_context_tokens=len(tokenizer.encode(prompt)) + room)

    class Long:
        def generate(self, prompt_token_ids, *, max_tokens, temperature):
            assert max_tokens <= room
            tokens = tuple(range(2, 2 + max_tokens))
            return [TurnGeneration(tokens) for _ in prompt_token_ids]

    rollouts, _generator, _tokenizer, _codec = _run(["FG-0002"], generator=Long(), limits=limits)
    rollout = rollouts[0]
    assert len(rollout.completion_ids) <= room
    assert len(rollout.prompt_ids) + len(rollout.completion_ids) <= limits.max_context_tokens
    assert rollout.terminated_by == "error"


def test_tool_text_is_clipped_before_it_is_appended():
    limits = RolloutLimits(max_tool_chars=40)
    rollouts, _generator, _tokenizer, _codec = _run(["FG-0001"], limits=limits)
    tool_spans = [span for span in rollouts[0].spans if span.role == "tool"]
    assert tool_spans
    for span in tool_spans:
        assert "...[truncated]" in span.text
        payload = span.text.split("<tool_response>\n", 1)[1].rsplit("\n</tool_response>", 1)[0]
        assert len(payload) <= 40


def test_duplicate_task_ids_are_independent_episodes():
    rollouts, _generator, _tokenizer, _codec = _run(["FG-0002", "FG-0002"])
    assert [rollout.reward["total"] for rollout in rollouts] == [1.0, 1.0]
    assert [rollout.trajectory["seed"] for rollout in rollouts] == [0, 1]
    assert rollouts[0].trajectory["id"] != rollouts[1].trajectory["id"]


def test_qwen_parser_accepts_string_arguments_and_a_bare_answer():
    codec = QwenToolCodec()
    parsed = codec.parse(
        '<tool_call>\n{"name": "search_approvals", "arguments": "{\\"generic_name\\": \\"metformin\\"}"}\n</tool_call>'
    )
    assert parsed == {"tool": "search_approvals", "arguments": {"generic_name": "metformin"}}
    bare = codec.parse('noise {"answer": "2017", "citations": []}')
    assert bare["tool"] == "submit_answer"
    assert bare["arguments"]["answer"] == "2017"
    plain = JsonActionCodec().parse('{"tool": "search_approvals", "arguments": {"generic_name": "metformin"}}')
    assert plain["tool"] == "search_approvals"


def test_vllm_adapter_sends_token_prompts_and_reads_logprobs():
    class Logprob:
        def __init__(self, logprob):
            self.logprob = logprob

    class Output:
        def __init__(self):
            self.token_ids = [10, 11]
            self.logprobs = [{10: Logprob(-0.25), 3: Logprob(-2.0)}, {11: Logprob(-0.5)}]
            self.text = "ignored"

    class Result:
        def __init__(self):
            self.outputs = [Output()]

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            self.prompts = prompts
            self.params = sampling_params
            self.use_tqdm = use_tqdm
            return [Result(), Result()]

    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    llm = FakeLLM()
    turns = generate_with_vllm(llm, [[1, 2, 3], [4]], max_tokens=32, temperature=0.7, sampling_params_cls=Params)
    assert llm.prompts == [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4]}]
    assert llm.params.kwargs == {"n": 1, "temperature": 0.7, "max_tokens": 32, "logprobs": 1}
    assert llm.use_tqdm is False
    assert turns[0].token_ids == (10, 11)
    assert turns[0].logprobs == (-0.25, -0.5)
    assert chosen_logprobs([7], [{8: Logprob(-1.0)}]) == [None]
    assert chosen_logprobs([7], None) == [None]


def test_rollout_module_does_not_import_torch_trl_or_vllm():
    tree = ast.parse((ROOT / "fdagym" / "train" / "rollout.py").read_text())
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "vllm" not in imported
    assert "torch" not in imported
    assert "trl" not in imported


def test_snapshot_sheet_defaults_come_from_train_not_fg0001():
    train_path = ROOT / "data" / "tasks" / "train.jsonl"
    test_path = ROOT / "data" / "tasks" / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        pytest.skip("150-drug task sheet absent")
    import json

    train = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    test = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    train_ids = {task["id"] for task in train}
    test_ids = {task["id"] for task in test}
    assert "FG-0001" in test_ids
    assert "FG-0001" not in train_ids
    ids = resolve_task_ids(train + test, None)
    assert ids == resolve_task_ids(train, None)
    assert all(task_id in train_ids for task_id in ids)
    assert "FG-0001" not in ids
    assert resolve_task_ids(train + test, ["FG-0001"]) == ["FG-0001"]


def test_resolve_task_ids_uses_loaded_rows_not_stale_fixture_defaults():
    tasks = [
        {"id": "FG-0343", "type": "approval"},
        {"id": "FG-0344", "type": "approval"},
        {"id": "FG-0001", "type": "approval"},
        {"id": "FG-1503", "type": "label_open"},
    ]
    assert resolve_task_ids(tasks, None) == ["FG-0343", "FG-0344"]
    assert resolve_task_ids(tasks, ["FG-0001"]) == ["FG-0001"]
    assert resolve_task_ids(tasks, None, n=1) == ["FG-0343"]
    with pytest.raises(ValueError, match="FG-0002"):
        resolve_task_ids(tasks[:2], ["FG-0001", "FG-0002"])
    with pytest.raises(ValueError, match="no gradable"):
        resolve_task_ids([{"id": "FG-1503", "type": "label_open"}], None)


def test_dry_run_on_two_tasks():
    report = run_rollout_dry(JsonSnapshot(), ALL_TASKS, ["FG-0001", "FG-0002"])
    assert report["codec"] == "qwen"
    assert [episode["task_id"] for episode in report["episodes"]] == ["FG-0001", "FG-0002"]
    assert all(episode["reward"]["total"] == 1.0 for episode in report["episodes"])
    assert all(episode["dense_total"] == 1.0 for episode in report["episodes"])
    assert all(episode["training_reward"] == pytest.approx(1.05) for episode in report["episodes"])
    assert report["dense_reward"] is True
    assert all(episode["tool_tokens"] > 0 for episode in report["episodes"])
    assert all(episode["loss_on"] == episode["model_tokens"] for episode in report["episodes"])
    assert report["batch_sizes"] == [2, 2, 1]
