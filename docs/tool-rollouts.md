# GRPO with live tool calls

Status: **implementation, not a GPU run.** JP approved the eight decisions on 2026-10-18, then a tool-protocol baseline (`eval-tools-base`) and a first GRPO run of **150 steps**. This branch subclasses `GRPOTrainer` on TRL 0.21. It does not start Modal or RunPod. JP approves every training run.

Plan A pastes `oracle_calls` into the prompt and the model only writes `submit_answer`. That packing truncates the record that holds the answer (see the 2026-10-18 data audit). Live rollouts drop the pasted documents. The prompt is the question plus the tool schema. During a GRPO step the model calls `search_drug_label`, `get_label_section`, `query_adverse_events`, and `search_approvals` against `FDAEnv` on `DiskSnapshot`, then `submit_answer`. `fdagym.rewards.score` grades that episode.

`FDAEnv.step` and the LiteLLM loop in `fdagym/agents/litellm_agent.py` already do this for eval. Training does not. `GRPOTrainer` in TRL 0.21 generates one completion and stops.

## Recommendation

Stay on **`trl==0.21.0`** and subclass `GRPOTrainer`. Put the multi-turn loop in our own runner (`fdagym/train/rollout.py`), and teach the subclass to call that runner. Do not upgrade TRL in the same change.

The subclass is `ToolGRPOTrainer` in `fdagym/train/grpo_tools.py`. The runner does not import TRL or vLLM.

## Approach

### A. Subclass `GRPOTrainer` on 0.21 (chosen)

Colocate vLLM already lives on the trainer as `trainer.llm` (`vllm.LLM` from vLLM 0.8.5). One GRPO step syncs LoRA weights into that engine (`_move_model_to_vllm`), then generates. The override replaces the single `llm.generate` inside `_generate_and_score_completions` with the batched runner: generate, parse one action, `FDAEnv.step`, append the tool result, repeat.

Pros:

- The image stays `torch==2.6.0`, `transformers==4.53.0`, `trl==0.21.0`, `vllm==0.8.5`. That is the stack `modal_train.py` and the RunPod image already pin, and the only stack Plan A has run.
- `FDAEnv` stays the environment. Reward, citation ids, the 6-call budget, and `submit_answer` validation do not get reimplemented as TRL tool functions.
- The runner's return value is the contract: prompt ids, completion ids, a loss mask, and an attention mask. A later TRL upgrade can consume the same object.

Cons:

- `_generate_and_score_completions` and `compute_loss` are private. In 0.21 the same `completion_mask` is the loss mask and the attention mask (`attention_mask = cat(prompt_mask, completion_mask)`). Writing zeros on tool tokens in that tensor **hides the tool output from the answer tokens**, so the policy is trained as if it never saw the JSON. The subclass has to keep two masks: attention over every real token, loss only on model-written tokens. Getting that wrong is silent. The loss still moves; it just conditions on the wrong context.
- With `num_iterations == 1` (our config), 0.21 does not read logprobs from vLLM. It recomputes them in the training forward. Those logprobs are only meaningful if the attention mask matches the tokens the engine scored. The runner still records engine logprobs so we can check the alignment before we trust the loss.

### B. Upgrade TRL (rejected for this change)

| Version | What it adds | What it breaks |
|---|---|---|
| 0.21 (pinned) | Colocate vLLM, `max_prompt_length` | Single-turn generate only |
| 0.25 (2025-11-06) | Experimental `rollout_func` | Signature was `(prompts, GRPOConfig, processing_class)`. Tool-token masking was an open TODO on the PR. Docs showed vLLM **server** mode, which wants a second GPU |
| 0.28 | — | Drops `max_prompt_length`. `select_grpo_kwargs` already strips it |
| 0.29 (2026-02-25) | `environment_factory`: env methods are the tools, `reset()` required, optional `get_reward()` | Still marked experimental. `FDAEnv.step` is one action channel, not a method per tool |
| 1.0 (2026-03) | `rollout_func(prompts, trainer) -> {prompt_ids, completion_ids, logprobs}`. Colocate is the default. Current main splits `completion_mask` (attention) from `tool_mask` (loss) | vLLM extra is `>=0.10.2,<=0.17.1` |

Version-pin impact of landing on 1.0, which is the first release where the mask split is actually the supported path: **vLLM 0.10.0+ pins `torch==2.7.1`** (and a newer transformers). vLLM 0.8.5 cannot stay. The `[gpu]` extra in `pyproject.toml` is the pin list, and `modal_train.py` installs that extra, so they move together. Colocate weight-sync also changed (`sleep`, engine internals). Plan A numbers would come from a different stack than the tool-rollout numbers, on top of a different task.

`environment_factory` does not remove the need for a runner. It wants each tool to be a Python callable with a Google-style docstring, and it builds the transcript through the chat template. That is the retokenization path TRL spent 0.29–1.0 fixing (tool mask shorter than `completion_ids` after `apply_chat_template`). `FDAEnv` already has the schema, the snapshot, and the reward. Wrapping it as eight callables would throw that away to fit a hook that is still marked experimental.

Revisit the upgrade after the runner has a logged GPU step and the 0.21 subclass's two masks match the runner on a single batch. Until then the pin stays.

## Tool-call format, turns, budgets

Two codecs implement the same runner interface (`prompt`, `parse`, `tool_message`, `action_text`). The GPU run and the CPU dry run use Qwen's native template. The JSON codec stays as a test double.

**Qwen2.5-Instruct native (the GPU run).** The chat template is the Hermes-style one shipped in the tokenizer: the model writes

```text
<tool_call>
{"name": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
</tool_call>
```

and the tool result is a user turn of `<tool_response>...</tool_response>`, followed by a masked `<|im_start|>assistant\n` so the next sample is an assistant continuation. vLLM parses this with `--tool-call-parser hermes` (`vllm>=0.6`). Qwen2.5-Instruct was trained on that string.

**JSON action (what the dry run speaks).** One object, the same shape `FDAEnv._parse_action` already accepts:

```text
{"tool": "search_drug_label", "arguments": {"generic_name": "warfarin"}}
```

`submit_answer` is either that wrapper or a bare `{"answer", "citations"}` object. No special tokens, so a character tokenizer can round-trip it in tests. This is a fallback and a test double, not the format the 3B saw in pretraining.

Both codecs execute **one** parsed call per model turn (the first `<tool_call>` or the first JSON object). Extra calls in the same generation stay in the text, and they are model tokens, but they do not hit the env.

Recommended budgets, already the defaults on `RolloutLimits`. They are arguments, not constants baked into the loop:

| Knob | Default | Why |
|---|---|---|
| `tool_budget` | 8 | What `FDAEnv` enforces on a new tool-mode run, and the number in the system prompt. `6` reproduces a v3 run. The baseline agent still defaults to 6 when it constructs `FDAEnv` with no budget |
| `prompt_version` | `v4` | v4 is JP's v3 prompt plus the `query_adverse_events` argument line, with the budget sentence filled from `tool_budget`. `v3` is the previous text (budget number still follows `tool_budget`) |
| `max_turns` | 9 | One past `tool_budget`, so the model can still `submit_answer` after the last tool call. `force_submit` is off. A v3 run sets `tool_budget=6`, which makes this 7 |
| `max_new_tokens` | 256 | Same cap as Plan A. A tool call is a short JSON object. A gold `submit_answer` for the warfarin fixture is 216 characters, so 256 BPE tokens covers the citation |
| `max_tool_chars` | 3500 | Same clip as Plan A's `_clip`, applied to the string the model sees. The env keeps the full tool output, which is what citation matching reads |
| `max_context_tokens` | 8192 | Prompt is the question plus the schema (about 1–2k tokens), not four packed documents. TRL 0.21 sizes the engine as `max_prompt_length + max_completion_length` (2048 + 6144) |
| `temperature` | 0.7 (tool run: 0.8 via `ToolTrainConfig`) | GRPO sampling. Eval uses 0 |
| `force_submit` | false | A missing `submit_answer` ends as an invalid submit and scores format 0. No tokens are injected |

A tool string that does not fit in the remaining context is not sliced mid-JSON. The episode ends. The model is not shown a partial record.

## Loss masking

Only tokens the model sampled get loss. Tool results and any string we inject (a forced submit, if that flag is turned on) get loss 0 and attention 1. Padding, which the runner does not add, is attention 0 and loss 0.

The runner builds that mask from spans as it appends them. It does not recover the mask by scanning the decoded text, and it does not retokenize a model span. vLLM is called with `prompt_token_ids`. The returned ids are concatenated. The tool message is the only thing we encode ourselves. That avoids the BPE merge drift TRL hit when it decoded a tool call and ran it back through `apply_chat_template`.

`loss_mask[i] == 1` for a model span, including the EOS token that ended the turn. `attention_mask` is 1 for every completion token. Prompt tokens are not in the completion. GRPO already ignores them.

## Reward

The runner does not call `play_oracle`. When the episode ends, the reward is `info["reward"]` from `FDAEnv`, which is `fdagym.rewards.score`:

`0.60 * correctness + 0.25 * citation + 0.10 * format + 0.05 * efficiency`

Efficiency already uses `min_tool_calls / calls`, and only when correctness is above 0. A citation counts only if its id showed up in a tool result of **this** episode. A model that writes the gold JSON without calling the tool scores 0.10 (format only). That is the opposite of Plan A, where `score_completion` plays the oracle before it reads the completion.

Shaping, approved and small: if the episode parsed at least one non-submit tool call, the GRPO reward is `env_total + tool_parse_bonus`. The default bonus is **0.05, once per episode**, not once per call. Six parses would otherwise add 0.30 and swamp the 0.05 efficiency term. `tool_parse_bonus` is the knob (`ToolTrainConfig`, default 0.05; pass 0 to disable). A parsed call still counts if the env later returns an error. `submit_answer` does not count; format already pays 0.10. The bonus is **not** written into `trajectory.reward`. That field is capped at 1 by the schema. A perfect episode is env total 1.0 and training reward 1.05. Advantages use the shaped float.

An unfinished episode is closed with an invalid `submit_answer` so `trajectory` exists; those closing tokens are not added to the completion.

## Warm-up

The 3B will emit malformed calls for a long time. Every one of those episodes scores 0 or 0.10, the group advantage is 0, and LoRA does not move. That is the same failure mode as Plan A groups that all score 0.10, reached earlier.

Warm-up, implemented as `fdagym/train/sft_tools.py` and `modal_train.py --action sft`:

1. For each train task, play the scripted trajectory on `DiskSnapshot`, then `gold_submit`. Approval and FAERS tasks replay `oracle_calls`. Boxed-warning and composite tasks search labels without `has_section` and read `boxed_warning` only when a label of that drug lists the section.
2. Render the transcript in the chosen tool format. Model spans are the action strings. Tool spans are the clipped observations.
3. Supervised loss on the model spans only, with the same mask the runner uses. Same LoRA (r=16, α=32) on Qwen2.5-3B-Instruct.
4. About one pass over the ~1,230 train rows. Short job on the A100 (well under an hour, on the order of $2 at the rates below).
5. Start GRPO from that adapter. The GRPO prompt is still only the question and the schema, so the gold answer is not sitting in the prompt. TEST drugs stay held out. The risk is the usual one: the adapter can memorize train-set tool arguments. TEST is the number that matters, same as Plan A.

GRPO from the raw 3B is the alternative if we would rather not spend the warm-up on teaching oracle arguments. Expect a long stretch of zero advantage first.

## Memory, step time, Modal cost

Target: **150 steps, Qwen2.5-3B-Instruct, LoRA, 1× A100 80GB on Modal.** No job was launched for this estimate. The 6–8 s single-turn step is the figure from the data audit, not a new measurement. Memory below does not change with the step count.

Published 3B shape used for the arithmetic: 36 layers, hidden 2048, intermediate 11008, 2 KV heads, head dim 128, vocab 151936. Weights in bf16 are about **6.2 GB**. LoRA on `q,k,v,o,gate,up,down` at r=16 is about **30M parameters**. AdamW on those parameters is about **0.24 GB**.

Colocate keeps two weight copies. vLLM's pool is `vllm_gpu_memory_utilization * 80 GB` and holds its own copy plus the KV cache. KV is 36 KiB per token (2 × 36 × 2 × 128 × 2 bytes). Eight sequences at 8192 tokens are about **2.3 GB** of KV, so the cache is not the tight part. Lower the utilization from Plan A's 0.35 to **0.30** (24 GB pool) to leave room for a longer training forward.

Training forward, the shape used for the dollar estimate: **8 generations, gradient checkpointing on** (Plan A leaves it off for the 3B, because completions are 256 tokens), sequence about 6k, completion about 4k.

| Piece | About |
|---|---|
| vLLM pool at 0.30 | 24 GB (6.2 GB of that is the engine's weights) |
| HF weights + LoRA AdamW | 6.4 GB |
| Checkpointed activations (`2 * layers * batch * seq * hidden * 2` bytes) | ~14 GB |
| Logits for the completion (`8 * 4096 * 151936 * 2` bytes), which 0.21 materializes for the whole local batch | ~10 GB |
| Allocator headroom | ~4 GB |
| **Total** | **~58 GB of 80** |

Turning checkpointing off, or letting a completion run out to the full 8192, pushes the logits and activations through the remaining 20 GB. The first GPU run should checkpoint, and it should stay at 8 generations only while completions stay near 4k tokens. L40S (48 GB) is the wrong card for this loop.

Step time. TRL 0.21's `LLM(...)` does not pass `enable_prefix_caching`, so each turn re-prefills. The snapshot read is local and is not the cost. A step that dies on a bad first call is close to today's 6–8 s. A step whose episodes take two or three real turns is about **25 s**. A full 6-tool episode is about **45 s**. These are planning numbers, not a measured GPU step.

150 steps at 25 s is **1.0 hour**. A 400-step run would have been 2.8 hours.

Modal's public rate (A100 80GB) is **$2.50 per GPU-hour** ($0.000694/s). The tool functions request 64 GiB of RAM at $0.00000222 per GiB-second, about $0.51/hour. Together that is about **$0.000837/s**.

| Step | Seconds | Cost per step |
|---|---|---|
| Parse fails on the first call | ~8 s | **~$0.007** |
| Two or three real turns (planning step) | ~25 s | **~$0.021** |
| All 6 tool calls | ~45 s | **~$0.038** |

150 steps at $0.021 is **about $3.15** (1.0 h). The range is **about $1–$6**. SFT is one epoch, on the order of **$1–2**. Each tool-protocol TEST eval (about 324 tasks, base or adapter) is about **0.5–1.5 h and $1.50–$4**. A cold volume also downloads the 3B weights once; that is outside these jobs. None of this was measured on a GPU.

## What this branch implements

`fdagym/train/rollout.py` runs a batch of episodes. Each episode has its own `FDAEnv`. Each turn calls a generator with the token ids so far, appends the model span, steps the env, and appends the tool span under the loss mask. The Qwen tool span includes the next `<|im_start|>assistant\n`. `generate_with_vllm` adapts a vLLM 0.8 `LLM.generate` (`prompt_token_ids`, `SamplingParams`, `logprobs=1`) without importing vLLM until that function runs.

`ToolGRPOTrainer` overrides `_generate_and_score_completions` and `_compute_loss`. Attention uses `completion_mask`. The policy loss, KL, and clip ratios use `loss_mask`. Liger is refused. Prompts are the question plus the tool schema: no packed docs, no distractors, same `stage_pool` as Plan A, `label_open` skipped. A missing SFT adapter raises. A `checkpoint-*` directory resumes and is not stacked on the warm-up.

SFT (`fdagym/train/sft_tools.py`, `SFTConfig`) plays the scripted trajectory through the same runner, drops any row below 0.97 or over 8192 tokens, holds out 5% of train drugs as eval, and trains one epoch at lr `1e-4`, batch 4, grad accumulation 2, the same LoRA. Eval during SFT is teacher-forced (loss and tool-call match). The TEST eval (`fdagym/train/eval_tools.py`) is greedy and multi-turn. `eval-tools-base` scores the raw 3B with no adapter (`kind: baseline`, run `baseline-qwen3b-tools`). `eval-tools` scores the GRPO adapter. Both write `protocol: tool_rollout`. The shaped +0.05 bonus is training-only. TEST `summary.json` stores the strict env total on `models[].mean_reward` and the dense total on `dense_reward`.

`search_approvals` takes `generic_name` or `application_number`, plus optional `page` (default 1). The application number may be written `NDA020977`, `NDA 020977`, `020977`, or the same way for a BLA or ANDA. That lookup reads Drugs@FDA records already stored under the generic-name search. It does not add an openFDA query. Each returned application is one compact line: NDA and BLA before ANDA, products collapsed to brand, dosage form, and route, and `submissions` limited to the original approval (`n_supplements` counts the rest). A page is packed so the observation JSON stays within 3,500 characters. Unknown numbers return an empty result list.

CPU, no GPU:

```bash
uv run python scripts/tool_rollout_dry.py
uv run python scripts/sft_warmup.py --dry-run
uv run python scripts/grpo_tools.py --dry-run
uv run python scripts/eval_tools.py --dry-run
uv run python scripts/check_approval_reachability.py
uv run pytest -q tests/train
```

`check_approval_reachability.py` writes `results/approval-reachability.json`. For every gold application id on an approval task it runs one `search_approvals` by that number and records whether the id and the answer text sit in the first 3,500 characters the model would see. It also replays FG-0002 and expects reward 1.0.

Modal, only after JP approves the run. Commands and the per-step cost are in [docs/train.md](train.md).

## Decisions (JP, 2026-10-18)

1. Stay on TRL 0.21 and subclass `GRPOTrainer`. Do not pay the torch 2.7.1 / vLLM ≥0.10.2 pin.
2. Qwen `<tool_call>` / `<tool_response>`. The JSON codec remains a test double.
3. Up to 6 tool calls (7 generation turns), 256 new tokens, 3500-character clip, 8192 context.
4. `force_submit` stays off. A missing submit scores 0.
5. Mask tool outputs and any forced or canonicalized span, including the assistant continuation header. Model tokens, including EOS, keep loss 1.
6. Shaping bonus 0.05, once per episode, when a non-submit tool call parses. Configurable. Not stored on the trajectory.
7. SFT on scripted tool trajectories before GRPO. Boxed-warning and composite label steps are answer-independent. One epoch, keep reward >= 0.97, 5% of train drugs held out for teacher-forced eval. GRPO does not start from the raw base.
8. First GPU shape: 8 generations, gradient checkpointing, A100-80GB, vLLM memory fraction 0.30. The first run is 150 steps.
9. Dense reward (JP, 2026-10-19), wired from `track/train` at `61f059c`. Tool GRPO passes `TrainConfig.dense_reward` explicitly. `tool_train_config()` sets it true. `FDAEnv` still scores with the strict default, so the trajectory total stays comparable to the 0.03 gate. Eval reports both totals: the strict env total on `models[].mean_reward`, and the dense total on `summary["dense_reward"]`. The +0.05 parse bonus stays training-only, added to the dense total.
10. Tool run decisions (JP, 2026-10-19), implemented, not launched: tool SFT from the base model; GRPO prompts from `type_mix_pool` (then 30% composite with yes/no balanced, 30% boxed_warning, 20% faers_top_reaction, 15% approval, 5% faers_count; not stage 3); composite half credit (0.5 per correct half) in the dense training reward only, eval strict; temperature 0.8 with 8 generations; tied-group fraction, parse rates, and termination reasons logged to Weights & Biases; stop at step 20 if the mean parse rate is below 0.9; gate +0.03 strict over `eval-tools-base`; $20 cap. Later defaults (distill GRPO): temperature **0.65**, composite **out of the pool**, `boxed_warning` still floored. Details in [RUNBOOK.md](RUNBOOK.md).

