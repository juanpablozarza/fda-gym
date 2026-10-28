# Modal runbook and lessons from the 2026-10-18 Plan A run

## Results that night (324 test questions, strict score)
| model | strict score | pass | citation |
|---|---|---|---|
| Base Qwen2.5-3B-Instruct | 0.0185 | 0% | 0.00 |
| SFT warm-up (`sft-plan-a-qwen3b`) | 0.575 | 53.7% | 0.62 |
| GRPO 150 steps from warm-up (`grpo-qwen3b-sft150`) | 0.839 | 80.6% | 0.87 |

By type (warm-up -> RL score): faers_top_reaction 0.39 -> 0.95, approval 0.73 -> 0.86,
faers_count 0.97 -> 1.00, boxed_warning 0.11 -> 0.41, composite 0.31 -> 0.35 (0% pass).
Artifacts (adapters, evals, charts) are saved on the box at `/workspace/bio-rl-artifacts/2026-10-18/`.
The Modal volume was deleted afterwards to stop storage cost.

## Lessons
1. **SFT warm-up first.** Without it every GRPO group scored the same (zero variance, no learning).
   The warm-up taught the answer format and citation copying: 0.0185 -> 0.575, then RL -> 0.839.
2. **Groups saturate.** By about step 100 most groups were fully tied (`frac_reward_zero_std` = 1), reward ~0.98.
   Steps after that add little. Next time use harder tasks (boxed_warning, composite) or ~100 steps.
3. **Composite questions always answer "no"** on the boxed warning half, even when the gold is yes.
   Citations are perfect there; the model is not reading the boxed warning section. Add more positive
   boxed warning examples to SFT and/or reward the yes/no half separately.
4. **boxed_warning is weak** (32% pass). Same root cause as 3.
5. **Regressions are citation quotes**, not answers: RL sometimes copies a quote longer than the field or
   below the 0.90 fuzzy match, and the strict grader gates correctness on a valid citation.

## Infra lessons (why the pipeline looks the way it does)
- `modal run` without `--detach` dies when the laptop disconnects. Even `--detach` was cancelled by a local
  signal once. **Always**: `modal deploy modal_train.py`, then `python scripts/launch_pipeline.py ...`
  which `spawn()`s on the deployed app and exits.
- A deployed function was restarted once and re-ran the whole chain from SFT. So stages are idempotent
  (`fdagym/train/pipeline.py`): a stage whose output adapter exists is skipped and never overwritten;
  the runner refuses to start when `status.json` says done / gate-failed / failed (use `--force` after checking).
- `retries=0` everywhere, long timeouts (pipeline 16 h, train 4 h), `scaledown_window=60` on evals so idle
  GPUs shut down fast.
- Watch progress from the volume: `runs/<pipeline>/status.json` and `pipeline.log`.
  Step metrics (reward, strict, parse rates, memorization) are in Weights & Biases, not on the volume.
- When done: `modal app stop fdagym-grpo-3b`, download what you need, delete the volume if spend must be $0.

## Weights & Biases

Training metrics are the W&B run whose name and id are the job's `run_id` (`sft-qwen3b-tools-v4`, `grpo-qwen3b-tools`, `grpo-qwen3b-tools-eval`, and the Plan A ids). A checkpoint restart uses `resume='allow'` on that same id, and steps already stored are not logged again.

The API key is only the Modal secret named `wandb`. Do not put it in the repo, in a function argument, or in the image.

```bash
# From a laptop that is already `modal` logged in. The value is the key from https://wandb.ai/authorize.
modal secret create wandb WANDB_API_KEY=...
# Optional, same secret, if the account default entity is not the one you want:
# modal secret create wandb WANDB_API_KEY=... WANDB_ENTITY=your-entity WANDB_PROJECT=bio-rl
```

`WANDB_PROJECT` defaults to `bio-rl` when the secret and the flag are both empty. Override with `--wandb-project` / `--wandb-entity` on `modal run modal_train.py` or `python scripts/launch_pipeline.py`, or export `WANDB_PROJECT` and `WANDB_ENTITY` before the launcher (it forwards them). A machine with no key (local tests, CI) trains with W&B off. A W&B error mid-run is logged and training continues.

Runs show up at `https://wandb.ai/<entity>/bio-rl/runs/<run_id>`. The chart (reward, strict, memorization gap, flags) is local:

```bash
export WANDB_API_KEY=...   # same key, for the API read. Do not commit it.
uv run python scripts/wandb_chart.py grpo-qwen3b-tools --entity your-entity
# writes results/grpo-qwen3b-tools-chart.png
```

Eval deliveries are unchanged: `summary.json`, `leaderboard.json`, `tasks.json`, and the per-model JSONL. Pull those with `modal volume get`. Checkpoints and adapters still land on the volume.

## Tool-call run (decisions JP, 2026-10-19)
Not launched. JP approves the deploy and the launch.

| Decision | What the code does |
|---|---|
| Start | Tool SFT warm-up from the **base** 3B on oracle tool trajectories. Not last night's Plan A adapter: its prompt said "Do not call tools", so it would submit without searching. `--init-adapter` stays available but is not the plan. |
| Answer-independent oracle | Boxed-warning and composite SFT trajectories do **not** replay stored `oracle_calls` for the label steps. Those stored calls set `has_section=boxed_warning` only when the answer is yes. The script searches the generic name with no section filter, keeps labels of that drug (single ingredient, plus the product type named in the question), and calls `get_label_section` only when one of those labels lists `boxed_warning`. The branch reads the search result, not the answer. Stored `oracle_calls` stay the short path that scores 1.0 on the integration gate. |
| SFT keep | Keep a scripted row when its reward is **>= 0.97** (and it fits 8192 tokens). Easy "yes" boxed tasks pay one extra section read against `min_tool_calls` of 1 and score 0.975. On the 2026-10-15 train split a 1.0 cutoff keeps 1,132 of 1,160 gradable rows; 0.97 keeps all 1,160. |
| SFT validation | Hold out **5% of train drugs** (`SFTConfig.val_drug_fraction`, seed `SFTConfig.seed`, split by drug so a drug's tasks cannot leak). Seed 0 holds out 6 of 120 drugs (54 tasks). The fit uses the other 1,106 unique rows. Eval is a forward pass every 50 steps: eval loss, plus teacher-forced exact match of the first tool call and of the full tool-call sequence. No generation pass. |
| Epochs | **1.** A second epoch is not in this run. |
| Recovery trajectories | **Not yet.** No bad-call-then-repair rows in the warm-up. |
| Task mix (GRPO prompts) | `ToolTrainConfig.pool_types` allowlist plus `type_mix_pool`: 600 rows, **boxed_warning / faers_top_reaction / approval / faers_count** (relative 0.30 / 0.20 / 0.15 / 0.05, renormalized). **Composite is out of the GRPO pool** (pass@8 95% all-fail; hard eval is compare + sponsor_hop). Hard-mix runs still take 60% `hard_train` (compare + sponsor_hop) and 40% that current-train allowlist, then floor `boxed_warning`. Not stage 3: its difficulty cut drops boxed_warning from 106 to 26 tasks. |
| Composite reward | Training (dense) reward only: 0.5 of correctness for each correct half of `"<count>; yes|no"` (`dense.composite_halves`). Both halves right but the strict string missed is capped at 0.85. Eval stays strict: "8714; no" for gold "8714; yes" scores strict 0.35, dense 0.70. |
| Sampling | 8 generations, **temperature 0.65** (`TOOL_TEMPERATURE`, `--temperature` on probe and `train-tools`). The rollout limit and GRPOConfig must agree, and the config refuses a mismatch. top_p stays the vLLM default 1.0. |
| Logging | Weights & Biases (`report_to=wandb`) records the TRL log: `frac_reward_zero_std` (tied groups), `rewards/parse_rate` (episodes with >=1 parsed tool call), `rewards/turn_parse_rate` (turns that parsed), `episodes/terminated_{submit,error,max_steps}`, `episodes/closed_{turn_limit,context}` (runner closed it), `episodes/tool_calls_mean`, `reward_std`, `rewards/env/mean` (strict), `rewards/dense/mean`, `completions/mean_length`, `loss_tokens/mean`, `grad_norm`. The run config stores prompt version, tool budget, hard_train_mix, seeds, adapter, and git SHA. |
| Gate | Same as Plan A: the starting adapter's strict tool eval `mean_reward` must beat `eval-tools-base` by **+0.03**, else the pipeline stops before GRPO (`gate-failed`). |
| Stop rule | `ParseRateGuard`: at step 20, if the mean of `rewards/parse_rate` or `rewards/turn_parse_rate` over steps 1-20 is **< 0.9**, training stops. The LoRA goes to `adapter-stopped/` (never `adapter/`), the reason to `stop.json`, and `ParseRateStop` marks the pipeline `failed-train-tools`. |
| Budget | **$20 cap** for the whole tool pipeline (estimate ~$7-18: three HF evals at $1.5-4.5 each, SFT ~$0.5-1, 150 GRPO steps ~$2-4). Watch Modal billing; stop the app if it nears $20. |

```
# From the track/tool-rollouts checkout (the image packs the directory modal_train.py sits in).
# Tool run starts from the BASE model: no upload_adapter.sh, no --init-adapter.
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET ~/modal-venv/bin/modal deploy modal_train.py
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET ~/modal-venv/bin/python scripts/launch_pipeline.py tools
# eval-tools-base -> sft-tools -> eval-tools-start (gate) -> train-tools -> eval-tools
```
Before a tool rerun, on CPU, check that gold applications are reachable by number:

```bash
uv run python scripts/check_approval_reachability.py
```

That writes `results/approval-reachability.json` (train and test approval tasks, plus an FG-0002 replay). No Modal and no GPU.

Watch `runs/pipeline-tools/status.json`, `pipeline.log`, and the W&B run `grpo-qwen3b-tools`.
Red flags: `rewards/parse_rate` < 0.9 (the stop rule fires at step 20); `frac_reward_zero_std` > 0.6 over the first 30 steps
(consider 16 generations next time); `episodes/closed_context` rising (search loops filling 8192); `kl` climbing steadily.
When done: `modal app stop fdagym-grpo-3b`, download the runs, delete the volume so spend returns to $0.

## Resume GRPO from the 2026-10-19 SFT adapter

The 2026-10-19 `pipeline_tools` run finished base eval, tool SFT, and the gated SFT eval, then died in GRPO before step 1 (`AttributeError` on `_get_per_token_logps`). Do not redo those three stages. The volume is fresh, so upload the artifacts below and launch the same pipeline. A stage whose output file is already there is skipped and not overwritten. A skipped eval is loaded back into `status["results"]`, which is what the gate reads. The strict scores that must stay usable for the final comparison are base **0.0086** and SFT eval **0.5552** (0.5552 >= 0.0086 + 0.03, so the gate passes).

The volume is mounted at `/vol`. `modal volume put` paths are relative to the volume root, without the `/vol` prefix.

| Stage | Skip when this exists | What it preserves |
|---|---|---|
| `eval-tools-base` | `/vol/runs/baseline-qwen3b-tools/summary.json` | strict mean_reward **0.0086** (`models["Qwen/Qwen2.5-3B-Instruct"]`) |
| `sft-tools` | `/vol/runs/sft-qwen3b-tools/adapter/adapter_model.safetensors` | warm-up LoRA, 1,106 rows |
| `eval-tools-start` | `/vol/runs/sft-qwen3b-tools-tools-eval/summary.json` | strict mean_reward **0.5552** (`models["Qwen/Qwen2.5-3B-Instruct+tools"]`) |
| `train-tools` | `/vol/runs/grpo-qwen3b-tools/adapter/adapter_model.safetensors` | absent, so GRPO runs |
| `eval-tools` | `/vol/runs/grpo-qwen3b-tools-eval/summary.json` | absent until GRPO finishes |

Trajectories, if you still have them, sit next to those directories: `/vol/runs/baseline-qwen3b-tools.jsonl` and `/vol/runs/sft-qwen3b-tools-tools-eval.jsonl`. The skip checks `summary.json` only. `compare_runs.py` reads `models.*.mean_reward` from `summary.json`.

Do **not** upload `runs/pipeline-tools/status.json` from the failed run. A fresh pipeline directory starts at `eval-tools-base` and skips forward. If that failed status is on the volume, the runner refuses to start until you pass `--force`.

GRPO itself starts from the SFT adapter when `grpo-qwen3b-tools/` has no `checkpoint-*` (`resolve_tool_start`). There is no checkpoint from the crashed run (it died before step 1). Do not pass `--init-adapter`: the default adapter path is the one in the table. Passing `--init-adapter /vol/runs/sft-qwen3b-tools/adapter` also works; it drops the `sft-tools` stage and points the gate at the same `sft-qwen3b-tools-tools-eval` summary.

```bash
# Run from the track/tool-rollouts checkout (the image packs the directory modal_train.py sits in).
# Artifacts are on the box, so use absolute paths; no Hugging Face download is needed.
cd /workspace/bio-rl-pr2
M="env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET $HOME/modal-venv/bin/modal"
A=/workspace/bio-rl-artifacts/2026-10-19-tools/runs
$M volume create fdagym-checkpoints
# 1. Base eval summary (strict 0.0086).
$M volume put fdagym-checkpoints $A/baseline-qwen3b-tools /runs/baseline-qwen3b-tools
# 2. SFT adapter from the local copy.
$M volume put fdagym-checkpoints $A/sft-qwen3b-tools/adapter /runs/sft-qwen3b-tools/adapter
$M volume put fdagym-checkpoints $A/sft-qwen3b-tools/sft.json /runs/sft-qwen3b-tools/sft.json
# 3. Gated SFT eval summary (strict 0.5552).
$M volume put fdagym-checkpoints $A/sft-qwen3b-tools-tools-eval /runs/sft-qwen3b-tools-tools-eval
# Not uploaded: runs/pipeline-tools/ (failed status.json) and the empty runs/grpo-qwen3b-tools/.
# 4. Redeploy (the old image still called _get_per_token_logps), then the same launcher. No --init-adapter, no --force.
$M deploy modal_train.py
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET $HOME/modal-venv/bin/python scripts/launch_pipeline.py tools
```

Confirm the three paths before launch (the adapter listing must show `adapter_model.safetensors`, 119,801,528 bytes):

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET ~/modal-venv/bin/modal volume ls fdagym-checkpoints /runs/baseline-qwen3b-tools
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET ~/modal-venv/bin/modal volume ls fdagym-checkpoints /runs/sft-qwen3b-tools/adapter
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET ~/modal-venv/bin/modal volume ls fdagym-checkpoints /runs/sft-qwen3b-tools-tools-eval
```

The next log line should be `skip-eval-tools-base`, then `skip-sft-tools`, then `skip-eval-tools-start`, then `train-tools`. GRPO loads `/vol/runs/sft-qwen3b-tools/adapter` on top of `Qwen/Qwen2.5-3B-Instruct`. It does not run base eval or SFT again.

## Public leaderboard split

`data/tasks/public.jsonl` is the published question sheet. Train and test are other files. SFT and GRPO still call `load_split("train")`. `in_training_pool` drops `split: public` and types `compare` and `sponsor_hop`. Do not append the public file to `train.jsonl`.

`search_approvals` accepts one of `generic_name`, `application_number`, or `sponsor_name`. A sponsor question can open an application, copy `sponsor_name`, and list the snapshot drugs that share it.

A tool-mode eval uses prompt **v4** and tool budget **8** (greedy temperature 0, seed 0, strict reward, `protocol: tool_rollout`). GPU is A100-80GB (`TOOL_GPU`). This does not start training. Pass `--prompt-version v3 --tool-budget 6` to reproduce an older protocol.

```bash
uv run python scripts/eval_chat.py --model <slug> --split public --run-id public-eval

HF_TOKEN=... uv run python scripts/eval_tools.py --eval --split public \
  --adapter /path/to/adapter --run-id public-eval
```

`--action eval-tools` with no `--split` and no `--adapter` still scores the TEST split with the volume GRPO adapter. The pipeline stages do not pass `split`, so they stay on test.

GPU time is a planning estimate from [tool-rollouts.md](tool-rollouts.md), not a measurement. That note prices a 324-task tool TEST eval at about 0.5–1.5 h on A100-80GB ($2.50/GPU-hour plus about $0.51/h RAM) and a full 6-tool episode at about 45 s. This split is 100 tasks and the reference solver uses 3 to 6 calls (max 6). Scaling the TEST range by 100/324 is about 9–28 minutes if episodes were the same length; stretching toward the 45 s figure is about 75 minutes per adapter. A middle estimate is **20–35 GPU-minutes per adapter, 50–80 GPU-minutes for both**, about $2.50–$4. Do not treat that as a measured step time.

Before spending GPU, on CPU:

```bash
uv run python scripts/audit_hard.py reachability --budget 6
uv run python scripts/audit_hard.py reachability --budget 8
```

Each run re-solves every hard question from the question text alone. The default budget is 8. Reports land in `results/hard-reachability-budget<N>.json`. The committed `results/hard-reachability.json` is the earlier budget-6 run.

Tool-mode probe, GRPO, and hard eval also run on a RunPod A100 with `scripts/run_gpu.py`. The commands, the output paths, and how to copy them off are in [train.md](train.md) under "Tool-mode jobs on a RunPod pod". Reward weights are not an argument there either.

## Hard RL: v4 SFT warm-up, hard eval, 3-step GRPO dry run

Not launched. These are deploy + spawn only, from the checkout whose directory contains `modal_train.py` (that directory is what the image packs). Do not use `modal run`. A laptop disconnect cannot cancel a spawn.

Reward weights are unchanged (`0.60/0.25/0.10/0.05`). On `compare` and `sponsor_hop`, when the answer itself scores 0, dense correctness is `0.5 * hop_rate` (`HOP_CAP` in `fdagym/rewards/hops.py`). Weighted by 0.60, that hop term is at most 0.30 of the total. Efficiency still uses the pre-hop answer. One-shot types get no hop credit. `hop_reward` in `fdagym/train/tool_score.py` returns that hop rate for logging. `episode_scores` does not add it again. The shaped return is the dense base plus the existing parse bonus.

A year hop reads the approval products in the tool output. The ingredient matches the question's generic when it is that name, or that name plus a salt, ester, or hydrate (`names_match` in `fdagym/drugnames.py`). Task generation uses the same function. `divalproex sodium` matches `divalproex`. `metformin and sitagliptin` does not match `metformin`, and `insulin glargine` does not match `insulin`. Strict eval does not call this matcher, so a trajectory's strict score is unchanged.

Unflagged commands stay on the old path. `train_tools`, `pipeline_tools`, and the local entrypoint default `hard_train_mix` to `0`, which does not mix. `python scripts/launch_pipeline.py tools` still spawns `pipeline_tools` with only `init_adapter` and `force`. SFT defaults `include_hard_train` to false, so the warm-up is the current train split, same as v3.

A new hard RL run passes `hard_train_mix=0.6` (60% `hard_train`, 40% current train, the 34-question holdout excluded), prompt `v4`, tool budget `8`. Before the first step it logs the holdout and hard-eval sizes, a sha256 of each id list, and the id lists themselves. It raises if any of those ids is in the training pool. The memorization callback is installed on that path and runs every 10 steps. A 3-step dry run therefore logs the exclusion proof and does not yet log a memorization report.

SFT hyperparameters are `SFTConfig`, the same as the v3 warm-up: base `Qwen/Qwen2.5-3B-Instruct`, lr `1e-4`, 1 epoch, batch 4, grad accumulation 2, LoRA from `TrainConfig` (r=16, alpha=32), keep reward `>= 0.97`. The CPU audit is `results/sft-v4-oracle.json` (prompt v4, budget 8, keep 0.97):

| pool | v3 kept | v4 kept | script failures | reward dropped |
|---|---|---|---|---|
| train fit (5% of drugs held out) | 1,106 | 1,106 | 0 | 0 |
| drug holdout inside train | 54 | 54 | 0 | 0 |

53 of those rows are longer than 8192 characters. The Qwen2.5-3B tokenizer measures a max of 2,967 tokens on them, so none miss the 8192-token cutoff. `include_hard_train` (off here) keeps all 319 hard-train rows on reward.

Example row `FG-0343` (approval, reward 1.0). The prompt baked into the row contains the v4 FAERS line and this budget sentence:

```
- You get at most 8 tool calls, then you must submit_answer.
```

The supervised tool call is `search_approvals` with `{"generic_name": "acoramidis"}`, then `submit_answer`. The full prompt is `example_row.prompt` in `results/sft-v4-oracle.json`.

```bash
# From the directory that contains modal_train.py.
M="env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET $HOME/modal-venv/bin"
$M/modal deploy modal_train.py

# 1. Warmup SFT from the base 3B. Train split only. Adapter lands at
#    /vol/runs/sft-qwen3b-tools-v4/adapter.
$M/python scripts/spawn_modal.py sft

# 2. Hard eval of that adapter. Greedy, v4, budget 8, split hard.
$M/python scripts/spawn_modal.py eval-hard

# 3. GRPO dry run, 3 steps, mix 0.6, from that adapter. Reward code is not an argument.
$M/python scripts/spawn_modal.py grpo-dry
```

`scripts/spawn_modal.py` calls `modal.Function.from_name("fdagym-grpo-3b", ...).spawn(...)` and exits. The kwargs it sends:

| command | function | kwargs |
|---|---|---|
| `sft` | `sft` | `run_id=sft-qwen3b-tools-v4`, `tool_budget=8`, `prompt_version=v4`, `include_hard_train=False` |
| `eval-hard` | `eval_tools` | `adapter=/vol/runs/sft-qwen3b-tools-v4/adapter`, `run_id=sft-qwen3b-tools-v4-hard`, `split=hard`, `tool_budget=8`, `prompt_version=v4` |
| `grpo-dry` | `train_tools` | `resume=False`, `run_id=grpo-qwen3b-tools-hard-dry`, `init_adapter=/vol/runs/sft-qwen3b-tools-v4/adapter`, `tool_budget=8`, `prompt_version=v4`, `hard_train_mix=0.6`, `max_steps=3`, `memorization_every=10` |

A full pipeline that uses the mix, instead of the 3-step check, is `python scripts/launch_pipeline.py tools --hard-train-mix 0.6` after the same deploy. Omit the flag to keep today's type mix.

## Hard GRPO signal (4 questions per step, no tied groups)

`grpo-qwen3b-tools-hard` (Qwen2.5-3B LoRA r=16, TRL 0.21, vLLM 0.8.5 colocate, one A100-80GB, prompt v4, tool budget 8, `hard_train_mix` 0.6, 150 steps, 1 question × 8 generations, `logprob_chunk_size` 256) passed its gate, but about half the steps had all 8 rewards tied (`frac_reward_zero_std` = 1), mostly easy current questions. Those steps have zero advantage. The hard-mix defaults below are what `train_tools` uses when `hard_train_mix` is above 0. An unflagged tool run (mix 0) stays at 1 question per optimizer step, with resampling and the filter off. Reward weights are unchanged.

| Knob | Hard-mix default | Off switch |
|---|---|---|
| `prompts_per_step` | **4** questions × 8 generations per optimizer step | `--prompts-per-step 1` |
| `dynamic_sampling` | on. Tied groups are dropped and replaced | `--no-dynamic-sampling` |
| `max_resample_attempts` | **3** extra draws per tied group | `--max-resample-attempts 0` masks without redrawing |
| `difficulty_filter` | on. Drops saturated current prompts, then floors boxed_warning. Composite is not in the pool | `--no-difficulty-filter` |

`grpo-dry` and `python scripts/launch_pipeline.py tools --hard-train-mix 0.6` pick these up with no extra flags. `None` on the Modal arguments means "use that default".

### Batch: gradient accumulation, micro-batch stays one group

TRL 0.21 sets `steps_per_generation` from `generation_batch_size / per_device_train_batch_size` and calls `_generate_and_score_completions` on that whole generation batch. The log-prob forward that fit after chunking is **8 sequences**. A 32-sequence forward would scale the checkpointed hidden states by four.

So the hard config keeps:

| Field | Value | Why |
|---|---|---|
| `num_generations` | 8 | one group |
| `per_device_train_batch_size` | 8 | the log-prob forward width |
| `generation_batch_size` | 8 | TRL `steps_per_generation` stays 1, so vLLM `max_num_seqs` stays 8 |
| `gradient_accumulation_steps` | 4 | four groups, one optimizer step |
| `logprob_chunk_size` | 256 | unchanged |

**Peak memory.** Same band as the last successful 8-generation step. The allocation that OOMed was the unchunked `[8, ~4k, 151936]` fp32 logit tensor, about 17 GiB. Chunks of 256 token rows stay about 156 MiB, and that chunk does not grow with the number of questions in the optimizer step because each micro-batch is still 8 sequences. Gradient accumulation keeps LoRA gradients, not four full forwards. vLLM still reserves `gpu_memory_utilization` 0.30 and schedules 8 sequences.

**Time and cost per optimizer step.** Four groups run one after another, so a step is about **4×** the last run's step before any resample. Each tied group adds another 8-episode rollout, up to 3 extras (`max_resample_attempts`). [docs/train.md](train.md) prices one group at about 25 s (a short episode) to 45 s (a full tool budget) and $0.000837/s, and those figures were not measured on a GPU. Scaling them:

| Schedule | Wall clock for 150 optimizer steps | Cost at $0.000837/s |
|---|---|---|
| Last run, 1 group, no resample | 1.0–1.9 h | about $3–$6 |
| 4 groups, no resample | 4.2–7.5 h | about $13–$23 |
| 4 groups, half of them redrawn once (6 rollouts) | 6.3–11.3 h | about $19–$34 |

The RUNBOOK note for the last hard run put 150 steps of 1×8 nearer **$2–4**. Four times that band is about **$8–16** before resampling. `max_steps` is still 150. Lower it on the launcher if the dollar cap is the constraint. Worst case is 4 × (1 + 3) = 16 group-rollouts in one optimizer step, when every group stays tied through the cap.

### Tied groups are not training examples

TRL 0.21 does not implement DAPO dynamic sampling (huggingface/trl#3758 is not in this pin). `ToolGRPOTrainer._sample_groups` does it inside the method the trainer already replaces:

1. Roll the dataloader's group (seed = `global_step`, same as before).
2. If the 8 shaped rewards are identical, drop that group and draw another prompt from the **training prompt list**, up to 3 times. A redraw uses seed `global_step + 10007 * round`.
3. A group that is still tied is masked (`example_mask` 0). Its policy term and its KL term are both left out of the loss, and it is left out of the mean, so it does not shrink the other group in that micro-batch. Across the four accumulation steps, Hugging Face still divides by 4, so a masked micro-batch contributes 0 and the other three are a little smaller than three solo steps. That only happens when the cap is hit.

The resample pool is the prompt rows for this run. It is checked against the published sheet, `hard_train_holdout`, and TEST. Those splits, and the memorization probes that read the holdout, are not filtered and are not drawn. `prove_training_pool` still runs on the unfiltered train and `hard_train` lists. A checkout without the private sheet treats that exclusion list as empty.

### The 60/40 mix after the filter

`base_pass_rate` is null on the frozen tasks. The filter therefore drops current prompts with structural `difficulty == "easy"`, except `composite` and `boxed_warning` (TEST strict score on the last hard run moved composite 0.993 → 0.810 and boxed_warning 0.886 → 0.799). A pass rate of 1.0 or 0.0, when one is present on the task or in `PromptFilter.pass_rates`, drops a non-protected prompt instead of the easy tag. Protected types stay even at 0 or 1; a tied rollout of one of them is still dropped online.

On seed 0 and 600 rows (`tests/train/test_group_sample.py::test_real_hard_prompts_keep_the_mix_and_the_regressed_types`):

| | count |
|---|---|
| Dropped before the mix | 432, all easy `approval` |
| Current pool kept | 728 (approval 44, boxed_warning 106, composite 69, faers_count 120, faers_top_reaction 389) |
| Hard-train rows in the 600 | 360 (compare 220, sponsor_hop 140) |
| Current rows in the 600 | 240 |
| After the floor (25% of the current slice each) | composite **60**, boxed_warning **60**, faers_top_reaction 83, faers_count 28, approval 9 |

The 60/40 counts do not move. The floor replaces other current rows; it does not replace `hard_train`. `label_open`, TEST, hard eval, and the holdout are not in this table.

### What is logged to W&B

Per optimizer step (the four questions are summed; fractions stay means):

| Key | Meaning |
|---|---|
| `groups/dropped` | Groups in the original batch whose 8 rewards matched. Those draws are not trained. |
| `groups/resampled` | Replacement groups rolled out. |
| `groups/resample_rounds` | Extra rollout rounds. |
| `groups/masked` | Groups still tied after the cap, excluded from the loss. |
| `groups/trained` | Groups that enter the loss. A full step is 4. |
| `groups/initial_frac_zero_std` | Share of the original groups that were tied. |
| `frac_reward_zero_std` | Share tied in the batch advantages are computed on, after resampling. |

The run config also stores `prompts_per_step`, `num_generations`, `gradient_accumulation_steps`, `per_device_train_batch_size`, `dynamic_sampling`, `max_resample_attempts`, `difficulty_filter`, and `post_filter_by_type` (rows, unique ids, and splits). The same mix, including `filter` and `floor`, is printed as `hard-rl post-filter mix ...` and written to `mix.json`.
