# Training (optional `[train]` extra)

The README, pytest, and the dashboard do not need this file. Training is
`uv sync --extra train` (CPU dry-run) or `uv sync --extra gpu` (CUDA). Do not
start a GPU job from an import, CI, or a training import. Weights & Biases is
optional: no `WANDB_API_KEY` means logging is off and training continues.

Artifacts land in `results/<run-id>/` (jsonl, `summary.json`, adapter,
`run.json`). Pass `--runs-dir results` (or another relative directory on this
machine). `run.json` records the job config, source revision, executor, and
each attempt. Copy the whole `<run-id>` directory to resume on another
machine; a changed config is rejected on resume.

## Offline check (no GPU)

Needs `data/snapshots/2026-10-15/manifest.jsonl` on disk (this repo already
has it).

```bash
uv sync --extra train --group dev
test -f data/snapshots/2026-10-15/manifest.jsonl
uv run python scripts/grpo_singleturn.py --dry-run
uv run python scripts/sft_plan_a.py --dry-build
uv run pytest -q tests/train
```

The dry run prints `prompt tokens measured=… vllm_max_model_len=…`. That
window is the longest rendered prompt plus the completion, rounded up and
capped at 8192. If `fits` is not true, stop. Launch also refuses if any prompt
exceeds the cap.

Count the SFT rows without a GPU (`984` kept on this snapshot):

```bash
uv run python scripts/sft_plan_a.py --dry-build
```

## Protocol

Plan A: prefetch `oracle_calls` through `FDAEnv` so citation ids exist in
`env.steps`, put 1–2 distractor documents in the **prompt only**, and let the
model emit JSON for `submit_answer`. Same `score()`. `label_open` is skipped.

Config (3B and 7B): LoRA r=16 α=32, **8 generations**, **400 steps**, docs
packed to fit vLLM context, **256** completion tokens, lr `1e-5`, vLLM
colocate (`trl==0.21.0` + `vllm==0.8.5` + `torch==2.6.0`), checkpoints every
**50** steps, resume from `checkpoint-*`. Default model
`Qwen/Qwen2.5-3B-Instruct`. Start 7B (`--preset 7b`) only if the 3B adapter
beats its own Plan A TEST baseline.

The number that decides “did GRPO work” is **held-out TEST mean reward** of
the untrained checkpoint vs the LoRA adapter, both scored by
`scripts/eval_local.py` at **temperature 0**. That is not the OpenRouter
multi-turn table.

Weights, unchanged: `0.60*correctness + 0.25*citation + 0.10*format + 0.05*efficiency`.
GRPO's reward is the dense score (`TrainConfig.dense_reward`, default on).
Eval jsonl `reward.total` stays the strict score. `summary.json` field
`dense_reward` is the training score.

| Job | Time | Rough $ |
|---|---|---|
| 3B untrained TEST eval (greedy) | ~20–40 min | ~$0.70–$1.30 |
| 3B SFT warm-up, 1 epoch (984 prompts) | **~25–45 min** | ~$0.80–$1.50 |
| 3B SFT adapter TEST eval | ~20–40 min | ~$0.70–$1.30 |
| 3B GRPO 150 steps from the SFT adapter | **~45–70 min** | ~$1.50–$2.30 |
| 3B GRPO 400 steps | **~1.5–2 h** | ~$2.90–$3.90 |
| 3B adapter TEST eval (greedy) | ~20–40 min | ~$0.70–$1.30 |
| 7B untrained TEST eval | ~45–75 min | ~$1.20–$2.00 |
| 7B GRPO 400 steps | **~3–4 h** | ~$5–$8 |
| 7B adapter TEST eval | ~45–75 min | ~$1.20–$2.00 |

Dollar figures scale a measured 400-step 3B run; they are not a new GPU
measurement.

## What “beats baseline” means

Same split (`test.jsonl`, skip `label_open`), same Plan A packing, **greedy
decoding**. Compare `summary.json` mean reward. Pass if **Δ ≥ 0.03**. Pass
rate is the secondary column.

A high train curve with a flat TEST score means the adapter memorized packed
train docs. TEST drugs are unseen; that is the number that matters.

## Local GPU jobs

`--init-adapter` is a local directory or a Hugging Face repo id. Private Hub
adapters need `HF_TOKEN` in the environment (the CLI does not take it as a
flag). `WANDB_API_KEY` is the same: env only, never printed. A finished job
writes `DONE` in the run directory. An exception writes `FAILED` and the
process exits non-zero.

```bash
# Plan A untrained baseline, then GRPO, then adapter eval.
uv run python scripts/eval_local.py --split test --run-id baseline-qwen3b
uv run python scripts/grpo_singleturn.py --run-id grpo-qwen3b
uv run python scripts/eval_local.py --split test --adapter results/grpo-qwen3b/adapter --run-id grpo-qwen3b-eval

# Warm start: one SFT epoch, then 150 GRPO steps from that adapter.
uv run python scripts/sft_plan_a.py
uv run python scripts/grpo_singleturn.py --init-adapter results/sft-plan-a-qwen3b/adapter --max-steps 150 --run-id grpo-qwen3b-sft150
uv run python scripts/eval_local.py --split test --adapter results/grpo-qwen3b-sft150/adapter --run-id grpo-qwen3b-sft150-eval
```

SFT writes `results/sft-plan-a-qwen3b/adapter` (1 epoch, lr `2e-4`, batch 1,
grad accum 8). Resume loads `checkpoint-*` inside the run directory and does
not reload `--init-adapter` when a checkpoint is already there.

```bash
mkdir -p results
uv run python scripts/compare_runs.py results/baseline-qwen3b results/grpo-qwen3b-eval
```

Exit 0 only if trained mean reward is **≥ 0.03 above** baseline. A 0.001 bump
is not a win. Optional stricter gate: re-run both evals with `--seeds 1,2,3`
so compare can use a 95% CI on the seed means.

The pinned GPU stack is the `[gpu]` extra in `pyproject.toml` (torch 2.6.0,
transformers 4.55.0, trl 0.21.0, peft 0.15.2, datasets 3.5.1, vllm 0.8.5,
accelerate 1.4.0). Install it with `uv sync --extra gpu --frozen`. `uv sync
--group dev` can pull a mismatched `tokenizers` / `huggingface-hub` pair;
do not train against that stack.

## Tool-call GRPO

This is a different protocol from Plan A. The prompt is the question and the
tool schema. The model calls `FDAEnv` during the rollout. Do not compare
`grpo-qwen3b-tools-eval` to `baseline-qwen3b`: that baseline saw packed
documents. `scripts/grpo_tools.py` is the CPU dry run and exits without
training.

Run order:

1. CPU checks (no GPU).
2. Greedy TEST eval of the **raw base 3B** (`eval-tools-base`, no adapter).
3. SFT warm-up on oracle trajectories (`sft`).
4. Tool GRPO from that adapter (`train-tools`, **150 steps**). A missing
   adapter raises. It does not fall back to the base 3B. If `checkpoint-*`
   already exists, the job resumes from it.
5. Greedy TEST eval of the tool adapter (`eval-tools`). Compare it to step 2.
   Both use `protocol: tool_rollout`.

```bash
uv run python scripts/tool_rollout_dry.py
uv run python scripts/sft_warmup.py --dry-run
uv run python scripts/grpo_tools.py --dry-run
uv run python scripts/eval_tools.py --dry-run
```

GPU jobs on a machine that already has CUDA:

```bash
uv run python scripts/run_gpu.py eval-tools --runs-dir results --run-id eval-tools-base --split test
uv run python scripts/sft_warmup.py --train
uv run python scripts/run_gpu.py train-tools --runs-dir results --run-id grpo-qwen3b-tools --max-steps 150
uv run python scripts/run_gpu.py eval-tools --runs-dir results --adapter results/grpo-qwen3b-tools/adapter --run-id grpo-qwen3b-tools-eval
```

Shape: LoRA r=16 α=32, **8 generations**, **150 steps**, batch 8, grad
accumulation 1, gradient checkpointing, vLLM utilization 0.30, context
**8192** (prompt 2048 + completion 6144), 256 new tokens per turn, 7 turns,
`force_submit` off. Checkpoints at 50, 100, and 150. SFT is one epoch, lr
`1e-4`, batch 4, grad accumulation 2, same LoRA and context. Parse bonus is
**+0.05 once per episode** on the GRPO reward only. Plan A remains 400 steps.

A hard-mix run (`hard_train_mix` 0.6) keeps that 8-sequence forward and
accumulates **4** questions per optimizer step. Tied groups are resampled
(cap 3) and, if they stay tied, left out of the loss. The current slice drops
easy prompts, **excludes `composite`**, and floors `boxed_warning`. Reward
weights are not an argument. `--temperature` (default 0.65) and `--pool-types`
are the same flags on probe and `train-tools`.

### Reward-dispersion probe

Before that hard-mix run, `probe` samples **8** completions of each chosen
training question and writes the training reward. GRPO only learns from
groups whose rewards differ. Scoring is `episode_scores(dense=True, bonus=0.05)`.
It does not change the reward. An existing `results/<run_id>/` is refused.

```bash
uv run python scripts/run_gpu.py probe \
  --runs-dir results \
  --adapter results/sft-qwen3b-tools/adapter \
  --task-ids-file data/probe/dispersion-warmup-mh1.txt \
  --run-id probe-dispersion \
  --k 8 \
  --seed 0 \
  --hard-train-mix 0.6 \
  --tool-budget 8 \
  --prompt-version v4
```

Cost per step is about **$0.000837/s**. Not measured on a GPU; scaled from
the 6–8 s single-turn step.

| Step | Time | Cost |
|---|---|---|
| Early step, the call does not parse | ~8 s | **~$0.007** |
| Planning step, two or three turns | ~25 s | **~$0.021** |
| All 6 tool calls | ~45 s | **~$0.038** |

The first GRPO run is **150 steps**. At 25 s that is about **1.0 h and $3.15**
(150 × $0.021). The range is about **$1–$6** ($0.007 × 150 at an early
parse-fail step, $0.038 × 150 if every episode uses all 6 calls). SFT one
epoch is about **$1–2**. Each greedy tool eval covers about 324 TEST tasks
and is about **0.5–1.5 h and $1.50–$4**, also not measured. The base eval and
the adapter eval are two jobs.

```bash
mkdir -p results
uv run python scripts/compare_runs.py results/eval-tools-base results/grpo-qwen3b-tools-eval
```

`compare_runs.py` uses the same 0.03 mean-reward gate as Plan A, on the
**strict env total** (`models[].mean_reward`). Training metrics, when
`WANDB_API_KEY` is set, go to Weights & Biases (project `fda-gym`, run id =
`run_id`, `resume='allow'`). Draw the chart with
`python scripts/wandb_chart.py grpo-qwen3b-tools`. Override the project with
`WANDB_PROJECT` / `--wandb-project`.

Tool GRPO passes `TrainConfig.dense_reward` explicitly. `tool_train_config()`
sets it true, and `run_tool_grpo` forwards `config.dense_reward` into
`score(..., dense=True, snapshot=snapshot)`. `FDAEnv` still writes the strict
total on the trajectory. Both `eval-tools-base` and `eval-tools` store that
strict total on the jsonl and add `summary["dense_reward"]` beside it. The
0.05 parse bonus stays on the training reward only.

### Hard-mix knobs

When `hard_train_mix` is above 0:

| Knob | Hard-mix default | Off switch |
|---|---|---|
| `prompts_per_step` | **4** questions × 8 generations per optimizer step | `--prompts-per-step 1` |
| `dynamic_sampling` | on. Tied groups are dropped and replaced | `--no-dynamic-sampling` |
| `max_resample_attempts` | **3** extra draws per tied group | `--max-resample-attempts 0` masks without redrawing |
| `difficulty_filter` | on. Drops saturated current prompts, then floors boxed_warning. Composite is not in the pool | `--no-difficulty-filter` |

The 8-sequence log-prob forward stays one group. Gradient accumulation 4
makes four groups one optimizer step. Peak memory stays in the same band as
an 8-generation step.

On seed 0 and 600 rows, after the filter:

| | count |
|---|---|
| Dropped before the mix | 432, all easy `approval` |
| Hard-train rows in the 600 | 360 (compare 215, sponsor_hop 145) |
| Current rows in the 600 | 240 |
| After the boxed_warning floor | boxed_warning **60**, faers_top_reaction 129, faers_count 40, approval 11 |

The 60/40 counts do not move. The floor replaces other current rows; it does
not replace `hard_train`. `label_open`, TEST, `public`, and the holdout are
not in this table.

## Distill SFT

Multi-turn SFT of **Qwen2.5-3B-Instruct** on **passing** teacher trajectories.
Nested `--sizes` share one seed so `100 ⊂ 250 ⊂ 500 ⊂ 1000 ⊂ all`, stratified
by `template`. Hyperparameters match the tool-mode warm-up SFT unless you
override them. Loss is assistant turns only. Contract:
[teacher-trajectories.md](teacher-trajectories.md). This job never calls a
teacher API and never rescores `fdagym.rewards`.

```bash
uv run python scripts/sft_distill.py --dry-run
uv run python scripts/run_gpu.py distill-sft --dry-run --sizes 3,5,all --runs-dir results
```

GPU job (starts from the **base 3B** unless `--init-adapter` points at an
existing LoRA). `--eval` chains each adapter into `eval-tools` on `public`.
Do not point this job at `test` or `hard_train_holdout`.

```bash
uv run python scripts/run_gpu.py distill-sft \
  --runs-dir results \
  --trajectories data/distill/teacher_trajectories.jsonl \
  --sizes 100,250,500,1000,all \
  --seed 0 \
  --run-id distill-sft-qwen3b \
  --eval \
  --eval-split public \
  --prompt-version v4 \
  --tool-budget 8

uv run python scripts/run_gpu.py eval-tools \
  --runs-dir results \
  --adapter results/distill-sft-qwen3b-n100/adapter \
  --run-id distill-sft-qwen3b-n100-public \
  --split public \
  --tool-budget 8 \
  --prompt-version v4

uv run python scripts/distill_curve.py \
  --runs-dir results \
  --prefix distill-sft-qwen3b \
  --sizes 100,250,500,1000,all \
  --out results/distill-sft-qwen3b/curve
```

`--dry-run` does not load Qwen and does not need CUDA. A few-step GPU smoke
uses the same command without `--dry-run` and with
`--max-steps 2 --sizes 3 --run-id distill-sft-smoke`.
