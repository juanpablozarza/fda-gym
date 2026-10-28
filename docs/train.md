# Plan A training: 3B on Modal, then 7B on RunPod

Do **not** start these from CI or from an import. JP runs the commands below. Nothing here rents a GPU until you do.

Protocol is Plan A: prefetch `oracle_calls`, generate `submit_answer` JSON, same `score()`. The number that decides “did GRPO work” is **held-out TEST mean reward** of the untrained checkpoint vs the LoRA adapter, both scored by `scripts/eval_local.py` at **temperature 0**. That is not the OpenRouter multi-turn table.

Config (both sizes): LoRA r=16 α=32, **8 generations**, **400 steps**, docs packed to fit vLLM context, **256** completion tokens, lr `1e-5`, vLLM colocate (`trl==0.21.0` + `vllm==0.8.5` + `torch==2.6.0`), checkpoints every **50** steps, resume from `checkpoint-*`.

| Job | GPU | Time | Rough $ |
|---|---|---|---|
| 3B untrained TEST eval (greedy) | Modal L40S | ~20–40 min | ~$0.70–$1.30 at $1.95/hr |
| 3B SFT warm-up, 1 epoch (984 prompts) | Modal L40S | **~25–45 min** | ~$0.80–$1.50 |
| 3B SFT adapter TEST eval | Modal L40S | ~20–40 min | ~$0.70–$1.30 |
| 3B GRPO 150 steps from the SFT adapter | Modal L40S | **~45–70 min** | ~$1.50–$2.30 |
| 3B GRPO 400 steps | Modal L40S | **~1.5–2 h** | ~$2.90–$3.90 |
| 3B adapter TEST eval (greedy) | Modal L40S | ~20–40 min | ~$0.70–$1.30 |
| **3B path total** | | | **fits in Modal’s $30/mo** (~$5–$7) |
| 7B untrained TEST eval | RunPod A100 80GB | ~45–75 min | ~$1.20–$2.00 |
| 7B GRPO 400 steps | RunPod A100 80GB | **~3–4 h** | ~$5–$8 at ~$1.6–$2/hr |
| 7B adapter TEST eval | RunPod A100 80GB | ~45–75 min | ~$1.20–$2.00 |

If L40S OOMs on 3B+vLLM, rerun with `--gpu A100-80GB` (still Modal). Only start the 7B pod if the 3B adapter **beats** `baseline-qwen3b` by at least **0.03** mean reward.

---

## 0. Offline check (no GPU)

Needs `data/snapshots/2026-10-15/manifest.jsonl` on disk (this repo already has it). `modal run` and `scripts/runpod_setup.sh` refuse to start without it. `.env` is not copied into the Modal image.

```bash
test -f data/snapshots/2026-10-15/manifest.jsonl
uv run python scripts/grpo_singleturn.py --dry-run
uv run pytest -q tests/train
```

The dry run prints `prompt tokens measured=… vllm_max_model_len=…`. That window is the longest rendered prompt plus the completion, rounded up and capped at 8192. If `fits` is not true, stop. Launch also refuses if any prompt exceeds the cap.

Count the SFT rows without a GPU (`984` kept on this snapshot):

```bash
uv run python scripts/sft_plan_a.py --dry-build
```

Whether a hard-mix GRPO group actually mixes rewards is measured by `probe_dispersion` (below), on real K=8 samples. The old synthetic simulator is gone.

## 0.1 Warm start before another GRPO run

The fixed-prompt GRPO run scored about 0.10 on almost every completion (format only), so groups had no advantage. Do this before spending another 400 steps. GRPO's reward is the dense score (`TrainConfig.dense_reward`, default on). Eval jsonl `reward.total` stays the strict score, comparable to the 0.0185 baseline; `summary.json` field `dense_reward` is the training score.

Weights, unchanged: `0.60*correctness + 0.25*citation + 0.10*format + 0.05*efficiency`. Format stays 0.10. An id that matches gold but has a bad field or quote earns 0.50 of the citation component. Near-miss correctness (only where strict correctness is 0): punctuation 0.85, year extracted from a date 0.70, brand/generic synonym from the snapshot 0.60, non-year counts inside 10% relative error up to 0.50. Dollar figures below scale the measured 400-step L40S run in the table; they are not a new GPU measurement.

```bash
modal run modal_train.py --action sft-plan-a
modal run modal_train.py --action eval-adapter --adapter /vol/runs/sft-plan-a-qwen3b/adapter
modal run modal_train.py --action train --init-adapter /vol/runs/sft-plan-a-qwen3b/adapter --max-steps 150 --run-id grpo-qwen3b-sft150
# strict score of that 150-step adapter, same jsonl field as the 0.0185 baseline:
modal run modal_train.py --action eval-adapter --adapter /vol/runs/grpo-qwen3b-sft150/adapter
```

SFT writes `/vol/runs/sft-plan-a-qwen3b/adapter` (1 epoch, lr `2e-4`, batch 1, grad accum 8). `--run-id grpo-qwen3b-sft150` keeps this GRPO run off the old `grpo-qwen3b` checkpoints. Resume loads `checkpoint-*` inside the run directory and does not reload `--init-adapter` when a checkpoint is already there.

## 1. Untrained 3B baseline (Modal)

```bash
pip install modal
modal setup   # once
modal run modal_train.py --action eval-base
```

Writes `/vol/runs/baseline-qwen3b/` on Volume `fdagym-checkpoints` (jsonl is overwritten on each eval, not appended).

## 2. Train 3B GRPO (Modal L40S)

```bash
modal run modal_train.py --action train
# resume after a crash (default):
modal run modal_train.py --action train --resume
# if L40S is too small:
modal run modal_train.py --action train --gpu A100-80GB
```

Adapter: `/vol/runs/grpo-qwen3b/adapter`. Checkpoints: `/vol/runs/grpo-qwen3b/checkpoint-50`, `...-100`, …

## 3. Score the 3B adapter, pull the volume, compare

```bash
modal run modal_train.py --action eval-adapter

mkdir -p results
modal volume get fdagym-checkpoints /runs/baseline-qwen3b ./results/baseline-qwen3b
modal volume get fdagym-checkpoints /runs/baseline-qwen3b.jsonl ./results/baseline-qwen3b.jsonl
modal volume get fdagym-checkpoints /runs/grpo-qwen3b-eval ./results/grpo-qwen3b-eval
modal volume get fdagym-checkpoints /runs/grpo-qwen3b-eval.jsonl ./results/grpo-qwen3b-eval.jsonl

uv run python scripts/compare_runs.py results/baseline-qwen3b results/grpo-qwen3b-eval
```

Exit 0 only if trained mean reward is **≥ 0.03 above** baseline. A 0.001 bump is not a win. Optional stricter gate: re-run both evals with `--seeds 1,2,3` so compare can use a 95% CI on the seed means.

---

## 4. Tool-mode jobs on a RunPod pod

Same jobs as Modal `probe_dispersion`, `train_tools`, and `eval_tools`. The pod is 1× **A100 80GB**, Ubuntu, CUDA 12.x, and a persistent disk at `/workspace` (weights, snapshot, checkpoints). The pinned stack is the `[gpu]` extra in `pyproject.toml` (torch 2.6.0, transformers 4.55.0, trl 0.21.0, peft 0.15.2, datasets 3.5.1, vllm 0.8.5, accelerate 1.4.0). Modal's image installs that same list. There is no Dockerfile.

`WANDB_API_KEY` and `HF_TOKEN` come from the environment. The CLI does not take them as flags and does not print them. `HF_HOME` becomes `/workspace/hf-cache` when it is unset, because `--runs-dir` is `/workspace/runs`.

```bash
git clone https://github.com/juanpablozarza/fda-gym.git
cd fda-gym
bash scripts/runpod_setup.sh
# copy the warm-up adapter onto the disk the jobs read
mkdir -p /workspace/runs/sft-qwen3b-tools-v4-warmup-mh1
# scp or: modal volume get fdagym-checkpoints /runs/sft-qwen3b-tools-v4-warmup-mh1/adapter \
#   /workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter
```

The setup script installs the extra, checks CUDA, imports vLLM, and exits. It does not start a job. Modal's function timeouts (probe 600s, GRPO 6h, eval 3h) are not applied here; the process runs until it exits. A finished job writes `DONE` in the run directory. An exception writes `FAILED` and the process exits non-zero.

```bash
# 1. Probe. Same arguments as the probe_dispersion spawn. 40 ids, sha of the draw order.
uv run python scripts/run_gpu.py probe \
  --runs-dir /workspace/runs \
  --adapter /workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter \
  --task-ids-file data/probe/dispersion-warmup-mh1.txt \
  --task-ids-sha256 51d68d3770e85bb49990468012f96b0c68f9b7e7a7d22408c37f993c416edfcf \
  --run-id probe-dispersion-warmup-mh1 \
  --k 8 \
  --seed 0 \
  --hard-train-mix 0.6 \
  --tool-budget 8 \
  --prompt-version v4

# 2. GRPO from that adapter. 60 steps, checkpoints every 10.
uv run python scripts/run_gpu.py train-tools \
  --runs-dir /workspace/runs \
  --run-id grpo-qwen3b-tools-hard-mh1 \
  --init-adapter /workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter \
  --tool-budget 8 \
  --prompt-version v4 \
  --hard-train-mix 0.6 \
  --max-steps 60 \
  --memorization-every 10 \
  --save-steps 10

# 2b. Two-step dry run of the same config. Separate run id, no resume.
uv run python scripts/run_gpu.py train-tools \
  --runs-dir /workspace/runs \
  --no-resume \
  --run-id grpo-qwen3b-tools-hard-mh1-dry \
  --init-adapter /workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter \
  --tool-budget 8 \
  --prompt-version v4 \
  --hard-train-mix 0.6 \
  --max-steps 2 \
  --memorization-every 10 \
  --save-steps 10

# 3. Hard eval of the GRPO adapter. Greedy, split hard.
uv run python scripts/run_gpu.py eval-tools \
  --runs-dir /workspace/runs \
  --adapter /workspace/runs/grpo-qwen3b-tools-hard-mh1/adapter \
  --run-id grpo-qwen3b-tools-hard-mh1-hard \
  --split hard \
  --tool-budget 8 \
  --prompt-version v4
```

`--init-adapter` is a local directory or a Hugging Face repo id. Private Hub adapters need `HF_TOKEN` in the environment (the CLI does not take it as a flag). Probe and `train-tools` share `--temperature` (default **0.65**) and `--pool-types` (default current-train types **without** `composite`).

### 4.2 Next GRPO run (distill SFT adapter)

`--init-adapter` is a local directory or a Hub repo id you control. Sampling is T=0.65. The 600-row hard mix stays 60/40. `composite` is not in the current-train allowlist (`boxed_warning` stays). `max_steps` 0 is 150.

```bash
# HF_TOKEN must already be in the environment.
uv run python scripts/run_gpu.py train-tools \
  --runs-dir /workspace/runs \
  --no-resume \
  --run-id grpo-qwen3b-distill-sft500 \
  --init-adapter /workspace/runs/sft/adapter \
  --tool-budget 8 \
  --prompt-version v4 \
  --hard-train-mix 0.6 \
  --temperature 0.65 \
  --pool-types boxed_warning,faers_top_reaction,approval,faers_count \
  --max-steps 150 \
  --memorization-every 10 \
  --save-steps 50
```

A matching probe uses the same `--temperature` and `--pool-types`. The published 40-id draw in `data/probe/dispersion-warmup-mh1.txt` includes two composite ids (`FG-0943`, `FG-0928`) that the new pool rejects; draw a new 40 from the current mix, or drop those two and recompute the sha.

```bash
uv run python scripts/run_gpu.py eval-tools \
  --runs-dir /workspace/runs \
  --adapter /workspace/runs/grpo-qwen3b-distill-sft500/adapter \
  --run-id grpo-qwen3b-distill-sft500-hard \
  --split hard \
  --tool-budget 8 \
  --prompt-version v4
```

Outputs:

| Job | On the pod |
|---|---|
| Probe | `/workspace/runs/probe-dispersion-warmup-mh1/` (`rollouts.jsonl`, `probe.json`, `run.json`, `DONE`) |
| GRPO | `/workspace/runs/grpo-qwen3b-tools-hard-mh1/` (`checkpoint-*`, `adapter`, `mix.json`, `run.json`, `DONE`) |
| Hard eval | `/workspace/runs/grpo-qwen3b-tools-hard-mh1-hard.jsonl` and `/workspace/runs/grpo-qwen3b-tools-hard-mh1-hard/` (`summary.json`, `run.json`) |

`run.json` is the provider-neutral record for every shared tool job. It records the effective job config, source revision, executor, Python/package/CUDA/GPU versions, and each attempt's status and timestamps. Modal writes it under `/vol/runs/<run_id>/`; RunPod writes the same file under `/workspace/runs/<run_id>/`. The only provider-specific parts are the mounted run directory and the save callback (Modal commits its Volume; RunPod writes to disk).

To continue the same GRPO run on the other provider, copy the whole `<run_id>` directory, keep the same `run_id`, and use the same training options with resume enabled. A changed config is rejected on resume; use a new `run_id` for a different experiment, or pass `--no-resume` to intentionally restart with new options. Copy the adjacent eval jsonl too when preserving dashboard results.

A watcher can poll `DONE` or `FAILED` in the run directory. Copy them off with `scp` (the jsonl sits next to the run directory, not inside it):

```bash
scp -r root@<pod>:/workspace/runs/probe-dispersion-warmup-mh1 ./results/
scp -r root@<pod>:/workspace/runs/grpo-qwen3b-tools-hard-mh1 ./results/
scp -r root@<pod>:/workspace/runs/grpo-qwen3b-tools-hard-mh1-hard ./results/
scp root@<pod>:/workspace/runs/grpo-qwen3b-tools-hard-mh1-hard.jsonl ./results/
```

Plan A 7B (`scripts/grpo_singleturn.py --preset 7b`) is unchanged and still waits on the 3B Plan A gate. It is not what this pod setup launches.

### 4.1 Distill SFT learning curve (teacher trajectories → LoRA student)

Multi-turn SFT of **Qwen2.5-3B-Instruct** on **passing** teacher trajectories (Claude Sonnet / Haiku, or any `teacher_model` in the JSONL). Nested `--sizes` share one seed so `100 ⊂ 250 ⊂ 500 ⊂ 1000 ⊂ all`, stratified by `template`. Hyperparameters match the tool-mode warm-up SFT (`SFTConfig`: lr `1e-4`, 1 epoch, batch 4, grad accum 2, LoRA r=16 α=32, prompt v4, budget 8) unless you override them. Loss is assistant turns only (system / user / tool tokens masked). Same `QwenToolCodec` as `eval-tools`.

The generator that writes teacher JSONL is a **separate script** (not this job). Contract: [teacher-trajectories.md](teacher-trajectories.md). This job never calls a teacher API and never rescores `fdagym.rewards`.

CPU dry-run (no Qwen weights, char tokenizer, fixtures). Default sizes on the fixture are `3,5,all` because the file is small; the GPU job defaults to `100,250,500,1000,all` and caps at the pool size.

```bash
uv run python scripts/sft_distill.py --dry-run
uv run python scripts/run_gpu.py distill-sft --dry-run --sizes 3,5,all
```

On the A100, after `bash scripts/runpod_setup.sh` and after the teacher JSONL is on disk. Starts from the **base 3B** unless `--init-adapter` points at an existing LoRA (for example the mh1 warm-up). `--eval` chains each adapter into the existing `eval-tools` hard split. Copy the JSONL onto the pod; do not point this job at `data/tasks/hard.jsonl` / test / holdout.

```bash
# 1. Train nested adapters. Identical hparams across sizes. Writes
#    /workspace/runs/distill-sft-qwen3b-n100/adapter, ...-n250, ...-nall
#    plus manifest.json (data hash, question ids, per-template counts, teacher mix, git sha).
uv run python scripts/run_gpu.py distill-sft \
  --runs-dir /workspace/runs \
  --trajectories /workspace/data/distill/teacher_trajectories.jsonl \
  --sizes 100,250,500,1000,all \
  --seed 0 \
  --run-id distill-sft-qwen3b \
  --eval \
  --eval-split hard \
  --prompt-version v4 \
  --tool-budget 8

# optional: continue the mh1 warm-up instead of a fresh LoRA
#   --init-adapter /workspace/runs/sft-qwen3b-tools-v4-warmup-mh1/adapter

# 2. If you trained without --eval, score one adapter the usual way:
uv run python scripts/run_gpu.py eval-tools \
  --runs-dir /workspace/runs \
  --adapter /workspace/runs/distill-sft-qwen3b-n100/adapter \
  --run-id distill-sft-qwen3b-n100-hard \
  --split hard \
  --tool-budget 8 \
  --prompt-version v4

# 3. CSV + SVG of hard-eval score vs training size (overall and per variant).
uv run python scripts/distill_curve.py \
  --runs-dir /workspace/runs \
  --prefix distill-sft-qwen3b \
  --sizes 100,250,500,1000,all \
  --out /workspace/runs/distill-sft-qwen3b/curve
```

`--dry-run` does not load Qwen and does not need CUDA. A few-step GPU smoke uses the same command without `--dry-run` and with `--max-steps 2 --sizes 3 --run-id distill-sft-smoke` (still the 3B, not a tiny random model). `WANDB_API_KEY` logs each size as its own run id (`distill-sft-qwen3b-n100`, …) on project `bio-rl`.

Known setup-script issue: `uv sync --group dev` (and some unpinned images) pull `tokenizers==0.23.3` / `huggingface-hub==1.33` which do not match `transformers==4.55`. Distill tests and the CPU dry-run never import those. The A100 job should keep using `uv sync --extra gpu --frozen` from `scripts/runpod_setup.sh`. If that extra is broken on a pod, install `setuptools` and the `[gpu]` pins before training; do not train against the mismatched Hub stack.

---


## What “beats baseline” means

Same split (`test.jsonl`, skip `label_open`), same Plan A packing, **greedy decoding**. Compare `summary.json` mean reward. Pass if **Δ ≥ 0.03**. Pass rate is the secondary column.

A high train curve with a flat TEST score means the adapter memorized packed train docs. TEST drugs are unseen; that is the number that matters.

---

## Tool-call GRPO (A100-80GB)

This is a different protocol from Plan A. The prompt is the question and the tool schema. The model calls `FDAEnv` during the rollout. Do not compare `grpo-qwen3b-tools-eval` to `baseline-qwen3b`: that baseline saw packed documents. Nothing here rents a GPU until JP runs `modal run` or, on a pod that is already up, `scripts/run_gpu.py`. `scripts/grpo_tools.py` is the CPU dry run and exits without training. `scripts/eval_tools.py --eval` is the local-GPU eval path and still does not call Modal.

Run order:

1. CPU checks (no GPU).
2. Greedy TEST eval of the **raw base 3B** (`baseline-qwen3b-tools`, no adapter). This does not need SFT or GRPO.
3. SFT warm-up on oracle trajectories (`sft-qwen3b-tools`).
4. Tool GRPO from that adapter (`grpo-qwen3b-tools`, **150 steps**). A missing adapter raises. It does not fall back to the base 3B. If `checkpoint-*` already exists, the job resumes from it.
5. Greedy TEST eval of the tool adapter (`grpo-qwen3b-tools-eval`). Compare it to step 2. Both use `protocol: tool_rollout`.

```bash
uv run python scripts/tool_rollout_dry.py
uv run python scripts/sft_warmup.py --dry-run
uv run python scripts/grpo_tools.py --dry-run
uv run python scripts/eval_tools.py --dry-run

modal run modal_train.py --action eval-tools-base
modal run modal_train.py --action sft
modal run modal_train.py --action train-tools
modal run modal_train.py --action eval-tools
```

Tool actions force `A100-80GB` even if `--gpu` is passed. Plan A stays on the L40S. Timeouts: SFT 2 h, GRPO 6 h, each eval 3 h. Volume `fdagym-checkpoints`, 64 GiB RAM, `retries=0`. Adapters: `/vol/runs/sft-qwen3b-tools/adapter`, `/vol/runs/grpo-qwen3b-tools/adapter`. The base eval loads `Qwen/Qwen2.5-3B-Instruct` with no LoRA and writes `kind: baseline`.

Shape: LoRA r=16 α=32, **8 generations**, **150 steps**, batch 8, grad accumulation 1, gradient checkpointing, vLLM utilization 0.30, context **8192** (prompt 2048 + completion 6144), 256 new tokens per turn, 7 turns, `force_submit` off. Checkpoints at 50, 100, and 150. SFT is one epoch, lr `1e-4`, batch 4, grad accumulation 2, same LoRA and context. Parse bonus is **+0.05 once per episode** on the GRPO reward only. Plan A remains 400 steps.

A hard-mix run (`hard_train_mix` 0.6) keeps that 8-sequence forward and accumulates **4** questions per optimizer step. Tied groups are resampled (cap 3) and, if they stay tied, left out of the loss. The current slice drops easy prompts, **excludes `composite`**, and floors `boxed_warning`. Knobs, the post-filter mix, and the memory and time estimates are in [RUNBOOK.md](RUNBOOK.md) under "Hard GRPO signal". Reward weights are not an argument. `--temperature` (default 0.65) and `--pool-types` are the same flags on probe and `train-tools`.

### Reward-dispersion probe

Before that hard-mix run, `probe_dispersion` samples **8** completions of each chosen training question and writes the training reward. GRPO only learns from groups whose rewards differ. The sampler is `VLLMGenerator`, so each turn is `SamplingParams(n=1, temperature=<TOOL_TEMPERATURE, default 0.65>, max_tokens=min(256, room), logprobs=1)` (vLLM defaults: top_p 1.0, top_k -1, repetition_penalty 1.0). The pool is the 600-row pool `train_tools` builds at `hard_train_mix=0.6`, prompt v4, tool budget 8, without `composite`. Scoring is `episode_scores(dense=True, bonus=0.05)`. It does not change the reward.

Deploy, then spawn. `main` has no action for it. A100-80GB, 600s, 32 GiB RAM, `retries=0`, no W&B. An existing `/vol/runs/<run_id>/` is refused. `rollouts.jsonl` is committed every 10 groups.

```python
modal.Function.from_name("fdagym-grpo-3b", "probe_dispersion").spawn(
    adapter="/vol/runs/<sft-run>/adapter",
    task_ids=ids,
    task_ids_sha256=sha256,  # sha256 of "\n".join(ids), draw order, no trailing newline
    run_id="probe-dispersion-...",
    k=8,
    seed=0,
    hard_train_mix=0.6,
    tool_budget=8,
    prompt_version="v4",
    temperature=0.65,
    pool_types="boxed_warning,faers_top_reaction,approval,faers_count",
)
```

Cost per step is GPU $2.50/h plus 64 GiB RAM, about **$0.000837/s**. Not measured on a GPU; scaled from the 6–8 s single-turn step.

| Step | Time | Cost |
|---|---|---|
| Early step, the call does not parse | ~8 s | **~$0.007** |
| Planning step, two or three turns | ~25 s | **~$0.021** |
| All 6 tool calls | ~45 s | **~$0.038** |

The first GRPO run is **150 steps**. At 25 s that is about **1.0 h and $3.15** (150 × $0.021). The range is about **$1–$6** ($0.007 × 150 at an early parse-fail step, $0.038 × 150 if every episode uses all 6 calls). SFT one epoch is about **$1–2**. Each greedy tool eval covers about 324 TEST tasks and is about **0.5–1.5 h and $1.50–$4**, also not measured. The base eval and the adapter eval are two jobs.

```bash
mkdir -p results
modal volume get fdagym-checkpoints /runs/baseline-qwen3b-tools ./results/baseline-qwen3b-tools
modal volume get fdagym-checkpoints /runs/baseline-qwen3b-tools.jsonl ./results/baseline-qwen3b-tools.jsonl
modal volume get fdagym-checkpoints /runs/grpo-qwen3b-tools-eval ./results/grpo-qwen3b-tools-eval
modal volume get fdagym-checkpoints /runs/grpo-qwen3b-tools-eval.jsonl ./results/grpo-qwen3b-tools-eval.jsonl

uv run python scripts/compare_runs.py results/baseline-qwen3b-tools results/grpo-qwen3b-tools-eval
```

`compare_runs.py` uses the same 0.03 mean-reward gate as Plan A, on the **strict env total** (`models[].mean_reward`). Training metrics live in Weights & Biases, not in a curve file. `reward` is the shaped training reward: the dense total plus the 0.05 parse bonus. `rewards/env/mean` is the strict total and `rewards/dense/mean` is the dense total before the bonus. Draw the chart with `python scripts/wandb_chart.py grpo-qwen3b-tools`.

Tool GRPO passes `TrainConfig.dense_reward` explicitly. `tool_train_config()` sets it true, and `run_tool_grpo` forwards `config.dense_reward` into `score(..., dense=True, snapshot=snapshot)`. `FDAEnv` still writes the strict total on the trajectory. Both `eval-tools-base` and `eval-tools` store that strict total on the jsonl and add `summary["dense_reward"]` beside it. The 0.05 parse bonus stays on the training reward only.
