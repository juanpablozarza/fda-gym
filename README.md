# FDA-Gym

A **tool-use RL environment** over a frozen [openFDA](https://open.fda.gov/) snapshot. An agent answers a regulatory question by calling tools, then `submit_answer`. The grader scores the answer **and** the trace, so a memorized fact with a fake citation does not pay.

FDA-Gym exists so RL researchers can train and evaluate tool-using models on a real, frozen government dataset without live API calls, and so a recruiter can see in two minutes that the grader actually blocks the shortcuts that usually inflate leaderboard scores.

The published sheet is 93 questions, `FG-9101`–`FG-9193`, in [`data/tasks/public.jsonl`](data/tasks/public.jsonl). Freeze: `2026-10-15` (`drug/label`, `drug/event`, `drug/drugsfda`).

![FDA-Gym viewer (design B1)](docs/design/B1-overview.png)

## Quickstart (offline, no GPU, no API keys)

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run python scripts/demo_public_episode.py
uv run pytest -q
```

That is the smoke run. It uses the committed snapshot and never calls live openFDA or an LLM.

The demo replays one published task (`FG-9101`, a 3-call FAERS compare) through `FDAEnv` on freeze `2026-10-15`. After every hop it prints the tool, a truncated observation, and the grader breakdown (`correctness` / `citation` / `format` / `efficiency` / `total` / `grader_version`). Partial scores before `submit_answer` stay at 0; the honest oracle should end at total 1.0. Override the row with `--task-id FG-9101`–`FG-9193`. No GPU, no `OPENROUTER_API_KEY`. This is one episode, not a 93-row model eval.

Score a Chat Completions model on the published sheet (needs `OPENROUTER_API_KEY` in `.env`):

```bash
cp .env.example .env
uv run python scripts/eval_chat.py --model openrouter/qwen/qwen-2.5-7b-instruct --split public --run-id public-eval
```

`--split` is any `*.jsonl` stem in `data/tasks/` (`public`, `hard_train`, `hard_train_holdout`, `train`, `test`). Existing `results/<run-id>` is refused unless you pass `--overwrite`.

Import the installed wheel without a git checkout:

```bash
uv build --wheel
uv venv /tmp/fdagym-wheel && uv pip install --python /tmp/fdagym-wheel/bin/python dist/*.whl
cd /tmp && /tmp/fdagym-wheel/bin/python -c "import fdagym, fdagym.env, fdagym.rewards"
```

The wheel does not ship the 87 MB snapshot. Point at a clone with `FDAGYM_SNAPSHOT_DIR=/path/to/fda-gym/data/snapshots`, or run `python -m fdagym.openfda`. Training extras are optional: `uv sync --extra train` (CPU) or `--extra gpu`. CPU dry-run: `uv run python scripts/grpo_singleturn.py --dry-run`. See [docs/train.md](docs/train.md). Pytest, the dashboard, and this README do not need a GPU vendor or Weights & Biases.

## Why a grader this strict

A model that already knows the answer can still fail if it does not retrieve the evidence. A model that guesses one of three drugs can still fail if the comparison is not in the trace. That is the point.

Total: `0.60 * correctness + 0.25 * citation + 0.10 * format + 0.05 * efficiency`. Every score includes `grader_version`. **Current grader: v5.**

| Term | Pays when |
| --- | --- |
| format | `submit_answer` matches the schema. Invalid format forces correctness to 0. |
| citation | cited `id` appeared in this trajectory, `field` is in `gold_citations`, quote fuzzy-matches ≥ 0.9. Quotes < 4 characters never count. |
| correctness | format is 1, at least one citation is valid, the answer matches gold, **and** the hop / compare proof is in the trace. |
| efficiency | only when correctness > 0. Distinct `(tool, arguments)` pairs. Repeated identical calls are free. |

`compare` pays correctness only when every named drug was queried (so unique/none is visible) and a valid citation is about the deciding drug. `sponsor_hop` already needed that kind of proof. One tool call and a short quote from `results` is not enough.

### Shortcuts the v5 suite catches

From `tests/rewards/test_public_attacks.py` and `tests/rewards/test_compare_proof.py` (grader v5, committed `2026-10-15` freeze):

| Attack | Score |
| --- | --- |
| Honest oracle (every `oracle_call`, gold answer, gold citations) | **93 / 93** pass |
| One-call compare guess (first `oracle_call` + generic quote + gold or first-named drug) | **0 / 50** |
| One-call sponsor-hop (same attack on `sponsor_hop`) | **0 / 43** |
| Quitter (`unknown`, no citations) | **0 / 93** |
| FG-9101: one FAERS call on abacavir, quote `"PAIN"`, answer `"pravastatin"` | correctness **0** (citation can still be 1) |

## Committed baselines

Numbers below are from files already in `results/`, labeled with the grader they were scored under. They are **not** invented. The OpenRouter smoke was re-scored offline with grader v5; totals did not move.

**OpenRouter smoke** (`results/baseline-2026-10-16.jsonl`, split `test`, 19 tasks, 1 seed). Stored rewards had no `grader_version` field; offline v5 re-score matches the stored totals.

| Model | n | mean reward (v5) | pass | mean citation |
| --- | --- | --- | --- | --- |
| `openrouter/nvidia/nemotron-3.5-lightning:free` | 19 | 0.1000 | 0.00 | 0.00 |
| `openrouter/cohere/north-mini-code:free` | 19 | 0.1000 | 0.00 | 0.00 |
| `openrouter/liquid/lfm-2.5-2.6b:free` | 19 | 0.0947 | 0.00 | 0.00 |

Requested paid slugs (`gpt-4o-mini`, `claude-3.5-haiku`, `qwen-2.5-7b-instruct`) were not run: the account had no credits. See `results/baseline-2026-10-16/summary.json`.

**Plan A local TEST** (`web/public/results/plan-a.json`, split `test`, 324 questions, greedy). These are the committed Qwen2.5-3B numbers, not a public-sheet run.

| Model | mean reward | pass | citation |
| --- | --- | --- | --- |
| Qwen2.5-3B-Instruct base | 0.0185 | — | — |
| SFT warm-up | 0.575 | — | — |
| GRPO 150 steps from warm-up | 0.839 | 80.6% | 0.87 |

There is no committed OpenRouter run on `public.jsonl` in this tree.

## Tools

The model does not send a raw search string. Each call takes a drug name, a section, a year, an application number, or a sponsor name.

| Tool | What it returns |
| --- | --- |
| `search_drug_label` | Label summaries for one generic name. `has_section` keeps labels that contain that section. |
| `get_label_section` | One section of a label that `search_drug_label` already returned in this episode. |
| `query_adverse_events` | FAERS reports for one generic name. `top_reactions` lists MedDRA terms. `count` returns how many reports match. |
| `search_approvals` | Drugs@FDA records. Pass exactly one of `generic_name`, `application_number`, or `sponsor_name`. |
| `submit_answer` | Ends the episode. The answer is one string, number, or boolean, or `NOT_FOUND`, plus up to 8 citations. |

`FDAEnv` is a tool-use environment, not a Gymnasium env. `reset(task_id)` returns `{question, tools}`. `step(action, latency_ms=...)` returns `(observation, reward, done, info)`. Reward stays 0.0 until the episode ends. Tool calls other than `submit_answer` are capped at `tool_budget` (default 8); `submit_answer` is still accepted after that. Invalid JSON is logged as `invalid_action`.

## Splits

| File | n | Role |
| --- | --- | --- |
| [`data/tasks/public.jsonl`](data/tasks/public.jsonl) | 93 | Published eval (`FG-9101`–`FG-9193`). Leaderboard sheet. |
| [`data/tasks/hard_train.jsonl`](data/tasks/hard_train.jsonl) | 334 | Hard-style training pool (`FG-8101`+). |
| [`data/tasks/hard_train_holdout.jsonl`](data/tasks/hard_train_holdout.jsonl) | 36 | In-distribution question holdout (~10% of the generated pool). Same drugs as `hard_train`. Never mixed into training. |
| [`data/tasks/train.jsonl`](data/tasks/train.jsonl) / [`test.jsonl`](data/tasks/test.jsonl) | 1229 / 342 | Original type mix (boxed warning, FAERS, approvals). |

`hard_train` and `hard_train_holdout` share no drugs, application numbers, sponsors, or question text with `public.jsonl` (symmetric check: a training drug that owns an eval application number is an overlap). They also share none of those with `test.jsonl`. The holdout may share drugs with `hard_train`.

openFDA records are United States government work. See [openFDA terms of service](https://open.fda.gov/terms/). Names, application numbers, and label text in the snapshot and in the task files come from that source. Code is MIT.

`OPENFDA_API_KEY` is only for recording a new snapshot. The tests do not need a key. Verify stored bodies with `python -m fdagym.openfda verify`.

## Dashboard

```bash
cp results/public-eval.jsonl web/public/results/
cd web && npm ci && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The overview reads the jsonl files in `web/public/results/`. Design reference: [`docs/design/B1-overview.png`](docs/design/B1-overview.png).

## Verifiers / Environments Hub

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 wrapper is in this repo: install `fdagym` with the optional `prime` extra, expose the five tools above, and grade with `fdagym.rewards.score` (grader v5). Default split: [`data/tasks/public.jsonl`](data/tasks/public.jsonl) (`FG-9101`–`FG-9193`). Ids outside that public range are refused. The snapshot is still the committed `2026-10-15` freeze via `FDAGYM_SNAPSHOT_DIR` (the wheel does not ship it). Tracking: [PrimeIntellect-ai/verifiers Discussion #3071](https://github.com/PrimeIntellect-ai/verifiers/discussions/3071). This is the local wrapper only — it is not an Environments Hub listing.

```bash
uv sync --extra prime
uv run vf-eval fdagym --dry-run --no-push
uv run vf-eval @ configs/prime.toml --dry-run --no-push
```

The optional extra pins `verifiers==0.4.0` so the v1 CLI is `vf-eval` (`uv run vf-eval @ configs/prime.toml`). Point `--select.include.ids` at a public id such as `FG-9101` for a one-task run; do not treat that as a 93-row model baseline. `--extra prime` cannot be combined with `--extra gpu`. The env contract remains `FDAEnv.reset` / `FDAEnv.step` above.

## HUD

Optional [HUD](https://docs.hud.ai) SDK Environment around the same `FDAEnv` / five tools / `score()` path, over `data/tasks/public.jsonl` only (`FG-9101`–`FG-9193`). No GPU. Default `uv sync --group dev` does not install the SDK. Do not combine with `--extra gpu` (solve `--extra hud` on its own, or with `--extra train`).

```bash
uv sync --extra hud
uv run hud eval tasks.py claude
uv run hud eval tasks.py claude --all
```

`hud eval` spawns the env locally (no Docker). The first command runs one public task; `--all` runs FG-9101–FG-9193. Use `claude`, another catalog model, or `--model …`. That run needs a provider key or `HUD_API_KEY` for the *model*; tests do not. Cap agent steps with `--max-steps` if you override the per-task default (`tool_budget` 8 plus `submit_answer`). Snapshot discovery is the same as the keyless demo: `FDAGYM_SNAPSHOT_DIR` or a checkout's `data/snapshots` freeze `2026-10-15`.

`Dockerfile.hud` is optional and deploy-only (`hud deploy` / `hud serve` in an image). Local `hud eval` does not use it. The wheel still does not ship the 87 MB snapshot.

## Tests and CI

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
cd web && npm ci && npm run lint && npm run build
```

MIT license. See [LICENSE](LICENSE).
