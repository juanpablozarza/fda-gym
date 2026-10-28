# FDA-Gym

A Gym-style RL environment where an LLM agent uses tools over a frozen openFDA snapshot to answer regulatory
questions, scored by automatically verifiable rewards. It includes baselines, a GRPO + LoRA fine-tune, and a Next.js dashboard.
The full plan is in [docs/PLAN.md](docs/PLAN.md) (in Spanish). The decisions below override it where they differ.

## Decisions (override PLAN.md)

- Name **FDA-Gym**, Python package `fdagym`.
- **3 data sources only:** `drug/label`, `drug/event` (FAERS), `drug/drugsfda`. No recalls, no NDC. Task type 4 (recalls) is dropped.
- Reward: `R = 0.60*correctness + 0.25*citation + 0.10*format + 0.05*efficiency`.
  - A citation is **valid** when its `id` appeared in a tool output of *this* trajectory, its `field` is one the task accepts
    (`gold_citations`), and its `quote` fuzzy-matches (≥ 0.9) that field's text. `citation` = valid / total cited, or 0 if none.
    Quotes shorter than 4 characters never count, and a field of 12 characters or fewer needs a full-string match.
  - `correctness` counts only if at least one citation is valid. Invalid format forces correctness to 0.
  - Boolean answers with a topic (boxed warnings) must match yes/no, hit a gold keyword, and name at most 2 distinct
    boxed-warning topics from the vocabulary derived from the snapshot's labels (blocks keyword stuffing).
  - `efficiency = min(1, min_tool_calls / max(calls, 1))`, counted only if correctness > 0.
  - Abstaining is correct only on `answerable: false` tasks, and only after a search (grounded empty-result citation).
  - Records with no id of their own are cited by the tool call that returned them: `id: "step:<n>"`. FAERS top reactions use
    field `results`, FAERS report counts and empty results use `meta.results.total`. Task `gold_citations` for these use
    `ids: []` and exactly those field names.
  - Absence of a label section ("No, no boxed warning") is cited from `search_drug_label` output: `id` = the label's
    `set_id`, field `sections`. Boolean tasks without a topic have no `keywords`; yes/no alone decides correctness.
- Difficulty is a structural rubric plus a measured base-model pass rate, scheduled as a curriculum: [docs/difficulty.md](docs/difficulty.md).
- Models through **OpenRouter** via LiteLLM (`OPENROUTER_API_KEY`). openFDA key in `OPENFDA_API_KEY`.
- Training: Plan A GRPO + LoRA on **Qwen2.5-3B-Instruct** (Modal L40S, `$30/mo` credit), then **Qwen2.5-7B-Instruct** on a RunPod A100 80GB only if the 3B adapter beats its own Plan A TEST baseline. Commands: [docs/train.md](docs/train.md). Do not start Modal or RunPod from an import, CI, or the training script.
- Dashboard follows design **B1** (dark glass, amber `#F5A524`): [docs/design/B1-overview.png](docs/design/B1-overview.png),
  [docs/design/B1-trajectory.png](docs/design/B1-trajectory.png). The HTML files next to them are the reference markup.

- Agent tools (names are part of the contract, since the dashboard and training prompts use them): `search_drug_label`,
  `get_label_section`, `query_adverse_events`, `search_approvals`, `submit_answer`.
- The published leaderboard uses `data/tasks/public.jsonl` (`split: public`, ids from `FG-9101`). Generate it with `scripts/generate_hard.py public` and check it with `scripts/audit_hard.py public`. `hard.jsonl`, `data/tasks/review.md`, and `results/hard-*` stay in this private repo. The public repo is a clean copy with no history from here.

## Tracks

| Track | Owns | Lesson |
|---|---|---|
| data | `fdagym/openfda/`, `scripts/snapshot.py`, `data/snapshots/`, `tests/data/` | `learn/01-data.md` |
| tasks | `fdagym/tasks/`, `data/tasks/`, `tests/tasks/` | `learn/02-tasks.md` |
| env | `fdagym/env.py`, `fdagym/tools.py`, `fdagym/rewards/`, `tests/env/`, `tests/rewards/` | `learn/03-env-rewards.md` |
| integration | `tests/integration/` | `learn/04-integration.md` |
| evals | `fdagym/agents/`, `scripts/run_baseline.py`, `results/`, `tests/agents/` | `learn/05-evals.md` |
| train | `fdagym/train/`, `scripts/grpo_singleturn.py`, `scripts/eval_local.py`, `scripts/run_gpu.py`, `modal_train.py`, `tests/train/` | `learn/07-training.md` |
| web | `web/` | `learn/06-dashboard.md` |
| planner | `AGENTS.md`, `schemas/`, `fdagym/snapshot.py`, `fdagym/queries.py`, `docs/`, `learn/00-*` | `learn/00-orientation.md` |

Everything reads openFDA data through the `Snapshot` protocol in [fdagym/snapshot.py](fdagym/snapshot.py), and every query is built
by [fdagym/queries.py](fdagym/queries.py). Tools take constrained arguments (drug, year, serious) and never raw search strings, so
the snapshot can answer every query an agent can make. Joining search clauses with a literal `+AND+` breaks under URL encoding (500); use spaces. Add dependencies with `uv add`;
the planner resolves `pyproject.toml` and `uv.lock` conflicts at merge. Copy `.env.example` to `.env` in the repo root. It is gitignored. `fdagym` loads it on import.

Baseline evals write `results/<run_id>.jsonl` (one trajectory per line, `trajectory.schema.json`) and `results/<run_id>/summary.json` (per-model mean reward, pass rate, citation validity, mean tool calls, cost_usd). The dashboard reads those files from `web/public/results/` (copy or symlink). Models go through OpenRouter via LiteLLM; do not call provider APIs directly.

Training is Plan A: single-turn GRPO. Prefetch each task's `oracle_calls` through `FDAEnv` so citation ids exist in `env.steps`, put 1–2 distractor documents in the **prompt only**, and let the model emit JSON for `submit_answer`. Same `score()`. `label_open` is skipped. LoRA r=16 α=32, 8 generations, 400 steps, ~3k prompt packing, 256 completion tokens, `use_vllm=True`, checkpoints every 50 steps. Default model `Qwen/Qwen2.5-3B-Instruct`. Torch lives behind `uv sync --extra train`. The untrained vs trained comparison is `scripts/eval_local.py` on the TEST split (not the OpenRouter multi-turn table).

Run metrics go to Weights & Biases (project `bio-rl`, run id = `run_id`, `resume='allow'`). The Modal secret is named `wandb` and holds `WANDB_API_KEY`. `scripts/wandb_chart.py` draws the PNG from the W&B API. Eval deliveries stay `results/<run_id>.jsonl` plus `summary.json`, `leaderboard.json`, and `tasks.json`. The dashboard still plots a `training-curve.jsonl` if one is dropped in `web/public/results/`; training no longer writes that file.

## Working agreement

- One track per agent, one branch and worktree per track. Stay inside your track's directories. JP merges.
- `schemas/` is the contract between tracks. Don't change it without saying so explicitly in your final report.
- Gate: `tests/integration/test_oracle.py` plays every task's `oracle_calls` through the real env and must score 1.0 on
  every gradable task. Any change to tasks, tools, or rewards keeps it green.
- Python 3.11+, `uv`, `pytest`, `ruff`. Tests run offline against recorded fixtures, never the live API.
- Known openFDA gotchas (evidence in `learn/00-orientation.md` §7): filters on list fields aren't correlated per element,
  one drug has many labels (`set_id`s; metformin has 365), and MedDRA terms are uppercase with British spelling. Many Drugs@FDA
  records lack the `openfda` block, so approvals are searched by `products.active_ingredients.name`. FAERS harmonizes
  generic names only for mapped drugs (finerenone: 0 by generic name, 2,960 by active substance), so FAERS queries OR both
  fields. An empty FAERS result is only a real NOT_FOUND if that union is empty. A generic-name search also
  matches combination products (e.g. metformin + sitagliptin), so filter those locally where a task needs the single-ingredient drug.

## Teaching protocol (required)

JP is learning from this build. Every track ships a lesson at `learn/<NN>-<track>.md` that follows
[learn/TEMPLATE.md](learn/TEMPLATE.md) exactly. Read the template and the example [learn/00-orientation.md](learn/00-orientation.md) before writing yours.
Quiz blocks must parse with `python3 scripts/quiz.py <lesson>`. Check that before finishing.

At the checkpoint, stop and report back. When resumed, read `learn/answers/<lesson-stem>.jsonl` and respond to every answer
self-graded `p` or `n`.
