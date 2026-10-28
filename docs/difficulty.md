# Task difficulty and curriculum

Status: **approved by JP (Oct 15), pending his review of the generated sample sheet.** The task-generator track implements it, and the training track consumes it.

## Why two layers

GRPO learns only from prompts where the model sometimes succeeds (see `learn/00-orientation.md` §3.3).
What matters is the *model's* pass rate, not how hard a human thinks a task is. But you can't review a pass rate,
and it changes as the model trains. So every task carries both:

1. **Structural difficulty:** derived deterministically from features a human can check by reading the task.
2. **Empirical pass rate:** the base model's success rate over 8 samples, measured once before training.

The structural tier is what we discuss and audit. The pass rate is what the curriculum actually schedules on.

## Layer 1: structural features

Each task records these in `difficulty_features` (see `schemas/task.schema.json`):

| Feature | Meaning |
|---|---|
| `sources` | Distinct endpoints needed (1 to 3) |
| `min_tool_calls` | Calls an oracle needs, excluding `submit_answer`. Also used by the efficiency reward |
| `needs_disambiguation` | More than one candidate record (many `set_id`s, several NDAs) and the agent must pick or accept a set |
| `needs_aggregation` | The answer is a count, top-k, or earliest/latest across records |
| `needs_normalization` | The answer must be mapped to a controlled vocabulary (MedDRA spelling, brand to generic) |
| `has_constraint` | Extra filter such as a date window, seriousness, or submission type |

Tier rule (first match wins):

- **hard:** `sources ≥ 2`, or `min_tool_calls ≥ 4`, or (`needs_aggregation` and `has_constraint`)
- **easy:** `sources == 1`, `min_tool_calls ≤ 2`, no disambiguation, no aggregation
- **medium:** everything else

The tier is computed from the features and never typed by hand, so changing the rule re-tiers every task consistently.

## Examples by task type

| Type | Example | Features | Tier |
|---|---|---|---|
| `approval` | "What year was Coumadin's original NDA approved?" | 1 source, 1 call, earliest ORIG approval in one record | easy* |
| `approval` | "What year was OZEMPIC's original NDA approved?" | Brand spans two NDAs (NDA209637, NDA213051), so the agent must pick the earlier | medium |
| `boxed_warning` | "Does warfarin's label carry a boxed warning, and about what risk?" | 1 source, 2 calls, disambiguation (74 labels) | medium |
| `faers_top_reaction` | "Most-reported reaction in FAERS for metformin?" | 1 source, 1 call, aggregation, normalization | medium |
| `faers_count` | "How many serious FAERS reports for drug X were received in 2025?" | 1 source, 1 call, aggregation + constraint | hard |
| `label_open` | "Summarize isotretinoin contraindications for a patient planning pregnancy." (LLM judge, ~10% of tasks) | 1 source, 2 calls, disambiguation | medium |
| `composite` | "Serious reports in 2025 for X, and does its label carry a boxed warning?" | 2 sources, 3 calls | hard |

\* "Earliest" over a single record's submission list does not count as aggregation (decided below).

## Layer 2: calibration

Before training, run the base model (Qwen2.5-3B-Instruct) 8 times per training task and store `base_pass_rate`.
A sample counts as a pass when correctness > 0.

- `base_pass_rate == 0`: dead today (no gradient). Keep it for later stages, since it may become learnable.
- `base_pass_rate == 1`: already solved (no gradient). Drop it from training and keep it as a regression check.
- Disagreements are review signals: an "easy" task at 0/8 is probably a broken gold answer or prompt, and a "hard" task at 8/8 probably has a leak.

## Curriculum

| Stage | Mix easy / medium / hard | Pool filter | Advance when |
|---|---|---|---|
| 1 | 60 / 30 / 10 | `0 < pass rate < 1` | Mean train reward over the last 50 steps ≥ 0.6, or 40% of the step budget |
| 2 | 30 / 50 / 20 | Re-measure pass rates on the remaining pool, same filter | Same rule, or 75% of the budget |
| 3 | 20 / 40 / 40 | Everything not solved | End of budget |

**When a tier runs short** (the first 100 tasks were 10 easy, 38 medium, 52 hard), the stage fills the gap from the
next tier up, taking the tasks with the highest `base_pass_rate` first. Before calibration, it takes them in task order. The
mix is a target, not a hard constraint, and the training log records the mix actually used for each stage.

Easy templates exist so stage 1 has a real pool. Each one is one source, at most 2 calls, with no disambiguation and no aggregation:
- "Does <drug>'s label carry a boxed warning?" as yes/no, only when every label for the drug agrees
- "Who is the sponsor of application <NDA/BLA number>?"
- "What brand name is marketed under application <number>?"
- "What is the dosage form (or route) of the product under application <number>?"

The test split is never filtered or staged. It keeps its natural mix so before/after numbers stay comparable.

## Human review

The task generator writes `data/tasks/review.md`: 10 random tasks per tier and type, each with the question, gold answer,
gold citations, features, and computed tier. JP marks the bad ones, and any task type with more than 2 bad out of 10 gets fixed before training.

## Decided (JP, Oct 15)

1. Unanswerable tasks (`answerable: false`): about **5%** of tasks, in both train and test.
2. Single-record "earliest" does **not** count as aggregation, so `approval` tasks stay easy and stage 1 has a clean easy pool.
3. Stages advance on **mean reward ≥ 0.6 or the step-budget cap**, whichever comes first.
