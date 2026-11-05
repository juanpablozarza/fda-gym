# FDA-Gym

A model answers a question about a drug label, a FAERS report, or an FDA approval. It calls tools against a frozen [openFDA](https://open.fda.gov/) snapshot. The grader scores the submitted answer and the trace that produced it.

The published sheet has 93 questions, `FG-9101` through `FG-9193`, in [`data/tasks/public.jsonl`](data/tasks/public.jsonl).

## Run a model

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Put an OpenRouter key in `OPENROUTER_API_KEY`. Then score the published sheet.

```bash
uv run python scripts/eval_chat.py --model <openrouter-slug> --split public --run-id public-eval
```

`results/public-eval.jsonl` has one trajectory per line. The `reward` object on each line is the grade. `reward.correctness` above 0 is a pass.

## What a grade means

The total is `0.60 * correctness + 0.25 * citation + 0.10 * format + 0.05 * efficiency`. Each term is from 0 to 1.

Format is 1 when the answer matches the schema. Otherwise format is 0, and correctness is 0 with it.

A citation is valid when its id appeared in a tool result from this trajectory, the task accepts that field, and the quote matches the field text. Quotes shorter than 4 characters never count. A field of 12 characters or fewer needs the whole string. The citation score is valid citations divided by citations submitted. No citations scores 0.

Correctness is 1 only when format is 1, at least one citation is valid, and the answer matches the gold. Otherwise correctness is 0. The pass rate is the share of questions with correctness above 0.

Efficiency counts only when correctness is above 0. The same tool call with the same arguments counts once. A trace shorter than the task minimum scores calls divided by that minimum. A longer trace scores the minimum divided by the calls.

Every score includes `grader_version`. The current grader is version 5.

A `compare` answer keeps correctness only when the trace queried every named drug (so the unique or none decision is in the observations) and a valid citation is about the deciding drug. A `sponsor_hop` answer already needed that kind of proof. One tool call and a short quote from `results` is not enough.

## Tools

The model does not send a raw search string. Each call takes a drug name, a section, a year, an application number, or a sponsor name.

| Tool | What it returns |
| --- | --- |
| `search_drug_label` | Label summaries for one generic name. `has_section` keeps labels that contain that section. |
| `get_label_section` | One section of a label that `search_drug_label` already returned in this episode. |
| `query_adverse_events` | FAERS reports for one generic name. `top_reactions` lists MedDRA terms. `count` returns how many reports match. |
| `search_approvals` | Drugs@FDA records. Pass exactly one of `generic_name`, `application_number`, or `sponsor_name`. |
| `submit_answer` | Ends the episode. The answer is one string, number, or boolean, or `NOT_FOUND`, plus up to 8 citations. |

## See a run

```bash
cp results/public-eval.jsonl web/public/results/
cd web && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The overview reads the jsonl files in `web/public/results/`.

## Data

The tools read [`data/snapshots/2026-10-15/`](data/snapshots/2026-10-15/), a freeze of three openFDA endpoints: `drug/label`, `drug/event`, and `drug/drugsfda`. The snapshot is on disk, so scoring and the tests do not call the live API.

| File | n | Role |
| --- | --- | --- |
| [`data/tasks/public.jsonl`](data/tasks/public.jsonl) | 93 | Published eval (`FG-9101`–`FG-9193`). Leaderboard sheet. |
| [`data/tasks/hard_train.jsonl`](data/tasks/hard_train.jsonl) | 337 | Hard-style training pool (`FG-8101`+). |
| [`data/tasks/hard_train_holdout.jsonl`](data/tasks/hard_train_holdout.jsonl) | 36 | In-distribution question holdout (~10% of the generated pool, stratified by variant). Same drugs as `hard_train`. Never mixed into training. Used to probe memorization. |
| [`data/tasks/train.jsonl`](data/tasks/train.jsonl) / [`test.jsonl`](data/tasks/test.jsonl) | 1229 / 342 | Original type mix (boxed warning, FAERS, approvals). Unchanged by the hard-train rebuild. |

`hard_train` and `hard_train_holdout` share no drugs, application numbers, sponsors, or question text with `public.jsonl`. They also share none of those with `test.jsonl`. The holdout is allowed to share drugs with `hard_train`; that is how a fact-swap can test memorization of known entities without leaking the published eval.

openFDA records are United States government work. Names, application numbers, and label text in the snapshot and in the task files come from that source.

`OPENFDA_API_KEY` is only for recording a new snapshot. The tests do not need a key.

## Tests

```bash
uv run pytest -q
```

MIT license. See [LICENSE](LICENSE).
