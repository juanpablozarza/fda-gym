# FDA-Gym

An RL environment where a model uses tools over a frozen [openFDA](https://open.fda.gov/) snapshot to answer regulatory questions. Answers are scored by an automatic grader.

The published leaderboard split is [`data/tasks/public.jsonl`](data/tasks/public.jsonl). Run it with:

```bash
uv sync
uv run python scripts/eval_chat.py --model <openrouter-slug> --split public --run-id public-eval
```

Copy `results/<run-id>.jsonl` into `web/public/results/` to show it on the dashboard. See [web/public/results/README.md](web/public/results/README.md).

## Data

Drug labels, FAERS reports, and Drugs@FDA records come from openFDA. The freeze in `data/snapshots/2026-10-15/` is a local copy of those public API responses so the tools and the grader run offline. openFDA data is United States government work. FDA names, application numbers, and label text in the snapshot and in the task files are from that source.

This repository does not include a private evaluation sheet. Training code still runs when that file is absent.

## Setup

Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
uv run pytest -q
```

`.env` is optional for the offline tests. `OPENFDA_API_KEY` is only needed to record a new snapshot. `OPENROUTER_API_KEY` is only needed to score a hosted model.
