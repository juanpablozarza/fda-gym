# fda-gym

Public [FDA-Gym](https://github.com/juanpablozarza/fda-gym) eval for Prime Intellect's Environments Hub. An agent answers regulatory questions with five tools over a frozen openFDA snapshot, then `submit_answer`. The grader scores the answer and the trace.

The published sheet is 93 tasks, `FG-9101`–`FG-9193` (50 `compare` and 43 `sponsor_hop`). This Hub package is `0.2.x`. Grader v6 is `fdagym.rewards.score()`. Data version `2026-11-16`. Freeze `2026-10-15` covers `drug/label`, `drug/event`, and `drug/drugsfda`. The prompt tells the model it may make 8 tool calls besides `submit_answer`. Ids outside the public range are refused.

| Hub package | Grader | Data version | Snapshot freeze |
| --- | --- | --- | --- |
| 0.1.x | 5 | 2026-10-28 | 2026-10-15 |
| 0.2.x | 6 | 2026-11-16 | 2026-10-15 |

Source: [juanpablozarza/fda-gym](https://github.com/juanpablozarza/fda-gym). See [verifiers Discussion #3071](https://github.com/PrimeIntellect-ai/verifiers/discussions/3071).

## Install

[Hub listing](https://app.primeintellect.ai/dashboard/environments/jpzarza/fda-gym).

```bash
prime env install jpzarza/fda-gym
uv run vf-eval fda-gym --dry-run --no-push
uv run vf-eval fda-gym --select.include.ids FG-9101
```

The environment wheel includes `fda_gym/data/tasks/public.jsonl` and `fda_gym/data/snapshots/2026-10-15`. A git checkout is not required. Set `FDAGYM_SNAPSHOT_DIR` or `FDAGYM_PUBLIC_JSONL` to override those files.

## From a clone

```bash
uv sync --extra prime
uv run vf-eval fdagym --dry-run --no-push
uv run vf-eval @ configs/prime.toml --dry-run --no-push
```

A clone uses the repo's `data/` tree.
