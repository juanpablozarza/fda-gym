# fda-gym

Public [FDA-Gym](https://github.com/juanpablozarza/fda-gym) eval for Prime Intellect's Environments Hub. Verifiers **v1** (`FDAGymTaskset`). Not a v0 `load_environment` package.

This README is the Hub listing text. A public listing appears only after `prime env push` from this directory. This package does not claim to be listed until that push succeeds.

## What it is

- **93** published tasks, ids `FG-9101`–`FG-9193` (50 `compare` + 43 `sponsor_hop`) in `data/tasks/public.jsonl`.
- Grader **v5**: `FDAGymTaskset` grades through `fdagym.rewards.score()`. This package does not fork the grader or the tools.
- Freeze **2026-10-15** (`drug/label`, `drug/event`, `drug/drugsfda`). The Hub wheel ships that freeze and `public.jsonl`. Ids outside the public range are refused.
- Soft tool budget advertised to the model: **8** non-submit calls.

Source: [juanpablozarza/fda-gym](https://github.com/juanpablozarza/fda-gym). Tracking: [verifiers Discussion #3071](https://github.com/PrimeIntellect-ai/verifiers/discussions/3071).

## Install

After the environment is pushed to the Hub (public, not `--visibility=PRIVATE`):

```bash
prime env install juanpablozarza/fda-gym
uv run vf-eval fda-gym --dry-run --no-push
uv run vf-eval fda-gym --select.include.ids FG-9101
```

`prime env install` is enough for the freeze. This environment wheel includes `fda_gym/data/snapshots/2026-10-15` and `fda_gym/data/tasks/public.jsonl`. You do not need a git checkout of FDA-Gym for those files. `FDAGYM_SNAPSHOT_DIR` and `FDAGYM_PUBLIC_JSONL` still override discovery.

From a clone of this repo, without a Hub listing:

```bash
uv sync --extra prime
uv run vf-eval fdagym --dry-run --no-push
uv run vf-eval @ configs/prime.toml --dry-run --no-push
```

That path uses the checkout's `data/` tree. `configs/prime.toml` stays `--dry-run --no-push` for CI (`push = false`).

To publish after merge: from this directory, `prime login` then `prime env push` (do not pass `--visibility=PRIVATE`).

## Layout

Hub package: `README.md`, `pyproject.toml`, `fda_gym/` (exports `FDAGymTaskset`). Depends on `verifiers==0.4.0` and `fdagym @ git+https://github.com/juanpablozarza/fda-gym.git`. The library wheel for `fdagym` still does not ship the freeze; this environment wheel does.
