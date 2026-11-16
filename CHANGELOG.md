# Changelog

## Grader 6

Grader version 6. Data version 2026-11-16. `snapshot_version` stays `2026-10-15`.

Version 5 paid dense hop credit for running tools without a real submit, treated an id-only citation as a full correctness match, and scored `No, but X has a boxed warning` as a valid no. Two public earliest-boxed questions were golded on an unfilled amoxicillin/clavulanate boxed-warning template.

- `search_drug_label(..., has_section=boxed_warning)` skips unfilled `[Insert …]` templates and heading-only stubs. FG-9118 gold is buprenorphine; FG-9140 gold is rivaroxaban. Downstream private task sets should be regenerated.
- Dense hop credit requires a schema-valid, non-abstain submit. No submit, a malformed submit, or `NOT_FOUND` / `unknown` / `""` pays 0 correctness. `False` and `0` are real answers.
- Dense correctness backed only by an id-only citation (right id, bad quote or field) is capped at `ID_ONLY` (0.50).
- Boolean no accepts `None`, `None of them`, and `Neither`. Hedged or clause-positive answers (`No, but X has a boxed warning`, `No. X has a boxed warning`) fail.
- Committed OpenRouter traces in `results/public-haiku-luna-2026-11-05.jsonl` stay the grader-v5 historical record. They were not rewritten. Offline `scripts/rescore_runs.py` moves 1 of 186 rows: Haiku FG-9140 0.1625 → 0.8000 (answered rivaroxaban, now gold). Dense-only leaks are not in that jsonl.

## Dashboard CVE, M1–M6, onboarding

Next.js 15.5.29 (GHSA-9qr9-h5gf-34mp). Env docstring matches the tool budget; invalid JSON is `invalid_action`; `label_open` is rejected on `reset`; `step(..., latency_ms=)` records latency; snapshot manifest rows carry `sha256`; `eval_chat.py --split` lists committed files and refuses overwrite. CI runs ruff, pyright, pytest, the wheel smoke, and `web` lint/build. README covers grader v6, the anti-hacking table, and committed baselines. Training docs are the optional `[train]` extra in `docs/train.md`; onboarding does not require a remote GPU vendor or Weights & Biases.

## Installable wheel

`schemas/` now lives at `fdagym/schemas/` and is loaded with `importlib.resources`. Importing `fdagym` does not read `.env` or `data/` via `Path.parents`. `DiskSnapshot` takes `root=` or `FDAGYM_SNAPSHOT_DIR` (`python -m fdagym.openfda`). `parse_question` / `Unsolvable` live in `fdagym.questions` so `fdagym.rewards` no longer imports `fdagym.train`.

## Symmetric disjointness

`assert_disjoint` now expands both sheets: a drug that owns an application number cited on the other sheet is an overlap. `hard_train.jsonl` (334) and `hard_train_holdout.jsonl` (36) were rebuilt on the `2026-10-15` freeze so the check is 0/0/0/0 both ways against `public.jsonl`. GitHub Actions pins `astral-sh/setup-uv` at v10.3.3.

## hard_train decontamination

`hard_train.jsonl` (337) and `hard_train_holdout.jsonl` (36) were rebuilt so they share no drugs, application numbers, sponsors, or question text with `public.jsonl`. The previous files overlapped the published sheet on 51 drugs and one question (old `FG-8393` = `FG-9171`). `hard_train_holdout` remains an in-distribution ~10% question carve of the same pool, not a disjoint eval. `tests/tasks/test_public.py` now always checks the three files.

## Grader 5

Grader version 5. Data version stays 2026-10-28. `snapshot_version` stays `2026-10-15`.

Version 4 gated `sponsor_hop` correctness on a hop proof. Version 5 applies the same idea to `compare`. A right answer still needs a valid citation; it now also needs the comparison to be visible in the trace.

- A `compare` score pays correctness only when every named drug was queried so the unique/none decision is in the observations, and (except `boxed_none`) a valid citation is about the deciding drug. One FAERS call plus a four-character quote from `results` no longer opens the gate.
- `HopScore.proves()` is true for compare variants only when every hop flag hits. Dense training still shapes a partial hop into the correctness slot after that gate, the same way sponsor-hop does.
- `generate_hard()` has no default seed. It reads `FDAGYM_HARD_SEED` / `FDAGYM_HARD_ID_START` when those arguments are omitted.

## 2026-10-28 grader 4

Grader version 4. Data version stays 2026-10-28. `snapshot_version` stays `2026-10-15`.

Version 3 required every hop on every sponsor question, including the boxed-warning keyword. Version 4 leaves that keyword optional once the sponsor match has named one drug. The published questions are `data/tasks/public.jsonl`.

`snapshot_version` is the openFDA freeze a task's tools read. `DATA_VERSION` is the task sheet this grader was written against. It moves when questions or gold change and the freeze does not. `GRADER_VERSION` moves when `score()` changes what a trace is worth.

- A `sponsor_hop` score pays correctness only when the trace proves the join. Strict eval and dense training both use this gate. The published questions need every hop, including both boxed-warning searches. An older sponsor question needs the application, the sponsor match, and the FAERS step when the question has one. The boxed-warning keyword is not required once that match has left a single drug.
- Efficiency counts distinct `(tool, arguments)` pairs. Four copies of one search count as one call.
- A `sponsor_name` lookup counts toward the unnamed-sponsor hop only when that string is the sponsor on the opened application.
- The `search_approvals` description is shorter so a tool rollout still fits the 8192-token context.

## 2026-10-28

Grader version 2. Data version 2026-10-28.

- A correct trace that uses fewer calls than `min_tool_calls` no longer receives efficiency 1. Efficiency is `calls / min_tool_calls` when the trace is short, and `min_tool_calls / calls` when it is long.
- `search_approvals` accepts `sponsor_name` and lists the snapshot drugs that share that sponsor.
- Compare citations may name any drug in the question. A `boxed_none` question accepts an empty boxed-warning search. Those citation rules shipped earlier without a version. They are part of grader 2.

`scripts/rescore_runs.py` reprints a stored jsonl with the current `score()`.
