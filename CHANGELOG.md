# Changelog

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
