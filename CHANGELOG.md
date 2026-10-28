# Changelog

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
