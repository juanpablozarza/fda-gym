# Teacher trajectories for distill SFT

This is the JSONL contract between the OpenRouter teacher runner ([`scripts/generate_teacher_trajectories.py`](../scripts/generate_teacher_trajectories.py)) and the student (`scripts/run_gpu.py distill-sft`). One object per line. Schema: [schemas/teacher_trajectory.schema.json](../schemas/teacher_trajectory.schema.json). Example: [schemas/examples/teacher_trajectory.example.json](../schemas/examples/teacher_trajectory.example.json). Fixture used by tests: [tests/train/fixtures/teacher_trajectories.jsonl](../tests/train/fixtures/teacher_trajectories.jsonl).

Do not write ChatML (`<|im_start|>`, `<|im_end|>`) into `messages[].content`. Distill SFT re-renders with `QwenToolCodec` (prompt **v4**, budget **8**), the same text `FDAEnv` eval uses. A stored system prompt is provenance only.

## Required fields

| Field | Meaning |
|---|---|
| `id` | Unique attempt id (`{teacher}/{question_id}/a{attempt_index}` is fine). |
| `question_id` | Question id in the distill pool. Not limited to `FG-NNNN`. |
| `template` | Task type / template. Nested subsets are **stratified by this field**. |
| `variant` | Template variant, or `""` if none. |
| `teacher_model` | Teacher id, e.g. `openrouter/anthropic/claude-sonnet-4.5`. |
| `attempt_index` | 0-based retry on this question by this teacher. |
| `messages` | `system` / `user` / `assistant` / `tool` turns. |
| `reward` | Strict components from `fdagym.rewards.score` (same object as eval jsonl). |
| `passed` | `true` iff `reward.correctness > 0`. Same rule as baseline `pass_rate`. |

Optional: `prompt_version` (`v3` or `v4`), `tool_budget`. Extra keys are allowed.

## Messages

1. Optional `system` (ignored at train time; the student uses [rollout.py](../fdagym/train/rollout.py) `system_prompt("v4", 8)` plus the tool catalog).
2. `user`: the question text only.
3. Then one or more pairs:
   - `assistant`: a Qwen Hermes `<tool_call>` block **or** a JSON object `{"tool": "...", "arguments": {...}}`. `submit_answer` is an assistant turn, not a tool turn.
   - `tool`: the observation JSON the env returned, with keys `tool`, `output`, `error`, `tool_calls_remaining`. Set `name` to the tool.

Assistant example (matches `QwenToolCodec.action_text`):

```
<tool_call>
{"name": "search_drug_label", "arguments": {"generic_name": "warfarin", "has_section": "boxed_warning"}}
</tool_call>
```

Tool example (matches `observation_payload` in `rollout.py`):

```json
{"tool": "search_drug_label", "output": {"total": 1, "results": []}, "error": null, "tool_calls_remaining": 7}
```

## Scoring

The generator calls JP's `score()` on the finished episode and copies `reward` plus `passed = reward["correctness"] > 0`. Distill SFT **does not rescore** and **does not import** `fdagym.rewards` to change it. It only reads `passed`.

## Generate teacher episodes

The runner calls LiteLLM through OpenRouter, uses the task's frozen snapshot, and validates each row against the schema before appending it. It stores an assistant action followed by the observation for each tool call; the final `submit_answer` is an assistant action with no extra observation. It resumes by `(teacher_model, question_id, attempt_index)`, so rerunning the same output fills gaps without duplicating completed attempts.

```bash
# Small paid pilot on the training split, capped at $2. The default teachers are Claude Sonnet 5.5 and Haiku 5.5.
uv run python scripts/generate_teacher_trajectories.py \
  --run-id teachers-pilot \
  --split train \
  --limit 5 \
  --cost-cap 2

# Continue the same file after interruption; --tasks can point to any single-version JSONL pool.
uv run python scripts/generate_teacher_trajectories.py --run-id teachers-pilot --split train
uv run python scripts/generate_teacher_trajectories.py --run-id teachers-train --split train
```

The API key comes from `OPENROUTER_API_KEY` (the project `.env` is loaded on import). Change teachers with `--models model-a,model-b`; `--attempts` sets retries per teacher and task. The default output is `results/distill/<run-id>.jsonl`. Keep the run id stable to resume; use a new id to start a separate dataset. The cost cap is checked before each episode, so one final episode can take the run slightly over the cap.

## What the student keeps

Passing rows only (`passed === true`). Optional filters: teacher, template, max attempts per question. Dedupe key: `(question_id, teacher_model, attempt_index)`. Nested `--sizes` are stratified by `template` and prefer one attempt per `question_id` before repeats.
