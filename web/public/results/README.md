# Results handoff

The dashboard is offline-static. It reads JSON from this folder at request time (no API).

The published leaderboard split is `public` (`data/tasks/public.jsonl`). Score it with `scripts/eval_chat.py --split public`. The private hard sheet is not in the public tree.

After an eval run:

```bash
cp results/<run_id>.jsonl web/public/results/
cp -R results/<run_id> web/public/results/
```

Expected layout:

```
web/public/results/
  <run_id>.jsonl          # one trajectory per line (schemas/trajectory.schema.json)
  <run_id>/summary.json   # per-model mean reward, pass rate, citation validity, mean tool calls, cost_usd
  leaderboard.json        # optional; Overview derives rows from jsonl if missing
  tasks.json              # optional catalog (id, question, type, sources, difficulty) for cards and replay titles
```

`fixture-run.jsonl` is synthetic review data (fictional drugs). Replace or keep it; the footer caption stays on while `example: true` is set on the leaderboard or a summary.
