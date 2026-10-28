# FDA-Gym dashboard

Next.js 15 App Router UI for replaying eval trajectories. Dark glass, signal amber `#F5A524`, matching [B1](../docs/design/B1-overview.html).

Offline-static: no backend. Trajectories are JSON files in `public/results/`.

## Run

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

| Route | What it is |
|---|---|
| `/` | Overview: KPIs, leaderboard, training-curve placeholder, task cards |
| `/trajectories/[...id]` | Step replay (←/→, autoplay). Env ids are `run/FG-0001/s0`; fixture ids stay one segment. |
| `/runs` `/tasks` `/training` `/docs` | Same shell; training is empty until a GRPO dump lands |

## Fixture

Until evals write real `results/<run_id>.jsonl`, the app ships three synthetic trajectories in `public/results/fixture-run.jsonl` (fictional drugs: zorvetinib, velmorastat, pemradine). Footer on every page: *Example data · fictional drugs, models and records*.

```bash
npm run check-fixture
```

## Handoff from evals

```bash
cp results/<run_id>.jsonl web/public/results/
cp -R results/<run_id> web/public/results/
```

Details: [public/results/README.md](public/results/README.md).
