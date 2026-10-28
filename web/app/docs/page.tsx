import { Caption } from "@/components/shell";
import { loadCatalog } from "@/lib/results";

export default function DocsPage() {
  const { example } = loadCatalog();
  return (
    <>
      <header className="top">
        <div>
          <h1>Docs</h1>
          <div className="muted">Offline static dashboard. No backend.</div>
        </div>
      </header>
      <section className="card glass prose">
        <h2>Handoff from evals</h2>
        <p>
          Baseline evals write <code>results/&lt;run_id&gt;.jsonl</code> (one trajectory per line, matching{" "}
          <code>schemas/trajectory.schema.json</code>) and <code>results/&lt;run_id&gt;/summary.json</code>. Copy those
          into this app:
        </p>
        <pre className="json mono">{`cp results/<run_id>.jsonl web/public/results/
cp -R results/<run_id> web/public/results/`}</pre>
        <p>
          Optional: drop <code>leaderboard.json</code> and a <code>tasks.json</code> catalog next to them. If
          leaderboard.json is missing, Overview derives rows from the jsonl. Refresh — <code>next dev</code> reads the
          folder on each request.
        </p>
        <h2>What the replay is for</h2>
        <p>
          A mean reward of 0.48 can hide a memorized citation that never appeared in the tool output. The replay shows
          the steps, the quotes, and whether each citation is <span className="ok">verified</span> against this
          trajectory. Hide those badges and a reviewer will bless a 0.10 episode that reads like a correct label
          answer.
        </p>
        <h2>Keyboard</h2>
        <p>
          On a replay URL (<code>/trajectories/&lt;id&gt;</code>, slashes allowed): ← previous step, → next
          step, space toggles autoplay.
        </p>
      </section>
      <Caption example={example} />
    </>
  );
}
