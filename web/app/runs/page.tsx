import { LiveRefresh } from "@/components/live-refresh";
import { RunsTable } from "@/components/runs-table";
import { Caption } from "@/components/shell";
import { listRuns, loadCatalog } from "@/lib/results";

export default function RunsPage() {
  const { example } = loadCatalog();
  const runs = listRuns();

  return (
    <>
      <LiveRefresh />
      <header className="top">
        <div>
          <h1>Runs</h1>
          <div className="muted">One eval dump per jsonl. Click a row for the full report.</div>
        </div>
      </header>
      <section className="card glass">
        <div className="ch">
          <h2>Eval runs</h2>
          <span className="muted sm">{runs.length} on disk</span>
        </div>
        {runs.length === 0 ? (
          <div className="empty">
            <div className="big">No runs</div>
            <div className="sm">Copy results/&lt;run_id&gt;.jsonl into web/public/results/</div>
          </div>
        ) : (
          <RunsTable runs={runs} />
        )}
      </section>
      <Caption example={example} />
    </>
  );
}
