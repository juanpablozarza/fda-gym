import Link from "next/link";
import { notFound } from "next/navigation";

import { RunReportView } from "@/components/run-report";
import { Caption } from "@/components/shell";
import { getRun, loadCatalog } from "@/lib/results";
import { buildRunReport } from "@/lib/run-stats";

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const trajectories = getRun(id);
  if (trajectories.length === 0) notFound();
  const { tasks, example } = loadCatalog();
  const report = buildRunReport(id, trajectories, tasks);

  return (
    <>
      <header className="top">
        <div>
          <div className="muted sm crumbs">
            <Link href="/runs">Runs</Link> / <span className="mono">{id}</span>
          </div>
          <h1>Run report</h1>
          <div className="muted">
            {report.nSeeds} seed{report.nSeeds === 1 ? "" : "s"} · snapshot {report.snapshot} · {report.n} scored
            episodes
          </div>
        </div>
        <div className="acts">
          <Link className="btn ghost" href={`/trajectories?run=${encodeURIComponent(id)}`}>
            All trajectories
          </Link>
        </div>
      </header>
      <RunReportView report={report} />
      <Caption example={example} />
    </>
  );
}
