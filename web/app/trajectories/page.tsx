import Link from "next/link";

import { LiveRefresh } from "@/components/live-refresh";
import { Caption } from "@/components/shell";
import { TrajectoriesTable } from "@/components/trajectories-table";
import { toolCallCount, TRAJECTORY_PAGE_SIZE } from "@/lib/format";
import { getTask, loadCatalog } from "@/lib/results";

function pageHref(page: number, bits: { run?: string; model?: string; type?: string; difficulty?: string }): string {
  const params = new URLSearchParams();
  if (bits.run) params.set("run", bits.run);
  if (bits.model) params.set("model", bits.model);
  if (bits.type) params.set("type", bits.type);
  if (bits.difficulty) params.set("difficulty", bits.difficulty);
  if (page > 0) params.set("page", String(page));
  const query = params.toString();
  return query ? `/trajectories?${query}` : "/trajectories";
}

export default async function TrajectoriesPage({
  searchParams,
}: {
  searchParams: Promise<{ run?: string; model?: string; type?: string; difficulty?: string; page?: string }>;
}) {
  const { run, model, type, difficulty, page: pageRaw } = await searchParams;
  const { trajectories, example } = loadCatalog();
  const filtered = trajectories.filter((traj) => {
    if (run && traj.run_id !== run) return false;
    if (model) {
      const short = traj.model.name.split("/").slice(-1)[0];
      if (short !== model && traj.model.name !== model) return false;
    }
    const task = getTask(traj.task_id);
    if (type && task?.type !== type) return false;
    if (difficulty && task?.difficulty !== difficulty) return false;
    return true;
  });
  const total = filtered.length;
  const pageCount = Math.max(1, Math.ceil(total / TRAJECTORY_PAGE_SIZE));
  const page = Math.min(pageCount - 1, Math.max(0, Number.parseInt(pageRaw ?? "0", 10) || 0));
  const paged = filtered.slice(page * TRAJECTORY_PAGE_SIZE, (page + 1) * TRAJECTORY_PAGE_SIZE).map((traj) => {
    const task = getTask(traj.task_id);
    return {
      id: traj.id,
      taskId: traj.task_id,
      question: task?.question ?? "—",
      modelName: traj.model.name.split("/").slice(-1)[0] ?? traj.model.name,
      kind: traj.model.kind,
      total: traj.reward.total,
      citationsValid: traj.reward.citations_valid,
      citationsTotal: traj.reward.citations_total,
      tools: toolCallCount(traj),
    };
  });
  const filterBits = [run, model, type, difficulty].filter(Boolean);
  const bits = { run, model, type, difficulty };

  return (
    <>
      <LiveRefresh />
      <header className="top">
        <div>
          <h1>Trajectories</h1>
          <div className="muted">
            Click a row to replay steps. ←/→ once you are inside.
            {filterBits.length ? ` Filtered: ${filterBits.join(" · ")}.` : ""}
          </div>
        </div>
        {filterBits.length ? (
          <Link className="btn ghost" href="/trajectories">
            Clear filter
          </Link>
        ) : null}
      </header>
      <section className="card glass">
        <div className="ch">
          <h2>Logged episodes</h2>
          <span className="muted sm">
            {total}
            {total !== trajectories.length ? ` of ${trajectories.length}` : ""} in public/results/
          </span>
        </div>
        {filtered.length === 0 ? (
          <div className="empty">
            <div className="big">No trajectories</div>
            <div className="sm">Waiting on an eval jsonl dump.</div>
          </div>
        ) : (
          <>
            <TrajectoriesTable rows={paged} hidePager />
            <div className="pager">
              <span className="chip">{TRAJECTORY_PAGE_SIZE} / page</span>
              <span className="muted sm">
                {paged.length} of {total}
                {pageCount > 1 ? ` · page ${page + 1}/${pageCount}` : ""}
              </span>
              {page > 0 ? (
                <Link className="btn ghost" href={pageHref(page - 1, bits)}>
                  ◀ Prev
                </Link>
              ) : (
                <button type="button" className="btn ghost" disabled>
                  ◀ Prev
                </button>
              )}
              {page + 1 < pageCount ? (
                <Link className="btn ghost" href={pageHref(page + 1, bits)}>
                  Next ▶
                </Link>
              ) : (
                <button type="button" className="btn ghost" disabled>
                  Next ▶
                </button>
              )}
            </div>
          </>
        )}
      </section>
      <Caption example={example} />
    </>
  );
}
