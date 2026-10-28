"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { fmtPct, fmtReward, runPath } from "@/lib/format";
import type { RunListItem } from "@/lib/types";

export function RunsTable({ runs }: { runs: RunListItem[] }) {
  const router = useRouter();
  return (
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Models</th>
          <th className="num">Episodes</th>
          <th className="num">Mean reward</th>
          <th className="num">Pass</th>
          <th className="num">Cost</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => {
          const href = runPath(run.runId);
          return (
            <tr
              key={run.runId}
              className="clickable"
              onClick={(event) => {
                if (event.metaKey || event.ctrlKey) {
                  window.open(href, "_blank", "noopener,noreferrer");
                  return;
                }
                event.preventDefault();
                router.push(href);
              }}
            >
              <td className="mono">
                <Link href={href} className="traj-id">
                  {run.runId}
                </Link>
              </td>
              <td>
                <div className="trow">
                  {run.models.map((model) => (
                    <span key={model.name} className="chip">
                      {model.short}
                      <span className="muted">{fmtReward(model.mean)}</span>
                    </span>
                  ))}
                </div>
              </td>
              <td className="num mono">
                {run.n}
                <div className="muted sm">{run.nTasks} tasks</div>
              </td>
              <td className="num">
                <div className="rbar">
                  <span style={{ width: `${run.meanReward * 100}%` }} />
                </div>
                <span className="mono b">{fmtReward(run.meanReward)}</span>
              </td>
              <td className="num mono">{fmtPct(run.passRate)}</td>
              <td className="num mono">{run.cost > 0 ? `$${run.cost.toFixed(2)}` : "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
