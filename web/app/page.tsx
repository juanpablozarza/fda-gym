import Link from "next/link";
import { Suspense } from "react";

import { LiveRefresh } from "@/components/live-refresh";
import { Caption } from "@/components/shell";
import { TaskExplorer } from "@/components/task-explorer";
import { TrainingCurve } from "@/components/training-curve";
import { fmtPct, fmtReward, fmtSigned, kindLabel } from "@/lib/format";
import { loadCatalog } from "@/lib/results";
import { loadPlanA, loadTraining } from "@/lib/training";

function Kpi({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="kpi glass">
      <div className="muted sm">{label}</div>
      <div className="big mono">{value}</div>
      <div className="sm muted">{hint}</div>
    </div>
  );
}

export default function OverviewPage() {
  const { leaderboard, tasks, taskStats, example } = loadCatalog();
  const { kpis, models } = leaderboard;
  const training = loadTraining();
  const planA = loadPlanA();
  const lift = kpis.reward_lift ?? planA?.lift ?? null;

  return (
    <>
      <LiveRefresh />
      <header className="top">
        <div>
          <h1>Overview</h1>
          <div className="muted">
            {leaderboard.split} split · {leaderboard.n_tasks} tasks · {leaderboard.n_seeds} seed
            {leaderboard.n_seeds === 1 ? "" : "s"} · temperature {leaderboard.temperature}
          </div>
        </div>
        <div className="acts">
          <span className="chip">Split: {leaderboard.split}</span>
          <span className="chip">Seed avg</span>
          <Link href="/docs" className="btn">
            Run eval
          </Link>
        </div>
      </header>

      <section className="kpis">
        <Kpi
          label="Reward lift (3B)"
          value={lift == null ? "—" : fmtSigned(lift)}
          hint={
            lift == null
              ? `Plan A TEST vs ${planA ? planA.untrained_test_mean.toFixed(4) : "0.0185"}`
              : `${planA ? planA.untrained_test_mean.toFixed(4) : "0.0185"} → Plan A TEST · packed docs`
          }
        />
        <Kpi
          label="Citation validity"
          value={kpis.citation_validity == null ? "—" : fmtPct(kpis.citation_validity)}
          hint="mean citation component"
        />
        <Kpi
          label="Tool calls / task"
          value={kpis.tool_calls_per_task == null ? "—" : kpis.tool_calls_per_task.toFixed(1)}
          hint={
            kpis.tool_calls_base != null
              ? `down from ${kpis.tool_calls_base.toFixed(1)} (base)`
              : "submit_answer excluded"
          }
        />
        <Kpi
          label="Gap to frontier"
          value={kpis.gap_to_frontier == null ? "—" : fmtSigned(kpis.gap_to_frontier)}
          hint={kpis.gap_to_frontier == null ? "no Plan A frontier eval yet" : "GRPO − Luna Plan A"}
        />
      </section>

      <section className="grid2">
        <div className="card glass lb">
          <div className="ch">
            <h2>Leaderboard</h2>
            <span className="muted sm">Plan A packed docs + OpenRouter tools · sorted by mean reward</span>
          </div>
          {models.length === 0 ? (
            <div className="empty">
              <div className="big">No eval run yet</div>
              <div className="sm">Copy results into web/public/results/ — see Docs.</div>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Model</th>
                  <th />
                  <th className="num">Mean reward</th>
                  <th className="num">Pass</th>
                  <th className="num">Citations ✓</th>
                  <th className="num">Tools</th>
                </tr>
              </thead>
              <tbody>
                {models.map((row, i) => (
                  <tr key={`${row.kind}:${row.name}`} className={row.ours ? "ours" : row.kind === "baseline" ? "base" : ""}>
                    <td className="rk mono">{i + 1}</td>
                    <td>
                      <div className="mname">{row.name.split("/").slice(-1)[0]}</div>
                      <div className="muted sm">{row.detail}</div>
                    </td>
                    <td>
                      <span className={`chip${row.kind === "trained" ? " acc" : ""}`}>{kindLabel(row.kind)}</span>
                    </td>
                    <td className="num">
                      <div className="rbar">
                        <span style={{ width: `${row.mean_reward * 100}%` }} />
                      </div>
                      <span className="mono b">{fmtReward(row.mean_reward)}</span>
                    </td>
                    <td className="num mono">{fmtPct(row.pass_rate)}</td>
                    <td className="num mono">{fmtPct(row.citation_validity)}</td>
                    <td className="num mono">{row.mean_tool_calls.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="card glass">
          <div className="ch">
            <h2>Training curve</h2>
            <span className="muted sm">GRPO + LoRA · 3B Plan A</span>
          </div>
          {planA ? (
            <div className="legend sm" style={{ marginBottom: 10 }}>
              {planA.checkpoints.map((ckpt) => (
                <a key={ckpt.id} href={ckpt.url} target="_blank" rel="noreferrer">
                  <span className={`chip${ckpt.stage === "grpo" ? " acc" : ""}`}>{ckpt.label}</span>
                </a>
              ))}
            </div>
          ) : null}
          <TrainingCurve runs={training.slice(0, 1)} />
        </div>
      </section>

      <Suspense fallback={<section className="card glass explorer" />}>
        <TaskExplorer tasks={tasks} taskStats={taskStats} />
      </Suspense>
      <Caption example={example} />
    </>
  );
}
