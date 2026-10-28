"use client";

import Link from "next/link";

import { NativeLine, TipRow } from "@/components/native-chart";
import { fmtReward, fmtSigned } from "@/lib/format";
import type { CurvePoint, TrainingRun } from "@/lib/training";

const COLORS = ["#f5a524", "#62c9a5", "#7aa2f7", "#e06c75"];
const FG_ID = /FG-\d+/g;

function fmtSci(n: number): string {
  if (n === 0) return "0";
  if (n >= 0.01 && n < 100) return n.toFixed(3);
  return n.toExponential(2);
}

function idsFrom(value: unknown): string[] {
  if (typeof value === "string") return value.match(FG_ID) ?? [];
  if (Array.isArray(value)) return value.flatMap(idsFrom);
  return [];
}

function TaskLinks({ ids }: { ids: string[] }) {
  if (ids.length === 0) return null;
  const unique = [...new Set(ids)];
  return (
    <div className="tip-links">
      {unique.map((id) => (
        <Link key={id} href={`/tasks?task=${encodeURIComponent(id)}`}>
          {id}
        </Link>
      ))}
    </div>
  );
}

function ExtraIds({ point }: { point: CurvePoint }) {
  const fromIds = idsFrom(point.task_ids);
  const fromBatch = idsFrom(point.batch);
  const ids = [...new Set([...fromIds, ...fromBatch])];
  if (ids.length === 0) return null;
  return (
    <>
      <TipRow k="tasks" v={`${ids.length} in this log step`} />
      <TaskLinks ids={ids} />
    </>
  );
}

function Spark({ run, color }: { run: TrainingRun; color: string }) {
  const last = run.points[run.points.length - 1];
  return (
    <NativeLine
      series={[
        {
          id: run.run_id,
          color,
          points: run.points.map((p) => ({ y: p.mean_reward, data: p })),
        },
      ]}
      ariaLabel={`${run.run_id} mean reward over ${run.points.length} log steps. Arrow keys inspect a step.`}
      floor={run.base ?? undefined}
      xStart={`step ${run.points[0]?.step ?? 0}`}
      xEnd={`step ${last.step} · ${run.points.length} logs`}
      liveText={({ data }) =>
        `${data.run_id} step ${data.step} mean ${data.mean_reward.toFixed(3)} stage ${data.stage}`
      }
      renderTip={({ i, data, pinned }) => {
        const prev = run.points[i - 1];
        const dPrev = prev ? data.mean_reward - prev.mean_reward : null;
        const dBase = run.base != null ? data.mean_reward - run.base : null;
        return (
          <>
            <div className="tip-title mono">{data.run_id}</div>
            <TipRow k="step" v={data.step} />
            <TipRow k="mean R" v={fmtReward(data.mean_reward, 4)} />
            <TipRow k="stage" v={data.stage} />
            {data.kl != null ? <TipRow k="kl" v={fmtSci(data.kl)} /> : null}
            {data.lr != null ? <TipRow k="lr" v={fmtSci(data.lr)} /> : null}
            {data.n != null ? <TipRow k="n" v={data.n} /> : null}
            {data.mix ? (
              <TipRow
                k="batch mix"
                v={`${data.mix.easy} easy · ${data.mix.medium} med · ${data.mix.hard} hard`}
              />
            ) : null}
            {dPrev != null ? <TipRow k="Δ prev" v={fmtSigned(dPrev, 3)} /> : null}
            {dBase != null ? <TipRow k="Δ base" v={fmtSigned(dBase, 3)} /> : null}
            <ExtraIds point={data} />
            {pinned ? <div className="muted sm">pinned · Esc to clear</div> : null}
          </>
        );
      }}
    />
  );
}

export function TrainingCurve({
  runs,
  emptyHint,
}: {
  runs: TrainingRun[];
  emptyHint?: string;
}) {
  if (runs.length === 0) {
    return (
      <div className="empty">
        <div className="big">no step log yet</div>
        <div className="sm">
          {emptyHint ?? (
            <>
              Plan A GRPO writes <span className="mono">results/&lt;run_id&gt;/training-curve.jsonl</span>. The dashed
              line is the untrained mean on the same split.
            </>
          )}
        </div>
      </div>
    );
  }
  return (
    <>
      {runs.map((run, i) => (
        <div key={run.run_id} className="curve-run">
          <Spark run={run} color={COLORS[i % COLORS.length]} />
          <div className="legend sm" style={{ marginTop: 10 }}>
            <span>
              <i className="lg-line" style={{ borderTopColor: COLORS[i % COLORS.length] }} />
              {run.run_id} · last {fmtReward(run.last.mean_reward)} · stage {run.last.stage}
            </span>
            {run.base != null ? (
              <span>
                <i className="lg-dash" />
                base {fmtReward(run.base)}
              </span>
            ) : null}
          </div>
        </div>
      ))}
    </>
  );
}
