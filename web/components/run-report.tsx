"use client";

import Link from "next/link";
import { useMemo, useState, type KeyboardEvent } from "react";

import { NativeLine, TipRow } from "@/components/native-chart";
import { fmtPct, fmtReward, kindLabel, replayPath, trajectoriesPath } from "@/lib/format";
import { stackWidths, type EpisodePoint, type ModelSlice, type RunReport } from "@/lib/run-stats";

type SparkPoint = EpisodePoint & { model: string };

function Spark({ models, hidden, runId }: { models: ModelSlice[]; hidden: Set<string>; runId: string }) {
  const visible = models.filter((m) => !hidden.has(m.name));
  return (
    <NativeLine
      series={visible.map((model) => ({
        id: model.name,
        color: model.color,
        points: model.points.map((p) => ({
          y: p.rollingMean,
          data: { ...p, model: model.short } satisfies SparkPoint,
        })),
      }))}
      ariaLabel="Cumulative mean reward by task order. Hover or arrow keys for the episode."
      floor={0.1}
      xStart="first task"
      xEnd="last task"
      liveText={({ data }) =>
        `${data.model} ${data.taskId} reward ${data.total.toFixed(2)} cumulative ${data.rollingMean.toFixed(2)}`
      }
      renderTip={({ data, pinned }) => (
        <>
          <div className="tip-title mono">{data.taskId}</div>
          <div className="sm tip-q">{data.question}</div>
          <TipRow k="model" v={data.model} />
          <TipRow k="episode R" v={fmtReward(data.total)} />
          <TipRow k="cum. mean" v={fmtReward(data.rollingMean)} />
          <div className="tip-links">
            <Link href={replayPath(data.id)}>replay →</Link>
            <Link href={`/?task=${encodeURIComponent(data.taskId)}`}>task →</Link>
            <Link href={trajectoriesPath({ run: runId, type: data.type })}>
              {data.type ? "type →" : "run →"}
            </Link>
          </div>
          {pinned ? <div className="muted sm">pinned · Esc to clear</div> : null}
        </>
      )}
    />
  );
}

function HistCell({
  model,
  n,
  max,
  modelN,
}: {
  model: ModelSlice;
  n: number;
  max: number;
  modelN: number;
}) {
  const [hover, setHover] = useState(false);
  const [pinned, setPinned] = useState(false);
  const show = hover || pinned;
  const pct = modelN === 0 ? 0 : n / modelN;
  const onKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "Escape") setPinned(false);
  };
  return (
    <button
      type="button"
      className="hist-col"
      onPointerEnter={() => setHover(true)}
      onMouseEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      onMouseLeave={() => setHover(false)}
      onClick={() => setPinned((prev) => !prev)}
      onBlur={() => setPinned(false)}
      onKeyDown={onKey}
      aria-label={`${model.short}: ${n} episodes, ${fmtPct(pct, 0)} of this model`}
    >
      <div className="hist-bar" style={{ height: `${(n / max) * 100}%`, background: model.color }} />
      <span className="mono sm">{n}</span>
      {show ? (
        <div className={`chart-tip glass hist-tip${pinned ? " pinned" : ""}`} role="status">
          <div className="tip-title">{model.short}</div>
          <TipRow k="count" v={n} />
          <TipRow k="of model" v={`${fmtPct(pct, 0)} · n=${modelN}`} />
        </div>
      ) : null}
    </button>
  );
}

function Heat({
  title,
  hint,
  rows,
  models,
  hidden,
  runId,
  facet,
}: {
  title: string;
  hint: string;
  rows: RunReport["byType"];
  models: ModelSlice[];
  hidden: Set<string>;
  runId: string;
  facet: "type" | "difficulty";
}) {
  const cols = models.filter((m) => !hidden.has(m.name));
  if (rows.length === 0 || cols.length === 0) return null;
  return (
    <div className="card glass">
      <div className="ch">
        <h2>{title}</h2>
        <span className="muted sm">{hint}</span>
      </div>
      <table className="heat">
        <thead>
          <tr>
            <th />
            {cols.map((model) => (
              <th key={model.name} className="num">
                {model.short}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              <td>{row.label}</td>
              {cols.map((model) => {
                const cell = row.cells[model.name];
                if (!cell) return <td key={model.name} className="num muted sm">—</td>;
                const href = trajectoriesPath({
                  run: runId,
                  model: model.short,
                  type: facet === "type" ? row.key : undefined,
                  difficulty: facet === "difficulty" ? row.key : undefined,
                });
                return (
                  <td key={model.name} className="num heat-cell">
                    <Link
                      href={href}
                      className="heat-link"
                      title={`${model.short} · ${row.label} · ${cell.n} episodes`}
                    >
                      <span className="heat-fill" style={{ opacity: 0.12 + cell.pass * 0.88, background: model.color }} />
                      <span className="mono b">{fmtReward(cell.mean)}</span>
                      <span className="muted sm">{fmtPct(cell.pass, 0)}</span>
                    </Link>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunReportView({ report }: { report: RunReport }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const visible = useMemo(() => report.models.filter((m) => !hidden.has(m.name)), [report.models, hidden]);
  const histMax = Math.max(
    1,
    ...report.histogram.flatMap((bucket) => visible.map((m) => bucket.counts[m.name] ?? 0)),
  );

  const toggle = (name: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else if (next.size < report.models.length - 1) next.add(name);
      return next;
    });
  };

  return (
    <>
      <div className="filters" style={{ marginTop: -4 }}>
        {report.models.map((model) => (
          <button
            key={model.name}
            type="button"
            className={`chip btnish${hidden.has(model.name) ? "" : " on"}`}
            onClick={() => toggle(model.name)}
            aria-pressed={!hidden.has(model.name)}
          >
            <i className="swatch" style={{ background: model.color }} />
            {model.short}
          </button>
        ))}
        <span className="muted sm">click to hide a series</span>
      </div>

      <section className="kpis">
        <div className="kpi glass">
          <div className="muted sm">Mean reward</div>
          <div className="big mono">{fmtReward(report.meanReward)}</div>
          <div className="sm muted">{report.n} episodes · {report.nTasks} tasks</div>
        </div>
        <div className="kpi glass">
          <div className="muted sm">Pass rate</div>
          <div className="big mono">{fmtPct(report.passRate)}</div>
          <div className="sm muted">correctness &gt; 0</div>
        </div>
        <div className="kpi glass">
          <div className="muted sm">Citation validity</div>
          <div className="big mono">{fmtPct(report.citation)}</div>
          <div className="sm muted">mean citation component</div>
        </div>
        <div className="kpi glass">
          <div className="muted sm">Spend</div>
          <div className="big mono">${report.cost.toFixed(2)}</div>
          <div className="sm muted">{report.meanTools.toFixed(1)} tools / task</div>
        </div>
      </section>

      <section className="card glass">
        <div className="ch">
          <h2>Models</h2>
          <span className="muted sm">this run only · bar is R = 0.60c + 0.25t + 0.10f + 0.05e</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Model</th>
              <th />
              <th className="num">Mean reward</th>
              <th className="num">Pass</th>
              <th className="num">Cite</th>
              <th className="num">Tools</th>
              <th className="num">Cost</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {report.models.map((model, i) => (
              <tr key={model.name} className={hidden.has(model.name) ? "dim" : undefined}>
                <td className="rk mono">{i + 1}</td>
                <td>
                  <div className="mname">{model.short}</div>
                  <div className="muted sm">{model.n} episodes · {model.meanLatencyS.toFixed(1)} s</div>
                </td>
                <td>
                  <span className="chip">{kindLabel(model.kind)}</span>
                </td>
                <td className="num">
                  <div className="rbar">
                    <span style={{ width: `${model.meanReward * 100}%`, background: model.color }} />
                  </div>
                  <span className="mono b">{fmtReward(model.meanReward)}</span>
                </td>
                <td className="num mono">{fmtPct(model.passRate)}</td>
                <td className="num mono">{fmtPct(model.citation)}</td>
                <td className="num mono">{model.meanTools.toFixed(1)}</td>
                <td className="num mono">${model.cost.toFixed(2)}</td>
                <td className="num">
                  <Link className="sm" href={trajectoriesPath({ run: report.runId, model: model.short })}>
                    list →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="stack-legend muted sm">
          {(["correctness", "citation", "format", "efficiency"] as const).map((key) => (
            <span key={key}>
              <i className={`stk ${key}`} /> {key}
            </span>
          ))}
        </div>
        <div className="stacks">
          {visible.map((model) => (
            <div key={model.name} className="stack-row">
              <div className="muted sm stack-lab">{model.short}</div>
              <div className="stack" title="weighted contribution">
                {stackWidths(model).map((part) => (
                  <i key={part.key} className={`stk ${part.key}`} style={{ width: `${part.pct}%` }} />
                ))}
              </div>
              <span className="mono sm">{fmtReward(model.meanReward)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="grid2">
        <div className="card glass">
          <div className="ch">
            <h2>How the run moved</h2>
            <span className="muted sm">cumulative mean reward, tasks in id order</span>
          </div>
          <Spark models={report.models} hidden={hidden} runId={report.runId} />
          <div className="muted sm">Dashed line is 0.10 — format-only floor (no valid citation).</div>
        </div>
        <div className="card glass">
          <div className="ch">
            <h2>Score mix</h2>
            <span className="muted sm">how many episodes landed on each atom</span>
          </div>
          <div className="hist">
            {report.histogram.map((bucket) => (
              <div key={bucket.key} className="hist-row">
                <div className="muted sm hist-lab">{bucket.label}</div>
                <div className="hist-bars">
                  {visible.map((model) => (
                    <HistCell
                      key={model.name}
                      model={model}
                      n={bucket.counts[model.name] ?? 0}
                      max={histMax}
                      modelN={model.n}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid2">
        <Heat
          title="By task type"
          hint="click a cell → those trajectories · cell is mean R · pass %"
          rows={report.byType}
          models={report.models}
          hidden={hidden}
          runId={report.runId}
          facet="type"
        />
        <Heat
          title="By difficulty"
          hint="structural mix, not empirical hardness · click a cell"
          rows={report.byDifficulty}
          models={report.models}
          hidden={hidden}
          runId={report.runId}
          facet="difficulty"
        />
      </section>

      <section className="grid2">
        <div className="card glass">
          <div className="ch">
            <h2>Where it broke</h2>
            <span className="muted sm">scorer notes, counted once per episode</span>
          </div>
          {report.notes.length === 0 ? (
            <div className="muted sm">No failure notes.</div>
          ) : (
            <ul className="fail-list">
              {report.notes.map((note) => (
                <li key={note.label}>
                  <span className="mono b">{note.n}</span>
                  <span>{note.label}</span>
                  <div className="fail-bar">
                    <span style={{ width: `${(note.n / report.n) * 100}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="card glass">
          <div className="ch">
            <h2>Edges</h2>
            <span className="muted sm">perfect 1.00 vs the worst six</span>
          </div>
          <div className="edge-col">
            <div className="muted sm">Best</div>
            {report.best.map((row) => (
              <Link key={row.id} href={replayPath(row.id)} className="edge">
                <span className="mono b ok">{fmtReward(row.total)}</span>
                <span className="muted sm mono">{row.model}</span>
                <span className="edge-q">{row.question}</span>
              </Link>
            ))}
            <div className="muted sm" style={{ marginTop: 10 }}>
              Worst
            </div>
            {report.worst.map((row) => (
              <Link key={row.id} href={replayPath(row.id)} className="edge">
                <span className="mono b bad">{fmtReward(row.total)}</span>
                <span className="muted sm mono">{row.model}</span>
                <span className="edge-q">{row.question}</span>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
