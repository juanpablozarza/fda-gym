"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { JsonBlock } from "@/components/json-block";
import {
  citationValid,
  compactArgs,
  fmtReward,
  fmtSigned,
  kindLabel,
  modelChip,
  sourceLabel,
  sourcesLabel,
  toolCallCount,
  WEIGHTS,
} from "@/lib/format";
import type { TaskMeta, Trajectory } from "@/lib/types";

function initialActive(trajectory: Trajectory): number {
  return Math.max(
    0,
    trajectory.steps.findIndex((s) => s.tool !== "submit_answer"),
  );
}

/** First two tool/result steps by `step.index` (usually 1 and 2) plus the active step. */
function defaultOpen(trajectory: Trajectory, activeIdx: number): Set<number> {
  const open = new Set<number>();
  for (const step of trajectory.steps.slice(0, 2)) open.add(step.index);
  const current = trajectory.steps[activeIdx];
  if (current) open.add(current.index);
  return open;
}

function stepMeta(trajectory: Trajectory, index: number): string {
  const step = trajectory.steps[index];
  if (!step) return "";
  if (step.tool === "submit_answer") return "graded";
  if (step.error) return "error";
  return `${Math.round(step.latency_ms)} ms`;
}

function RewardRow({
  label,
  detail,
  component,
  weight,
}: {
  label: string;
  detail: string;
  component: number;
  weight: number;
}) {
  const contribution = weight * component;
  return (
    <div className="rw">
      <div>
        <div>{label}</div>
        <div className="muted sm">{detail}</div>
      </div>
      <div className="rbar">
        <span className={component === 0 ? "neg" : undefined} style={{ width: `${Math.max(component, 0) * 100}%` }} />
      </div>
      <span className="mono b">{fmtSigned(contribution)}</span>
    </div>
  );
}

export function Replay({
  trajectory,
  task,
  prevId,
  nextId,
}: {
  trajectory: Trajectory;
  task?: TaskMeta;
  prevId?: string;
  nextId?: string;
}) {
  const [active, setActive] = useState(() => initialActive(trajectory));
  const [open, setOpen] = useState<Set<number>>(() => defaultOpen(trajectory, initialActive(trajectory)));

  const n = trajectory.steps.length;
  const citations = trajectory.final_answer?.citations ?? [];
  const calls = toolCallCount(trajectory);
  const tokens = trajectory.usage.prompt_tokens + trajectory.usage.completion_tokens;

  const go = useCallback(
    (dir: number) => {
      setActive((i) => {
        const next = Math.min(n - 1, Math.max(0, i + dir));
        setOpen((prev) => new Set(prev).add(trajectory.steps[next]?.index ?? next + 1));
        return next;
      });
    },
    [n, trajectory.steps],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        go(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        go(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  const toggle = (index: number) => {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const headerMeta = useMemo(() => {
    const bits = ["Question"];
    if (task) bits.push(sourcesLabel(task.sources));
    if (task) bits.push(task.difficulty);
    return bits.join(" · ");
  }, [task]);

  return (
    <>
      <header className="top">
        <div>
          <div className="muted sm crumbs">
            Trajectories / {trajectory.run_id} /{" "}
            <span className="mono">
              {trajectory.task_id} · seed {trajectory.seed}
            </span>
          </div>
          <h1>Trajectory replay</h1>
        </div>
        <div className="acts">
          <span className={`chip${trajectory.model.kind === "trained" ? " acc" : ""}`}>
            {kindLabel(trajectory.model.kind).split(" ")[0]} {modelChip(trajectory)}
          </span>
          {trajectory.model.checkpoint_step != null ? (
            <span className="chip">step {trajectory.model.checkpoint_step}</span>
          ) : (
            <span className="chip">{trajectory.model.kind}</span>
          )}
        </div>
      </header>

      <section className="card glass q">
        <div className="muted sm">{headerMeta}</div>
        <div className="qt">{task?.question ?? `${trajectory.task_id} · question not in results/tasks.json`}</div>
      </section>

      <section className="tgrid">
        <div className="card glass">
          <div className="ch">
            <h2>Steps</h2>
            <span className="muted sm">
              {calls} tool calls · {(trajectory.usage.latency_ms / 1000).toFixed(1)} s · {tokens.toLocaleString("en-US")}{" "}
              tokens
            </span>
          </div>
          <ol className="timeline">
            {trajectory.steps.map((step, i) => {
              const expanded = open.has(step.index);
              const isActive = i === active;
              return (
                <li key={step.index} className={`step${isActive ? " active" : ""}${expanded ? " open" : ""}`}>
                  <div className="node mono">{step.index}</div>
                  <div className="sbody glass2">
                    <div className="shead">
                      <button type="button" className="fn mono" onClick={() => setActive(i)}>
                        {step.tool}
                      </button>
                      <span className="muted sm mono">{stepMeta(trajectory, i)}</span>
                      <button type="button" className="tog sm" onClick={() => toggle(step.index)}>
                        {expanded ? "▾ hide result" : "▸ result"}
                      </button>
                    </div>
                    <div className="args mono sm">{compactArgs(step.arguments)}</div>
                    {step.thought ? <div className="thought sm">{step.thought}</div> : null}
                    {expanded ? (
                      step.error ? (
                        <pre className="json mono">{step.error}</pre>
                      ) : (
                        <JsonBlock value={step.output} />
                      )
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ol>
        </div>

        <div className="rcol">
          <div className="card glass">
            <div className="ch">
              <h2>Final answer</h2>
              {trajectory.reward.correctness > 0 ? (
                <span className="ok sm">✓ correctness &gt; 0</span>
              ) : (
                <span className="bad sm">✗ correctness 0</span>
              )}
            </div>
            {trajectory.final_answer ? (
              <>
                <p className="ans">
                  {String(trajectory.final_answer.answer)}{" "}
                  {citations.map((_, i) => (
                    <span key={i} className="cite">
                      {i + 1}
                    </span>
                  ))}
                </p>
                <ul className="cites">
                  {citations.map((cite, i) => {
                    const ok = citationValid(cite, trajectory.reward.notes);
                    return (
                      <li key={`${cite.id}:${cite.field}:${i}`}>
                        <span className="cn mono">[{i + 1}]</span>
                        <div>
                          <div className="mono sm">
                            {sourceLabel(cite.endpoint)} · {cite.id}
                          </div>
                          <div className="muted sm">{cite.field}</div>
                          <div className="muted sm it">&ldquo;{cite.quote}&rdquo;</div>
                        </div>
                        <span className={`cite-validity sm ${ok ? "ok" : "bad"}`}>
                          {ok ? "✓ verified" : "✗ invalid"}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </>
            ) : (
              <p className="muted">No valid submit_answer. terminated_by {trajectory.terminated_by}.</p>
            )}
          </div>

          <div className="card glass">
            <div className="ch">
              <h2>Reward</h2>
              <span className="total mono">{fmtReward(trajectory.reward.total)}</span>
            </div>
            <RewardRow
              label="Answer match"
              detail={trajectory.reward.correctness > 0 ? "correctness unlocked" : "gated or wrong"}
              component={trajectory.reward.correctness}
              weight={WEIGHTS.correctness}
            />
            <RewardRow
              label="Citation validity"
              detail={`${trajectory.reward.citations_valid} / ${trajectory.reward.citations_total} records resolve`}
              component={trajectory.reward.citation}
              weight={WEIGHTS.citation}
            />
            <RewardRow
              label="Format"
              detail={trajectory.reward.format === 1 ? "valid schema" : "invalid schema"}
              component={trajectory.reward.format}
              weight={WEIGHTS.format}
            />
            <RewardRow
              label="Efficiency"
              detail={`${calls} tool calls (submit_answer excluded)`}
              component={trajectory.reward.efficiency}
              weight={WEIGHTS.efficiency}
            />
            {trajectory.reward.notes.length > 0 ? (
              <div className="notes">
                {trajectory.reward.notes.map((note) => (
                  <div key={note} className="muted sm mono">
                    {note}
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          {(prevId || nextId) && (
            <div className="muted sm">
              {prevId ? <Link href={`/trajectories/${prevId}`}>← previous trajectory</Link> : null}
              {prevId && nextId ? " · " : null}
              {nextId ? <Link href={`/trajectories/${nextId}`}>next trajectory →</Link> : null}
            </div>
          )}
        </div>
      </section>
    </>
  );
}
