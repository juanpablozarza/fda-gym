"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  difficultyDots,
  fmtReward,
  replayPath,
  sourceLabel,
  sourcesLabel,
  WEIGHTS,
} from "@/lib/format";
import { SERIES, shortModel } from "@/lib/run-stats";
import type { Episode, TaskMeta, TaskStat } from "@/lib/types";

type SourceFilter = "all" | "drug/label" | "drug/event" | "drug/drugsfda";
type DiffFilter = "any" | TaskMeta["difficulty"];

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function solveLabel(task: TaskMeta, stat?: TaskStat): string {
  if (!stat || stat.n === 0) return "no run yet";
  return `mean reward ${stat.mean.toFixed(2)}`;
}

function fmtAnswer(traj: Episode): string {
  const value = traj.final_answer?.answer;
  if (value == null) return "—";
  const text = String(value);
  return text.length > 72 ? `${text.slice(0, 72)}…` : text;
}

function failHint(traj: Episode): string | null {
  if (traj.reward.correctness > 0) return null;
  const invalid = traj.reward.notes.find((note) => note.startsWith("invalid:"));
  if (invalid) return invalid.replace(/^invalid:\s*/, "");
  const gated = traj.reward.notes.find((note) => note.startsWith("correctness"));
  return gated ?? null;
}

function pickForModel(hits: Episode[], model: string): Episode | undefined {
  return hits
    .filter((traj) => traj.model.name === model)
    .sort((a, b) => b.seed - a.seed || b.run_id.localeCompare(a.run_id))[0];
}

export function TaskExplorer({
  tasks,
  taskStats = {},
}: {
  tasks: TaskMeta[];
  taskStats?: Record<string, TaskStat>;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const selectedId = params.get("task");
  const compareRef = useRef<HTMLDivElement>(null);
  const [hits, setHits] = useState<Episode[]>([]);
  const [hitsTask, setHitsTask] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedId) {
      setHits([]);
      setHitsTask(null);
      return;
    }
    let cancelled = false;
    fetch(`/api/episodes?task=${encodeURIComponent(selectedId)}`)
      .then((res) => res.json())
      .then((rows: Episode[]) => {
        if (!cancelled) {
          setHits(rows);
          setHitsTask(selectedId);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHits([]);
          setHitsTask(selectedId);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const models = useMemo(() => {
    const groups = new Map<string, Episode[]>();
    for (const traj of hits) {
      const list = groups.get(traj.model.name) ?? [];
      list.push(traj);
      groups.set(traj.model.name, list);
    }
    return [...groups.entries()]
      .sort((a, b) => mean(b[1].map((t) => t.reward.total)) - mean(a[1].map((t) => t.reward.total)))
      .map(([name], i) => ({
        name,
        short: shortModel(name),
        color: SERIES[i % SERIES.length] ?? SERIES[0],
      }));
  }, [hits]);

  const [source, setSource] = useState<SourceFilter>("all");
  const [difficulty, setDifficulty] = useState<DiffFilter>("any");

  const counts = useMemo(() => {
    const by = (key: SourceFilter) =>
      key === "all" ? tasks.length : tasks.filter((t) => t.sources.includes(key)).length;
    return {
      all: by("all"),
      "drug/label": by("drug/label"),
      "drug/event": by("drug/event"),
      "drug/drugsfda": by("drug/drugsfda"),
    };
  }, [tasks]);

  const visible = tasks.filter((task) => {
    if (source !== "all" && !task.sources.includes(source)) return false;
    if (difficulty !== "any" && task.difficulty !== difficulty) return false;
    return true;
  });

  const selected = tasks.find((task) => task.id === selectedId) ?? null;
  const selectedHits = hitsTask === selectedId ? hits : [];
  const loadingHits = Boolean(selectedId) && hitsTask !== selectedId;

  useEffect(() => {
    if (selected) compareRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedId, selected]);

  const setTask = (id: string | null) => {
    const next = new URLSearchParams(params.toString());
    if (!id || id === selectedId) next.delete("task");
    else next.set("task", id);
    const query = next.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  };

  return (
    <section className="card glass explorer">
      <div className="ch">
        <h2>Tasks</h2>
        <div className="filters">
          <span className="muted sm">Source</span>
          {(
            [
              ["all", `All ${counts.all}`],
              ["drug/label", `Drug labels ${counts["drug/label"]}`],
              ["drug/event", `Adverse events ${counts["drug/event"]}`],
              ["drug/drugsfda", `Approvals ${counts["drug/drugsfda"]}`],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`chip btnish${source === key ? " on" : ""}`}
              onClick={() => setSource(key)}
            >
              {label}
            </button>
          ))}
          <span className="sep" />
          <span className="muted sm">Difficulty</span>
          {(["any", "easy", "medium", "hard"] as const).map((key) => (
            <button
              key={key}
              type="button"
              className={`chip btnish${difficulty === key ? " on" : ""}`}
              onClick={() => setDifficulty(key)}
            >
              {key === "any" ? "Any" : key[0].toUpperCase() + key.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {selected ? (
        <div ref={compareRef} className="compare glass2">
          <div className="compare-head">
            <div>
              <div className="trow">
                <span className="mono sm muted">{selected.id}</span>
                <span className="chip">{sourcesLabel(selected.sources)}</span>
                <span className="chip">{difficultyDots(selected.difficulty)}</span>
              </div>
              <div className="tq" style={{ marginTop: 6 }}>
                {selected.question}
              </div>
            </div>
            <button type="button" className="btn ghost" onClick={() => setTask(null)}>
              Close
            </button>
          </div>
          {loadingHits ? (
            <div className="muted sm">Loading episodes…</div>
          ) : models.length === 0 ? (
            <div className="muted sm">No eval episodes for this task yet.</div>
          ) : (
            <div className="compare-grid">
              {models.map((model) => {
                const traj = pickForModel(selectedHits, model.name);
                if (!traj) {
                  return (
                    <div key={model.name} className="compare-col">
                      <div className="mname">{model.short}</div>
                      <div className="muted sm">no episode</div>
                    </div>
                  );
                }
                const hint = failHint(traj);
                const passed = traj.reward.correctness > 0;
                return (
                  <div key={model.name} className="compare-col">
                    <div className="compare-model">
                      <i className="swatch" style={{ background: model.color }} />
                      <span className="mname">{model.short}</span>
                    </div>
                    <div className={`big mono${passed ? " ok" : ""}`}>{fmtReward(traj.reward.total)}</div>
                    <div className="stack" title="weighted contribution">
                      {(Object.keys(WEIGHTS) as (keyof typeof WEIGHTS)[]).map((key) => (
                        <i
                          key={key}
                          className={`stk ${key}`}
                          style={{ width: `${WEIGHTS[key] * traj.reward[key] * 100}%` }}
                        />
                      ))}
                    </div>
                    <div className="muted sm">{passed ? "pass" : "fail"}</div>
                    <div className="compare-ans mono sm">{fmtAnswer(traj)}</div>
                    <div className="muted sm">
                      cite {traj.reward.citations_valid}/{traj.reward.citations_total} · {traj.tool_calls} tools
                    </div>
                    {hint ? <div className="compare-hint muted sm">{hint}</div> : null}
                    <Link href={replayPath(traj.id)} className="sm">
                      replay →
                    </Link>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <div className="muted sm" style={{ marginBottom: 10 }}>
          Click a task to compare models side by side.
        </div>
      )}

      <div className="tasks">
        {visible.map((task) => {
          const on = task.id === selectedId;
          return (
            <button
              key={task.id}
              type="button"
              className={`task glass2${on ? " on" : ""}`}
              aria-pressed={on}
              onClick={() => setTask(task.id)}
            >
              <div className="trow">
                <span className="mono sm muted">{task.id}</span>
                <span className="chip">{sourcesLabel(task.sources)}</span>
                <span className="chip">{difficultyDots(task.difficulty)}</span>
              </div>
              <div className="tq">{task.question}</div>
              <div className="trow sm muted">
                {sourceLabel(task.sources[0] ?? "")}{" "}
                <span className="mono b ink">{solveLabel(task, taskStats[task.id])}</span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
