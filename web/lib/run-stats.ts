import { toolCallCount, WEIGHTS } from "./format";
import type { Episode, ModelKind, TaskMeta } from "./types";

export const SERIES = ["#f5a524", "#7eb6ff", "#62c9a5", "#d4a5ff", "#ff8a7a"] as const;

export const TYPE_ORDER = [
  "boxed_warning",
  "approval",
  "faers_count",
  "faers_top_reaction",
  "composite",
  "label_open",
] as const;

export const DIFF_ORDER = ["easy", "medium", "hard"] as const;

const BUCKETS = [
  { key: "0.00", lo: -0.01, hi: 0.01, label: "0.00 format fail" },
  { key: "0.10", lo: 0.05, hi: 0.15, label: "0.10 format only" },
  { key: "0.35", lo: 0.25, hi: 0.45, label: "0.35 cited, wrong" },
  { key: "0.85", lo: 0.75, hi: 0.92, label: "0.85 correct, slow" },
  { key: "1.00", lo: 0.95, hi: 1.01, label: "1.00" },
] as const;

export type EpisodePoint = {
  i: number;
  id: string;
  taskId: string;
  question: string;
  total: number;
  rollingMean: number;
  type?: string;
};

export type ModelSlice = {
  name: string;
  short: string;
  kind: ModelKind;
  color: string;
  n: number;
  meanReward: number;
  passRate: number;
  correctness: number;
  citation: number;
  format: number;
  efficiency: number;
  meanTools: number;
  cost: number;
  meanLatencyS: number;
  points: EpisodePoint[];
};

export type FacetCell = {
  n: number;
  mean: number;
  pass: number;
};

export type FacetRow = {
  key: string;
  label: string;
  cells: Record<string, FacetCell>;
};

export type HistBucket = {
  key: string;
  label: string;
  counts: Record<string, number>;
};

export type RankedEpisode = {
  id: string;
  taskId: string;
  model: string;
  total: number;
  question: string;
};

export type RunReport = {
  runId: string;
  n: number;
  nTasks: number;
  nSeeds: number;
  snapshot: string;
  cost: number;
  meanReward: number;
  passRate: number;
  citation: number;
  meanTools: number;
  models: ModelSlice[];
  byType: FacetRow[];
  byDifficulty: FacetRow[];
  histogram: HistBucket[];
  notes: { label: string; n: number }[];
  best: RankedEpisode[];
  worst: RankedEpisode[];
};

export function shortModel(name: string): string {
  return name.split("/").slice(-1)[0] ?? name;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function clip(text: string, n = 80): string {
  if (text.length <= n) return text;
  return `${text.slice(0, n - 1)}…`;
}

function bucketFor(total: number): (typeof BUCKETS)[number]["key"] | "other" {
  for (const bucket of BUCKETS) {
    if (total > bucket.lo && total <= bucket.hi) return bucket.key;
  }
  return "other";
}

function noteLabel(note: string): string | null {
  if (note.startsWith("valid:")) return null;
  if (note.startsWith("format:")) return null;
  if (note.startsWith("total ")) return null;
  if (note.startsWith("efficiency")) return null;
  if (/^citation \d+\/\d+ valid/.test(note) && !note.includes(" 0/")) return null;
  if (note.startsWith("correctness gated")) return "correctness gated (no valid citation)";
  if (note.startsWith("invalid:")) {
    if (note.includes("field not in")) return "citation field missing from tool output";
    if (note.includes("quote")) return "quote did not match";
    if (note.includes("not in this trajectory") || note.includes("id")) return "cited id never retrieved";
    return "invalid citation";
  }
  return null;
}

function facet(
  groups: Map<string, Episode[]>,
  keys: readonly string[],
  label: (key: string) => string,
  pick: (traj: Episode, task?: TaskMeta) => string | undefined,
  tasks: Map<string, TaskMeta>,
): FacetRow[] {
  const extras = new Set<string>();
  for (const rows of groups.values()) {
    for (const traj of rows) {
      const key = pick(traj, tasks.get(traj.task_id));
      if (key && !keys.includes(key)) extras.add(key);
    }
  }
  const order = [...keys, ...[...extras].sort()];
  return order
    .map((key) => {
      const cells: Record<string, FacetCell> = {};
      let any = false;
      for (const [model, rows] of groups) {
        const slice = rows.filter((traj) => pick(traj, tasks.get(traj.task_id)) === key);
        if (slice.length === 0) continue;
        any = true;
        cells[model] = {
          n: slice.length,
          mean: mean(slice.map((t) => t.reward.total)),
          pass: mean(slice.map((t) => (t.reward.correctness > 0 ? 1 : 0))),
        };
      }
      return { key, label: label(key), cells, any };
    })
    .filter((row) => row.any)
    .map(({ key, label: text, cells }) => ({ key, label: text, cells }));
}

export function buildRunReport(runId: string, trajectories: Episode[], tasks: TaskMeta[]): RunReport {
  const taskMap = new Map(tasks.map((task) => [task.id, task]));
  const groups = new Map<string, Episode[]>();
  for (const traj of trajectories) {
    const list = groups.get(traj.model.name) ?? [];
    list.push(traj);
    groups.set(traj.model.name, list);
  }
  for (const rows of groups.values()) {
    rows.sort((a, b) => a.task_id.localeCompare(b.task_id) || a.seed - b.seed);
  }

  const ranked = [...groups.entries()].sort(
    (a, b) => mean(b[1].map((t) => t.reward.total)) - mean(a[1].map((t) => t.reward.total)),
  );

  const models: ModelSlice[] = ranked.map(([name, rows], i) => {
    let sum = 0;
    const points: EpisodePoint[] = rows.map((traj, idx) => {
      sum += traj.reward.total;
      const task = taskMap.get(traj.task_id);
      const question = task?.question ?? traj.task_id;
      return {
        i: idx,
        id: traj.id,
        taskId: traj.task_id,
        question: clip(question),
        total: traj.reward.total,
        rollingMean: sum / (idx + 1),
        type: task?.type,
      };
    });
    return {
      name,
      short: shortModel(name),
      kind: rows[0]?.model.kind ?? "baseline",
      color: SERIES[i % SERIES.length] ?? SERIES[0],
      n: rows.length,
      meanReward: mean(rows.map((t) => t.reward.total)),
      passRate: mean(rows.map((t) => (t.reward.correctness > 0 ? 1 : 0))),
      correctness: mean(rows.map((t) => t.reward.correctness)),
      citation: mean(rows.map((t) => t.reward.citation)),
      format: mean(rows.map((t) => t.reward.format)),
      efficiency: mean(rows.map((t) => t.reward.efficiency)),
      meanTools: mean(rows.map(toolCallCount)),
      cost: rows.reduce((acc, t) => acc + (t.usage.cost_usd ?? 0), 0),
      meanLatencyS: mean(rows.map((t) => t.usage.latency_ms)) / 1000,
      points,
    };
  });

  const histKeys = [...BUCKETS.map((b) => b.key), "other"] as const;
  const histogram: HistBucket[] = histKeys.map((key) => {
    const meta = BUCKETS.find((b) => b.key === key);
    const counts: Record<string, number> = {};
    for (const slice of models) {
      const rows = groups.get(slice.name) ?? [];
      counts[slice.name] = rows.filter((t) => bucketFor(t.reward.total) === key).length;
    }
    return { key, label: meta?.label ?? "other", counts };
  }).filter((bucket) => Object.values(bucket.counts).some((n) => n > 0));

  const noteCounts = new Map<string, number>();
  for (const traj of trajectories) {
    const seen = new Set<string>();
    for (const note of traj.reward.notes) {
      const label = noteLabel(note);
      if (!label || seen.has(label)) continue;
      seen.add(label);
      noteCounts.set(label, (noteCounts.get(label) ?? 0) + 1);
    }
  }
  const notes = [...noteCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label, n]) => ({ label, n }));

  const rankedEps = [...trajectories]
    .sort((a, b) => b.reward.total - a.reward.total || a.task_id.localeCompare(b.task_id))
    .map((traj) => ({
      id: traj.id,
      taskId: traj.task_id,
      model: shortModel(traj.model.name),
      total: traj.reward.total,
      question: taskMap.get(traj.task_id)?.question ?? traj.task_id,
    }));

  const typeLabel: Record<string, string> = {
    boxed_warning: "Boxed warning",
    approval: "Approval",
    faers_count: "FAERS count",
    faers_top_reaction: "FAERS top reaction",
    composite: "Composite",
    label_open: "Label open",
  };

  return {
    runId,
    n: trajectories.length,
    nTasks: new Set(trajectories.map((t) => t.task_id)).size,
    nSeeds: new Set(trajectories.map((t) => t.seed)).size,
    snapshot: trajectories[0]?.snapshot_version ?? "—",
    cost: trajectories.reduce((sum, t) => sum + (t.usage.cost_usd ?? 0), 0),
    meanReward: mean(trajectories.map((t) => t.reward.total)),
    passRate: mean(trajectories.map((t) => (t.reward.correctness > 0 ? 1 : 0))),
    citation: mean(trajectories.map((t) => t.reward.citation)),
    meanTools: mean(trajectories.map(toolCallCount)),
    models,
    byType: facet(groups, TYPE_ORDER, (key) => typeLabel[key] ?? key, (traj, task) => task?.type, taskMap),
    byDifficulty: facet(groups, DIFF_ORDER, (key) => key, (traj, task) => task?.difficulty, taskMap),
    histogram,
    notes,
    best: rankedEps.filter((row) => row.total >= 0.99).slice(0, 6),
    worst: [...rankedEps].reverse().slice(0, 6),
  };
}

export function stackWidths(model: ModelSlice): { key: keyof typeof WEIGHTS; pct: number }[] {
  return (Object.keys(WEIGHTS) as (keyof typeof WEIGHTS)[]).map((key) => ({
    key,
    pct: WEIGHTS[key] * model[key] * 100,
  }));
}
