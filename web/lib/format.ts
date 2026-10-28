import type { Citation, ModelKind, TaskMeta, Trajectory } from "./types";

export const TRAJECTORY_PAGE_SIZE = 25;

export const WEIGHTS = {
  correctness: 0.6,
  citation: 0.25,
  format: 0.1,
  efficiency: 0.05,
} as const;

export const SOURCE_LABEL: Record<string, string> = {
  "drug/label": "Drug labels",
  "drug/event": "Adverse events",
  "drug/drugsfda": "Approvals",
};

export function sourceLabel(source: string): string {
  return SOURCE_LABEL[source] ?? source;
}

export function sourcesLabel(sources: string[]): string {
  return sources.map(sourceLabel).join(" + ");
}

export function difficultyDots(level: TaskMeta["difficulty"]): string {
  if (level === "easy") return "●○○ easy";
  if (level === "medium") return "●●○ medium";
  return "●●● hard";
}

export function kindLabel(kind: ModelKind): string {
  if (kind === "reference") return "◇ reference";
  if (kind === "trained") return "● trained";
  return "○ baseline";
}

export function fmtReward(n: number, digits = 2): string {
  return n.toFixed(digits);
}

export function fmtPct(n: number, digits = 1): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtSigned(n: number, digits = 2): string {
  const body = n.toFixed(digits);
  return n >= 0 ? `+${body}` : body;
}

export function toolCallCount(row: { tool_calls?: number; steps?: Trajectory["steps"] }): number {
  if (typeof row.tool_calls === "number") return row.tool_calls;
  return (row.steps ?? []).filter((step) => step.tool !== "submit_answer").length;
}

export function compactArgs(args: Record<string, unknown>): string {
  return JSON.stringify(args);
}

export function citationValid(citation: Citation, notes: string[]): boolean {
  const hit = (prefix: string) =>
    notes.some((note) => note.startsWith(prefix) && note.includes(citation.id));
  if (hit("valid:")) return true;
  if (hit("invalid:")) return false;
  return false;
}

export function modelChip(trajectory: Trajectory): string {
  const short = trajectory.model.name.split("/").slice(-1)[0] ?? trajectory.model.name;
  if (trajectory.model.kind === "trained" && trajectory.model.checkpoint_step != null) {
    return `${short} · step ${trajectory.model.checkpoint_step}`;
  }
  return short;
}

/** Catch-all path. Slashes in env ids stay segments; do not encodeURIComponent. */
export function replayPath(id: string): string {
  return `/trajectories/${id}`;
}

export function runPath(runId: string): string {
  return `/runs/${runId}`;
}

export function trajectoriesPath(filters: {
  run?: string;
  model?: string;
  type?: string;
  difficulty?: string;
}): string {
  const params = new URLSearchParams();
  if (filters.run) params.set("run", filters.run);
  if (filters.model) params.set("model", filters.model);
  if (filters.type) params.set("type", filters.type);
  if (filters.difficulty) params.set("difficulty", filters.difficulty);
  const query = params.toString();
  return query ? `/trajectories?${query}` : "/trajectories";
}
