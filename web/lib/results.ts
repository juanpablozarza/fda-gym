import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { cache } from "react";

import { loadPlanA, type PlanA } from "./training";
import type {
  Catalog,
  Episode,
  Leaderboard,
  LeaderboardRow,
  ModelKind,
  RunListItem,
  RunSummary,
  TaskMeta,
  TaskStat,
  Trajectory,
} from "./types";

const RESULTS = path.join(process.cwd(), "public", "results");

const TASK_REQUIRED = ["id", "question", "type", "sources", "difficulty", "entity"] as const;
const TRAJ_REQUIRED = [
  "id",
  "run_id",
  "task_id",
  "model",
  "seed",
  "snapshot_version",
  "started_at",
  "steps",
  "final_answer",
  "terminated_by",
  "reward",
  "usage",
] as const;

function readJson(file: string): unknown | null {
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
  } catch {
    reportInvalid(file, "invalid JSON");
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function reportInvalid(file: string, reason: string, line?: number): void {
  console.warn("[dashboard] Ignoring invalid result data", {
    file: path.relative(RESULTS, file),
    ...(line === undefined ? {} : { line }),
    reason,
  });
}

function isKind(value: unknown): value is ModelKind {
  return value === "reference" || value === "baseline" || value === "trained";
}

function isTrajectory(value: unknown): value is Trajectory {
  if (!isRecord(value) || !TRAJ_REQUIRED.every((key) => key in value)) return false;
  return (
    typeof value.id === "string" && typeof value.run_id === "string" && typeof value.task_id === "string" &&
    isRecord(value.model) && typeof value.model.name === "string" && isKind(value.model.kind) &&
    (value.model.checkpoint_step === undefined || value.model.checkpoint_step === null ||
      typeof value.model.checkpoint_step === "number") &&
    typeof value.seed === "number" && Number.isFinite(value.seed) &&
    typeof value.snapshot_version === "string" && typeof value.started_at === "string" &&
    Array.isArray(value.steps) && value.steps.every(isStep) &&
    (value.final_answer === null || isAnswer(value.final_answer)) &&
    (value.terminated_by === "submit" || value.terminated_by === "max_steps" || value.terminated_by === "error") &&
    isReward(value.reward) && isUsage(value.usage)
  );
}

function isTask(value: unknown): value is TaskMeta {
  if (!isRecord(value) || !TASK_REQUIRED.every((key) => key in value)) return false;
  return (
    typeof value.id === "string" && typeof value.question === "string" && typeof value.type === "string" &&
    Array.isArray(value.sources) && value.sources.every((source) =>
      source === "drug/label" || source === "drug/event" || source === "drug/drugsfda",
    ) &&
    (value.difficulty === "easy" || value.difficulty === "medium" || value.difficulty === "hard") &&
    isRecord(value.entity) && typeof value.entity.generic_name === "string" &&
    (value.entity.brand_name === undefined || typeof value.entity.brand_name === "string")
  );
}

function isStep(value: unknown): boolean {
  return isRecord(value) && typeof value.index === "number" && typeof value.tool === "string" &&
    isRecord(value.arguments) && "output" in value && typeof value.latency_ms === "number" &&
    (value.thought === undefined || value.thought === null || typeof value.thought === "string") &&
    (value.error === undefined || value.error === null || typeof value.error === "string");
}

function isAnswer(value: unknown): boolean {
  if (!isRecord(value) || !Array.isArray(value.citations)) return false;
  const answer = value.answer;
  return (typeof answer === "string" || typeof answer === "number" || typeof answer === "boolean") &&
    value.citations.every((citation) => isRecord(citation) &&
      (citation.endpoint === "drug/label" || citation.endpoint === "drug/event" || citation.endpoint === "drug/drugsfda") &&
      typeof citation.id === "string" && typeof citation.field === "string" && typeof citation.quote === "string");
}

function isReward(value: unknown): boolean {
  return isRecord(value) && ["total", "correctness", "citation", "format", "efficiency", "citations_valid", "citations_total"]
    .every((key) => typeof value[key] === "number" && Number.isFinite(value[key])) &&
    Array.isArray(value.notes) && value.notes.every((note) => typeof note === "string");
}

function isUsage(value: unknown): boolean {
  return isRecord(value) && typeof value.prompt_tokens === "number" && typeof value.completion_tokens === "number" &&
    typeof value.latency_ms === "number" &&
    (value.cost_usd === undefined || value.cost_usd === null || typeof value.cost_usd === "number");
}

function isLeaderboardRow(value: unknown): value is LeaderboardRow {
  return isRecord(value) && typeof value.name === "string" && isKind(value.kind) && typeof value.detail === "string" &&
    typeof value.mean_reward === "number" && typeof value.pass_rate === "number" &&
    typeof value.citation_validity === "number" && typeof value.mean_tool_calls === "number" &&
    (value.cost_usd === null || typeof value.cost_usd === "number") &&
    (value.ours === undefined || typeof value.ours === "boolean") &&
    (value.n === undefined || typeof value.n === "number");
}

function isLeaderboard(value: unknown): value is Leaderboard {
  if (!isRecord(value) || typeof value.example !== "boolean" || typeof value.split !== "string" ||
      typeof value.n_tasks !== "number" || typeof value.dataset_tasks !== "number" || typeof value.n_seeds !== "number" ||
      typeof value.temperature !== "number" || typeof value.snapshot_version !== "string" || typeof value.env !== "string" ||
      typeof value.n_sources !== "number" || !Array.isArray(value.models) || !isRecord(value.kpis)) return false;
  const kpis = value.kpis;
  return ["reward_lift", "citation_validity", "tool_calls_per_task", "tool_calls_base", "gap_to_frontier"]
    .every((key) => kpis[key] === null || typeof kpis[key] === "number") && value.models.every(isLeaderboardRow);
}

function isRunSummary(value: unknown): value is RunSummary {
  return isRecord(value) && typeof value.run_id === "string" && typeof value.n_trajectories === "number" &&
    (value.example === undefined || typeof value.example === "boolean") &&
    Array.isArray(value.models) && value.models.every(isLeaderboardRow);
}

function validatedArray<T>(value: unknown, file: string, guard: (item: unknown) => item is T, label: string): T[] {
  if (!Array.isArray(value)) {
    if (value !== null) reportInvalid(file, `expected an array of ${label}s`);
    return [];
  }
  return value.flatMap((item, index) => {
    if (guard(item)) return [item];
    reportInvalid(file, `invalid ${label} row`, index + 1);
    return [];
  });
}

type Offset = { file: string; start: number; length: number };

type Store = {
  fingerprint: string;
  episodes: Episode[];
  offsets: Map<string, Offset>;
};

let store: Store | null = null;

function fingerprint(): string {
  if (!fs.existsSync(RESULTS)) return "";
  return fs
    .readdirSync(RESULTS)
    .filter((name) => name.endsWith(".jsonl"))
    .sort()
    .map((name) => {
      const st = fs.statSync(path.join(RESULTS, name));
      return `${name}:${st.size}:${st.mtimeMs}`;
    })
    .join("|");
}

function toEpisode(row: Trajectory): Episode {
  const { steps, ...rest } = row;
  return {
    ...rest,
    tool_calls: steps.filter((step) => step.tool !== "submit_answer").length,
  };
}

function indexJsonl(file: string): { episodes: Episode[]; offsets: Offset[] } {
  const buf = fs.readFileSync(file);
  const episodes: Episode[] = [];
  const offsets: Offset[] = [];
  let start = 0;
  let lineNo = 0;
  while (start < buf.length) {
    let end = buf.indexOf(0x0a, start);
    if (end < 0) end = buf.length;
    const line = buf.subarray(start, end).toString("utf8");
    const trimmed = line.trim();
    const length = end - start;
    start = end + 1;
    lineNo += 1;
    if (!trimmed) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed) as unknown;
    } catch {
      reportInvalid(file, "invalid JSON", lineNo);
      continue;
    }
    if (!isTrajectory(parsed)) {
      reportInvalid(file, "row does not match the trajectory shape", lineNo);
      continue;
    }
    episodes.push(toEpisode(parsed));
    offsets.push({ file: path.basename(file), start: end - length, length });
  }
  return { episodes, offsets };
}

const CACHE_DIR = path.join(RESULTS, ".cache");
const CACHE_FILE = path.join(CACHE_DIR, "store.json");

type DiskStore = {
  fingerprint: string;
  episodes: Episode[];
  offsets: Array<Offset & { id: string }>;
};

function storeFromDisk(fp: string): Store | null {
  const raw = readJson(CACHE_FILE);
  if (raw === null) return null;
  if (!isDiskStore(raw)) {
    reportInvalid(CACHE_FILE, "invalid cache shape; rebuilding from JSONL");
    return null;
  }
  if (raw.fingerprint !== fp) return null;
  const disk = raw;
  const offsets = new Map<string, Offset>();
  for (const rec of disk.offsets) offsets.set(rec.id, { file: rec.file, start: rec.start, length: rec.length });
  return { fingerprint: fp, episodes: disk.episodes, offsets };
}

function isEpisode(value: unknown): value is Episode {
  return isRecord(value) && typeof value.id === "string" && typeof value.run_id === "string" &&
    typeof value.task_id === "string" && isRecord(value.model) && typeof value.model.name === "string" &&
    isKind(value.model.kind) && typeof value.seed === "number" && typeof value.snapshot_version === "string" &&
    typeof value.started_at === "string" &&
    (value.terminated_by === "submit" || value.terminated_by === "max_steps" || value.terminated_by === "error") &&
    isReward(value.reward) && isUsage(value.usage) && typeof value.tool_calls === "number" &&
    (value.final_answer === null || isAnswer(value.final_answer));
}

function isDiskStore(value: unknown): value is DiskStore {
  return isRecord(value) && typeof value.fingerprint === "string" && Array.isArray(value.episodes) &&
    value.episodes.every(isEpisode) && Array.isArray(value.offsets) && value.offsets.every((offset) =>
      isRecord(offset) && typeof offset.id === "string" && typeof offset.file === "string" &&
      typeof offset.start === "number" && Number.isInteger(offset.start) && offset.start >= 0 &&
      typeof offset.length === "number" && Number.isInteger(offset.length) && offset.length >= 0,
    );
}

function writeDisk(next: Store): void {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  const payload: DiskStore = {
    fingerprint: next.fingerprint,
    episodes: next.episodes,
    offsets: [...next.offsets.entries()].map(([id, rec]) => ({ id, ...rec })),
  };
  const temp = path.join(CACHE_DIR, `store.${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temp, JSON.stringify(payload), { flag: "wx" });
    fs.renameSync(temp, CACHE_FILE);
  } catch (error) {
    reportInvalid(CACHE_FILE, `cache write failed (${error instanceof Error ? error.name : "unknown error"})`);
  } finally {
    try {
      if (fs.existsSync(temp)) fs.unlinkSync(temp);
    } catch {
      // Cache files are disposable; cleanup failure must not break page rendering.
    }
  }
}

function loadStore(): Store {
  const fp = fingerprint();
  if (store && store.fingerprint === fp) return store;
  const cached = storeFromDisk(fp);
  if (cached) {
    store = cached;
    return store;
  }
  const episodes: Episode[] = [];
  const offsets = new Map<string, Offset>();
  if (fs.existsSync(RESULTS)) {
    for (const name of fs.readdirSync(RESULTS)) {
      if (!name.endsWith(".jsonl")) continue;
      const full = path.join(RESULTS, name);
      if (!fs.statSync(full).isFile()) continue;
      const indexed = indexJsonl(full);
      for (let i = 0; i < indexed.episodes.length; i++) {
        const episode = indexed.episodes[i];
        const offset = indexed.offsets[i];
        if (!episode || !offset) continue;
        episodes.push(episode);
        offsets.set(episode.id, offset);
      }
    }
  }
  store = { fingerprint: fp, episodes, offsets };
  writeDisk(store);
  return store;
}

function taskStatsFrom(episodes: Episode[]): Record<string, TaskStat> {
  const groups = new Map<string, number[]>();
  for (const row of episodes) {
    const list = groups.get(row.task_id) ?? [];
    list.push(row.reward.total);
    groups.set(row.task_id, list);
  }
  const out: Record<string, TaskStat> = {};
  for (const [id, values] of groups) out[id] = { mean: mean(values), n: values.length };
  return out;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function modelDetail(kind: ModelKind, name: string): string {
  if (kind === "trained") return "LoRA · ours";
  if (kind === "reference") return "API · closed";
  if (name.includes("openrouter") || name.startsWith("openai/") || name.startsWith("anthropic/")) {
    return "API · tool-calling";
  }
  if (name.includes("base")) return "no fine-tune";
  return "no fine-tune";
}

function rowsFromPlanA(plan: PlanA): LeaderboardRow[] {
  const rows: LeaderboardRow[] = [
    {
      name: plan.base_model,
      kind: "baseline",
      detail: "no fine-tune · Plan A packed docs",
      mean_reward: plan.untrained_test_mean,
      pass_rate: 0,
      citation_validity: 0,
      mean_tool_calls: 1.13,
      cost_usd: 0,
      ours: false,
      n: 324,
    },
  ];
  for (const ckpt of plan.checkpoints) {
    if (ckpt.test_mean == null) continue;
    rows.push({
      name: ckpt.label,
      kind: "trained",
      detail: ckpt.stage === "sft" ? "LoRA · SFT · Plan A" : "LoRA · GRPO · Plan A",
      mean_reward: ckpt.test_mean,
      pass_rate: ckpt.pass_rate ?? 0,
      citation_validity: ckpt.citation ?? 0,
      mean_tool_calls: 1.13,
      cost_usd: 0,
      ours: true,
      n: 324,
    });
  }
  for (const row of plan.frontier) {
    rows.push({
      name: row.name,
      kind: "reference",
      detail: "API · Plan A packed docs",
      mean_reward: row.test_mean,
      pass_rate: row.pass_rate,
      citation_validity: row.citation,
      mean_tool_calls: 1.13,
      cost_usd: row.cost_usd,
      ours: false,
      n: row.n,
    });
  }
  return rows;
}

function mergePlanA(board: Leaderboard, plan: PlanA | null): Leaderboard {
  if (!plan) return board;
  const extra = rowsFromPlanA(plan);
  const names = new Set(extra.map((row) => row.name));
  const models = [...board.models.filter((row) => !names.has(row.name)), ...extra].sort(
    (a, b) => b.mean_reward - a.mean_reward,
  );
  const bestTrained = extra.filter((row) => row.kind === "trained").sort((a, b) => b.mean_reward - a.mean_reward)[0];
  const bestFrontier = extra.filter((row) => row.kind === "reference").sort((a, b) => b.mean_reward - a.mean_reward)[0];
  return {
    ...board,
    models,
    kpis: {
      ...board.kpis,
      reward_lift: plan.lift,
      gap_to_frontier:
        bestTrained && bestFrontier ? bestTrained.mean_reward - bestFrontier.mean_reward : board.kpis.gap_to_frontier,
    },
  };
}

function rowsFromTrajectories(trajectories: Episode[]): LeaderboardRow[] {
  const groups = new Map<string, Episode[]>();
  for (const traj of trajectories) {
    const key = `${traj.model.kind}::${traj.model.name}`;
    const list = groups.get(key) ?? [];
    list.push(traj);
    groups.set(key, list);
  }
  const rows: LeaderboardRow[] = [];
  for (const group of groups.values()) {
    const first = group[0];
    const kind = isKind(first.model.kind) ? first.model.kind : "baseline";
    const cited = group.map((t) => t.reward.citation);
    const costs = group.map((t) => t.usage.cost_usd ?? 0);
    rows.push({
      name: first.model.name,
      kind,
      detail: modelDetail(kind, first.model.name),
      mean_reward: mean(group.map((t) => t.reward.total)),
      pass_rate: mean(group.map((t) => (t.reward.correctness > 0 ? 1 : 0))),
      citation_validity: mean(cited),
      mean_tool_calls: mean(group.map((t) => t.tool_calls)),
      cost_usd: costs.some((c) => c > 0) ? costs.reduce((a, b) => a + b, 0) : null,
      ours: kind === "trained",
    });
  }
  rows.sort((a, b) => b.mean_reward - a.mean_reward);
  return rows;
}

function defaultLeaderboard(trajectories: Episode[], tasks: TaskMeta[], example: boolean): Leaderboard {
  const models = rowsFromTrajectories(trajectories);
  const baseline = models.find((m) => m.kind === "baseline");
  const datasetTasks = tasks.filter((task) => !task.id.startsWith("FG-9")).length || tasks.length;
  return {
    example,
    split: example ? "fixture" : "test",
    n_tasks: new Set(trajectories.map((t) => t.task_id)).size || datasetTasks,
    dataset_tasks: datasetTasks || 96,
    n_seeds: new Set(trajectories.map((t) => t.seed)).size || 1,
    temperature: 0.2,
    snapshot_version: trajectories[0]?.snapshot_version ?? "2026-10-15",
    env: "openfda-reg-v0.3",
    n_sources: 3,
    kpis: {
      reward_lift: null,
      citation_validity: baseline?.citation_validity ?? mean(models.map((m) => m.citation_validity)),
      tool_calls_per_task: baseline?.mean_tool_calls ?? mean(models.map((m) => m.mean_tool_calls)),
      tool_calls_base: null,
      gap_to_frontier: null,
    },
    models,
  };
}

export const loadShell = cache((): { env: string; nTasks: number; nSources: number; snapshot: string } => {
  const tasksFile = path.join(RESULTS, "tasks.json");
  const tasks = validatedArray(readJson(tasksFile), tasksFile, isTask, "task");
  const storedRaw = readJson(path.join(RESULTS, "leaderboard.json"));
  const stored = isLeaderboard(storedRaw) ? storedRaw : null;
  if (storedRaw !== null && stored === null) reportInvalid(path.join(RESULTS, "leaderboard.json"), "leaderboard does not match the expected shape");
  const datasetTasks = tasks.filter((task) => !task.id.startsWith("FG-9")).length || tasks.length;
  return {
    env: stored?.env ?? "openfda-reg-v0.3",
    nTasks: stored?.dataset_tasks ?? (datasetTasks || 96),
    nSources: stored?.n_sources ?? 3,
    snapshot: stored?.snapshot_version ?? "2026-10-15",
  };
});

function scanSummaries(): RunSummary[] {
  if (!fs.existsSync(RESULTS)) return [];
  const summaries: RunSummary[] = [];
  for (const name of fs.readdirSync(RESULTS)) {
    const full = path.join(RESULTS, name);
    if (!fs.statSync(full).isDirectory()) continue;
    const file = path.join(full, "summary.json");
    const summary = readJson(file);
    if (isRunSummary(summary)) summaries.push(summary);
    else if (summary !== null) reportInvalid(file, "summary does not match the expected shape");
  }
  return summaries;
}

export const loadCatalog = cache((): Catalog => {
  const indexed = loadStore();
  const scanned = scanSummaries();
  const tasksFile = path.join(RESULTS, "tasks.json");
  const tasks = validatedArray(readJson(tasksFile), tasksFile, isTask, "task");
  const real = indexed.episodes.filter((traj) => traj.run_id !== "fixture-run");
  const hasReal = real.length > 0;
  const trajectories = hasReal ? real : indexed.episodes;
  const summaries = hasReal ? scanned.filter((s) => !s.example) : scanned;
  const example = !hasReal;
  const storedRaw = readJson(path.join(RESULTS, "leaderboard.json"));
  const stored = isLeaderboard(storedRaw) ? storedRaw : null;
  if (storedRaw !== null && stored === null) reportInvalid(path.join(RESULTS, "leaderboard.json"), "leaderboard does not match the expected shape");
  const leaderboard = mergePlanA(
    hasReal
      ? defaultLeaderboard(trajectories, tasks, false)
      : (stored ?? defaultLeaderboard(trajectories, tasks, example)),
    loadPlanA(),
  );

  return {
    example,
    snapshot_version: leaderboard.snapshot_version,
    env: leaderboard.env,
    n_tasks: leaderboard.n_tasks,
    dataset_tasks: leaderboard.dataset_tasks ?? 96,
    n_sources: leaderboard.n_sources,
    leaderboard,
    tasks,
    trajectories,
    allTrajectories: indexed.episodes,
    summaries,
    taskStats: taskStatsFrom(trajectories),
  };
});

/** App Router `[...id]` splits on `/`. Join before lookup so env ids like `run/FG-0001/s0` resolve. */
export function joinTrajectoryId(segments: string | string[]): string {
  const parts = Array.isArray(segments) ? segments : [segments];
  return parts
    .map((part) => {
      try {
        return decodeURIComponent(part);
      } catch {
        return part;
      }
    })
    .join("/");
}

export function getTrajectory(id: string): Trajectory | undefined {
  const rec = loadStore().offsets.get(id);
  if (!rec) return undefined;
  const fd = fs.openSync(path.join(RESULTS, rec.file), "r");
  try {
    const buf = Buffer.alloc(rec.length);
    fs.readSync(fd, buf, 0, rec.length, rec.start);
    try {
      const parsed: unknown = JSON.parse(buf.toString("utf8").trim()) as unknown;
      if (isTrajectory(parsed)) return parsed;
      reportInvalid(path.join(RESULTS, rec.file), "trajectory shape changed after indexing");
      return undefined;
    } catch {
      reportInvalid(path.join(RESULTS, rec.file), "trajectory JSON changed after indexing");
      return undefined;
    }
  } finally {
    fs.closeSync(fd);
  }
}

export function getTask(taskId: string): TaskMeta | undefined {
  return loadCatalog().tasks.find((t) => t.id === taskId);
}

export function getRun(runId: string): Episode[] {
  return loadCatalog().trajectories.filter((traj) => traj.run_id === runId);
}

export function listRuns(): RunListItem[] {
  const { trajectories } = loadCatalog();
  const groups = new Map<string, Episode[]>();
  for (const traj of trajectories) {
    const list = groups.get(traj.run_id) ?? [];
    list.push(traj);
    groups.set(traj.run_id, list);
  }
  return [...groups.entries()]
    .map(([runId, rows]) => {
      const byModel = new Map<string, Episode[]>();
      for (const traj of rows) {
        const list = byModel.get(traj.model.name) ?? [];
        list.push(traj);
        byModel.set(traj.model.name, list);
      }
      const models = [...byModel.entries()]
        .map(([name, list]) => ({
          name,
          short: name.split("/").slice(-1)[0] ?? name,
          mean: mean(list.map((t) => t.reward.total)),
          n: list.length,
        }))
        .sort((a, b) => b.mean - a.mean);
      return {
        runId,
        n: rows.length,
        nTasks: new Set(rows.map((t) => t.task_id)).size,
        models,
        meanReward: mean(rows.map((t) => t.reward.total)),
        passRate: mean(rows.map((t) => (t.reward.correctness > 0 ? 1 : 0))),
        cost: rows.reduce((sum, t) => sum + (t.usage.cost_usd ?? 0), 0),
      };
    })
    .sort((a, b) => b.n - a.n);
}

export function neighbors(id: string): { prev?: Episode; next?: Episode } {
  const list = loadCatalog().trajectories;
  const i = list.findIndex((t) => t.id === id);
  if (i < 0) return {};
  return { prev: list[i - 1], next: list[i + 1] };
}

export function episodesForTask(taskId: string): Episode[] {
  return loadCatalog().trajectories.filter((row) => row.task_id === taskId);
}
