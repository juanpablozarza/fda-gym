import fs from "node:fs";
import path from "node:path";
import { cache } from "react";

export type CurvePoint = {
  step: number;
  mean_reward: number;
  stage: number;
  kl?: number | null;
  lr?: number | null;
  n?: number | null;
  mix?: { easy: number; medium: number; hard: number };
  base_mean_reward?: number | null;
  run_id: string;
  task_ids?: string[] | null;
  batch?: unknown;
};

export type TrainingRun = {
  run_id: string;
  points: CurvePoint[];
  last: CurvePoint;
  base: number | null;
};

export type PlanACheckpoint = {
  id: string;
  label: string;
  stage: "sft" | "grpo";
  hub: string;
  url: string;
  detail: string;
  test_mean: number | null;
  pass_rate: number | null;
  citation: number | null;
};

export type PlanATypeRow = {
  type: string;
  n: number;
  baseline: number;
  sft: number;
  grpo: number;
};

export type PlanAFrontier = {
  name: string;
  hub: string;
  test_mean: number;
  pass_rate: number;
  citation: number;
  cost_usd: number;
  n: number;
};

export type PlanA = {
  base_model: string;
  protocol: string;
  split: string;
  untrained_test_mean: number;
  beats_margin: number;
  first_grpo: { id: string; train_mean_approx: number; note: string };
  checkpoints: PlanACheckpoint[];
  frontier: PlanAFrontier[];
  by_type: PlanATypeRow[];
  lift: number | null;
};

const RESULTS = path.join(process.cwd(), "public", "results");

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function reportInvalid(file: string, reason: string, line?: number): void {
  console.warn("[dashboard] Ignoring invalid training data", {
    file: path.relative(RESULTS, file),
    ...(line === undefined ? {} : { line }),
    reason,
  });
}

function isPoint(value: unknown): value is Omit<CurvePoint, "run_id"> {
  if (!isRecord(value) || typeof value.step !== "number" || typeof value.mean_reward !== "number" ||
      typeof value.stage !== "number") return false;
  return (
    (value.kl === undefined || value.kl === null || typeof value.kl === "number") &&
    (value.lr === undefined || value.lr === null || typeof value.lr === "number") &&
    (value.n === undefined || value.n === null || typeof value.n === "number") &&
    (value.base_mean_reward === undefined || value.base_mean_reward === null || typeof value.base_mean_reward === "number") &&
    (value.task_ids === undefined || value.task_ids === null ||
      (Array.isArray(value.task_ids) && value.task_ids.every((id) => typeof id === "string"))) &&
    (value.mix === undefined || (isRecord(value.mix) && typeof value.mix.easy === "number" &&
      typeof value.mix.medium === "number" && typeof value.mix.hard === "number"))
  );
}

function loadFile(file: string, runId: string): CurvePoint[] {
  const text = fs.readFileSync(file, "utf8");
  const points: CurvePoint[] = [];
  for (const [index, line] of text.split("\n").entries()) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed) as unknown;
      if (isPoint(parsed)) points.push({ ...parsed, run_id: runId });
      else reportInvalid(file, "row does not match the training point shape", index + 1);
    } catch {
      reportInvalid(file, "invalid JSON", index + 1);
    }
  }
  return points;
}

export const loadTraining = cache((): TrainingRun[] => {
  if (!fs.existsSync(RESULTS)) return [];
  const runs: TrainingRun[] = [];
  for (const name of fs.readdirSync(RESULTS)) {
    const full = path.join(RESULTS, name);
    if (!fs.statSync(full).isDirectory()) continue;
    const file = path.join(full, "training-curve.jsonl");
    if (!fs.existsSync(file)) continue;
    const points = loadFile(file, name);
    if (points.length === 0) continue;
    const last = points[points.length - 1];
    const base = points.find((p) => p.base_mean_reward != null)?.base_mean_reward ?? null;
    runs.push({ run_id: name, points, last, base });
  }
  runs.sort((a, b) => b.last.mean_reward - a.last.mean_reward);
  return runs;
});

function isCheckpoint(value: unknown): value is PlanACheckpoint {
  if (!isRecord(value)) return false;
  return (
    typeof value.id === "string" &&
    typeof value.label === "string" &&
    (value.stage === "sft" || value.stage === "grpo") &&
    typeof value.hub === "string" &&
    typeof value.url === "string" &&
    typeof value.detail === "string" &&
    (value.test_mean === null || typeof value.test_mean === "number") &&
    (value.pass_rate === undefined || value.pass_rate === null || typeof value.pass_rate === "number") &&
    (value.citation === undefined || value.citation === null || typeof value.citation === "number")
  );
}

function isFrontier(value: unknown): value is PlanAFrontier {
  if (!isRecord(value)) return false;
  return (
    typeof value.name === "string" &&
    typeof value.hub === "string" &&
    typeof value.test_mean === "number" &&
    typeof value.pass_rate === "number" &&
    typeof value.citation === "number" &&
    (value.cost_usd === undefined || typeof value.cost_usd === "number") &&
    (value.n === undefined || typeof value.n === "number")
  );
}

function isTypeRow(value: unknown): value is PlanATypeRow {
  if (!isRecord(value)) return false;
  return (
    typeof value.type === "string" &&
    typeof value.n === "number" &&
    typeof value.baseline === "number" &&
    typeof value.sft === "number" &&
    typeof value.grpo === "number"
  );
}

export const loadPlanA = cache((): PlanA | null => {
  const file = path.join(RESULTS, "plan-a.json");
  if (!fs.existsSync(file)) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
  } catch {
    reportInvalid(file, "invalid JSON");
    return null;
  }
  if (!isRecord(parsed)) {
    reportInvalid(file, "expected an object");
    return null;
  }
  const raw = parsed;
  if (typeof raw.untrained_test_mean !== "number" || !Array.isArray(raw.checkpoints)) {
    reportInvalid(file, "Plan A data is missing required fields");
    return null;
  }
  const checkpoints = raw.checkpoints.filter(isCheckpoint);
  raw.checkpoints.forEach((checkpoint, index) => {
    if (!isCheckpoint(checkpoint)) reportInvalid(file, "invalid checkpoint row", index + 1);
  });
  if (checkpoints.length === 0) return null;
  const first = isRecord(raw.first_grpo) ? raw.first_grpo : undefined;
  if (!first || typeof first.train_mean_approx !== "number" || typeof first.note !== "string") return null;
  const scored = checkpoints.map((c) => c.test_mean).filter((n): n is number => n != null);
  const best = scored.length ? Math.max(...scored) : null;
  return {
    base_model: typeof raw.base_model === "string" ? raw.base_model : "Qwen/Qwen2.5-3B-Instruct",
    protocol: typeof raw.protocol === "string" ? raw.protocol : "plan-a",
    split: typeof raw.split === "string" ? raw.split : "test",
    untrained_test_mean: raw.untrained_test_mean,
    beats_margin: typeof raw.beats_margin === "number" ? raw.beats_margin : 0.03,
    first_grpo: {
      id: typeof first.id === "string" ? first.id : "grpo-qwen3b",
      train_mean_approx: first.train_mean_approx,
      note: first.note,
    },
    checkpoints,
    frontier: Array.isArray(raw.frontier) ? raw.frontier.filter(isFrontier) : [],
    by_type: Array.isArray(raw.by_type) ? raw.by_type.filter(isTypeRow) : [],
    lift: best == null ? null : best - raw.untrained_test_mean,
  };
});
