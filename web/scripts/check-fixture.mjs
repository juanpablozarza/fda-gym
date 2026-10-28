#!/usr/bin/env node
/**
 * Offline check: every trajectory in public/results/*.jsonl has required
 * schema fields, and reward.total equals the AGENTS.md weighted sum.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const RESULTS = path.join(ROOT, "public", "results");
const WEIGHTS = { correctness: 0.6, citation: 0.25, format: 0.1, efficiency: 0.05 };
const REQUIRED = [
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
];

function loadJsonl(file) {
  return fs
    .readFileSync(file, "utf8")
    .split("\n")
    .map((line, i) => ({ i: i + 1, line: line.trim() }))
    .filter((row) => row.line)
    .map((row) => {
      const traj = JSON.parse(row.line);
      traj._line = row.i;
      traj._file = path.basename(file);
      return traj;
    });
}

// FDAEnv ids are `{run_id}/{task_id}/s{seed}`. The catch-all route
// `app/trajectories/[...id]` yields ['run','FG-0001','s0']; join before lookup.
// Same helper as web/lib/results.ts joinTrajectoryId. One-segment fixture ids
// still resolve: ['tr-fx-9001-s0'].join('/') === 'tr-fx-9001-s0'.
function joinTrajectoryId(segments) {
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
const envId = joinTrajectoryId(["run", "FG-0001", "s0"]);
if (envId !== "run/FG-0001/s0") {
  console.error(`catch-all join failed: got ${envId}`);
  process.exit(1);
}
const fixtureId = joinTrajectoryId(["tr-fx-9001-s0"]);
if (fixtureId !== "tr-fx-9001-s0") {
  console.error(`one-segment join failed: got ${fixtureId}`);
  process.exit(1);
}
const slashId = joinTrajectoryId([
  "baseline-2026-10-16",
  "openrouter",
  "openai",
  "gpt-4o-mini",
  "FG-0001",
  "s0",
]);
if (slashId !== "baseline-2026-10-16/openrouter/openai/gpt-4o-mini/FG-0001/s0") {
  console.error(`slash env-id join failed: got ${slashId}`);
  process.exit(1);
}
console.log(`catch-all join: ['run','FG-0001','s0'] -> ${envId}`);

const files = fs.readdirSync(RESULTS).filter((name) => name.endsWith(".jsonl"));
if (files.length === 0) {
  console.error("no jsonl in public/results/");
  process.exit(1);
}

let failed = 0;
const rows = [];
for (const name of files) {
  for (const traj of loadJsonl(path.join(RESULTS, name))) {
    const missing = REQUIRED.filter((key) => !(key in traj));
    const r = traj.reward ?? {};
    const expected =
      WEIGHTS.correctness * r.correctness +
      WEIGHTS.citation * r.citation +
      WEIGHTS.format * r.format +
      WEIGHTS.efficiency * r.efficiency;
    const drift = Math.abs(expected - r.total);
    const okTask = /^FG-\d{4}$/.test(traj.task_id);
    const ok = missing.length === 0 && okTask && drift < 1e-9;
    if (!ok) failed += 1;
    rows.push({
      file: traj._file,
      line: traj._line,
      id: traj.id,
      task: traj.task_id,
      total: r.total,
      expected,
      cite: `${r.citations_valid}/${r.citations_total}`,
      notes: (r.notes ?? []).filter((n) => n.startsWith("invalid:") || n.includes("gated")).join(" | "),
      ok,
      missing,
    });
  }
}

console.log("id\t\t\ttask\ttotal\texpected\tcite\tok");
for (const row of rows) {
  console.log(
    `${row.id}\t${row.task}\t${Number(row.total).toFixed(2)}\t${row.expected.toFixed(2)}\t${row.cite}\t${row.ok ? "yes" : "NO"}`,
  );
  if (row.notes) console.log(`  ${row.notes}`);
}

if (failed) {
  console.error(`\n${failed} trajectory(ies) failed`);
  process.exit(1);
}
console.log(`\n${rows.length} trajectories, totals match 0.60/0.25/0.10/0.05`);
