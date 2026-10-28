export type ModelKind = "reference" | "baseline" | "trained";

export type Citation = {
  endpoint: "drug/label" | "drug/event" | "drug/drugsfda";
  id: string;
  field: string;
  quote: string;
};

export type Answer = {
  answer: string | number | boolean;
  citations: Citation[];
};

export type Step = {
  index: number;
  thought?: string | null;
  tool: string;
  arguments: Record<string, unknown>;
  output: unknown;
  error?: string | null;
  latency_ms: number;
};

export type Reward = {
  total: number;
  correctness: number;
  citation: number;
  format: number;
  efficiency: number;
  citations_valid: number;
  citations_total: number;
  notes: string[];
};

export type Trajectory = {
  id: string;
  run_id: string;
  task_id: string;
  model: {
    name: string;
    kind: ModelKind;
    checkpoint_step?: number | null;
  };
  seed: number;
  snapshot_version: string;
  started_at: string;
  steps: Step[];
  final_answer: Answer | null;
  terminated_by: "submit" | "max_steps" | "error";
  reward: Reward;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd?: number | null;
    latency_ms: number;
  };
};

export type LeaderboardRow = {
  name: string;
  kind: ModelKind;
  detail: string;
  mean_reward: number;
  pass_rate: number;
  citation_validity: number;
  mean_tool_calls: number;
  cost_usd: number | null;
  ours?: boolean;
  n?: number;
};

export type Leaderboard = {
  example: boolean;
  split: string;
  n_tasks: number;
  dataset_tasks: number;
  n_seeds: number;
  temperature: number;
  snapshot_version: string;
  env: string;
  n_sources: number;
  kpis: {
    reward_lift: number | null;
    citation_validity: number | null;
    tool_calls_per_task: number | null;
    tool_calls_base: number | null;
    gap_to_frontier: number | null;
  };
  models: LeaderboardRow[];
};

export type RunSummary = {
  run_id: string;
  example?: boolean;
  n_trajectories: number;
  models: LeaderboardRow[];
};

export type TaskMeta = {
  id: string;
  question: string;
  type: string;
  sources: Array<"drug/label" | "drug/event" | "drug/drugsfda">;
  difficulty: "easy" | "medium" | "hard";
  entity: { generic_name: string; brand_name?: string };
};

export type RunListItem = {
  runId: string;
  n: number;
  nTasks: number;
  models: { name: string; short: string; mean: number; n: number }[];
  meanReward: number;
  passRate: number;
  cost: number;
};

/** List/index row: no tool `output`. Replay still loads a full `Trajectory`. */
export type Episode = {
  id: string;
  run_id: string;
  task_id: string;
  model: {
    name: string;
    kind: ModelKind;
    checkpoint_step?: number | null;
  };
  seed: number;
  snapshot_version: string;
  started_at: string;
  terminated_by: "submit" | "max_steps" | "error";
  reward: Reward;
  final_answer: Answer | null;
  usage: Trajectory["usage"];
  tool_calls: number;
};

export type TaskStat = { mean: number; n: number };

export type Catalog = {
  example: boolean;
  snapshot_version: string;
  env: string;
  n_tasks: number;
  dataset_tasks: number;
  n_sources: number;
  leaderboard: Leaderboard;
  tasks: TaskMeta[];
  trajectories: Episode[];
  allTrajectories: Episode[];
  summaries: RunSummary[];
  taskStats: Record<string, TaskStat>;
};
