/**
 * Zod schemas mirroring the FastAPI eval router payloads.
 *
 * The shapes are produced by `eval/run_eval.py:run_all()` and the
 * `services/api/routers/eval.py` endpoints. Keep this file in sync if either
 * side changes.
 */

import { z } from "zod";

// ── heuristic compliance ──────────────────────────────────────────────────

export const HeuristicSchema = z.object({
  persona_signature: z.boolean(),
  h1_present: z.boolean(),
  h2_present: z.boolean(),
  h3_present: z.boolean(),
  banned_phrases_found: z.array(z.string()),
  length_ok: z.boolean(),
  overall_pass: z.boolean(),
});
export type Heuristic = z.infer<typeof HeuristicSchema>;

// ── 3-judge DAG panel score (per axis) ────────────────────────────────────

export const JudgeNodeOutputSchema = z.object({
  node: z.string(),
  verdict: z.union([z.boolean(), z.string(), z.number()]).nullable().optional(),
  score: z.number().nullable().optional(),
  reason: z.string().default(""),
  latency_ms: z.number().optional(),
  cost_usd: z.number().optional(),
});
export type JudgeNodeOutput = z.infer<typeof JudgeNodeOutputSchema>;

export const JudgeOpinionSchema = z.object({
  judge_id: z.string(),
  score: z.number().nullable(),
  verdict_path: z.array(z.string()),
  node_outputs: z.array(JudgeNodeOutputSchema),
  latency_ms: z.number(),
  input_tokens: z.number().int(),
  output_tokens: z.number().int(),
  cost_usd: z.number(),
  status: z.enum(["success", "judge_failed"]),
  error: z.string().nullable().optional(),
});
export type JudgeOpinion = z.infer<typeof JudgeOpinionSchema>;

export const PanelAgreementSchema = z.object({
  binary_unanimous: z.boolean(),
  geval_stdev: z.number().nullable(),
  kappa_avg: z.number().nullable(),
});
export type PanelAgreement = z.infer<typeof PanelAgreementSchema>;

export const PanelScoreSchema = z.object({
  aggregated_score: z.number().nullable(),
  verdict_path_majority: z.array(z.string()),
  judges: z.array(JudgeOpinionSchema),
  agreement: PanelAgreementSchema,
  // Axis-specific extras:
  category_majority: z.string().nullable().optional(),   // bias
  llamaguard_pre_signal: z.string().optional(),          // safety
  toxicity_flagged: z.boolean().optional(),              // safety
  violations: z.array(z.string()).optional(),            // role_violation
});
export type PanelScore = z.infer<typeof PanelScoreSchema>;

// ── static row (hallucination / bias / safety / role_violation) ────────────

export const StaticScoresSchema = z.object({
  hallucination: PanelScoreSchema,
  bias: PanelScoreSchema,
  safety: PanelScoreSchema,
  role_violation: PanelScoreSchema,
});
export type StaticScores = z.infer<typeof StaticScoresSchema>;

export const StaticSideSchema = z.object({
  response: z.string(),
  thinking: z.string().default(""),
  latency_ms: z.number(),
  input_tokens: z.number().int(),
  output_tokens: z.number().int(),
  cost_usd: z.number(),
  status: z.string(),
  error: z.string().optional(),
  heuristic: HeuristicSchema,
  scores: StaticScoresSchema,
});
export type StaticSide = z.infer<typeof StaticSideSchema>;

export const StaticRowSchema = z.object({
  prompt_id: z.string(),
  category: z.string(),
  prompt: z.string(),
  expected_behavior: z.string(),
  kind: z.literal("static"),
  oss: StaticSideSchema,
  frontier: StaticSideSchema,
  // Present when the eval ran with the 3-column Llama Guard ablation
  // (run_eval.py sets metadata.guardrails_ablation=true).
  oss_guarded: StaticSideSchema.optional(),
});
export type StaticRow = z.infer<typeof StaticRowSchema>;

// ── agent row (trajectory) ────────────────────────────────────────────────

export const ToolStepSchema = z.object({
  hop: z.number().int(),
  call_id: z.string(),
  name: z.string(),
  arguments: z.record(z.string(), z.unknown()),
  result: z.unknown(),
  is_error: z.boolean(),
  error: z.string().nullable(),
  latency_ms: z.number(),
});
export type ToolStep = z.infer<typeof ToolStepSchema>;

export const AgentScoresSchema = z.object({
  tool_selection: PanelScoreSchema,
  argument_correctness: PanelScoreSchema,
  task_completion: PanelScoreSchema,
  output_grounding: PanelScoreSchema,
  safety_with_tools: PanelScoreSchema,
});
export type AgentScores = z.infer<typeof AgentScoresSchema>;

export const TrajectorySchema = z.object({
  prompt_id: z.string(),
  provider: z.string(),
  model: z.string(),
  user_prompt: z.string(),
  tool_calls: z.array(ToolStepSchema),
  final_text: z.string(),
  thinking: z.string().default(""),
  hops_used: z.number().int(),
  hit_max_hops: z.boolean(),
  status: z.string(),
  error: z.string().nullable(),
  latency_ms: z.number(),
  input_tokens: z.number().int(),
  output_tokens: z.number().int(),
  cost_usd: z.number(),
  heuristic: HeuristicSchema,
  scores: AgentScoresSchema,
});
export type Trajectory = z.infer<typeof TrajectorySchema>;

export const AgentRowSchema = z.object({
  prompt_id: z.string(),
  category: z.string(),
  prompt: z.string(),
  expected_behavior: z.string(),
  expected_tools: z.array(z.string()).nullable(),
  kind: z.literal("agent"),
  oss: TrajectorySchema,
  frontier: TrajectorySchema,
  // Present when the eval ran with the 3-column Llama Guard ablation.
  oss_guarded: TrajectorySchema.optional(),
});
export type AgentRow = z.infer<typeof AgentRowSchema>;

// ── full payload + run status ─────────────────────────────────────────────

export const MetadataSchema = z.object({
  run_id: z.string(),
  started_at: z.string(),
  completed_at: z.string(),
  sections: z.array(z.string()),
  limit: z.number().int().nullable(),
  oss_provider: z.string(),
  oss_model: z.string(),
  frontier_model: z.string(),
  judge_model: z.string(),                                  // legacy: "panel"
  judge_panel: z.array(z.string()).optional(),              // new: ["anthropic/claude-sonnet-4-6", ...]
  judge_total_cost_usd: z.number().optional(),
  judge_total_latency_ms: z.number().optional(),
  judge_per_judge: z.record(z.string(), z.object({
    cost_usd: z.number(),
    latency_ms: z.number(),
    calls: z.number().int(),
  })).optional(),
  static_prompts_total: z.number().int(),
  agent_prompts_total: z.number().int(),
  system_prompt_hash: z.string(),
  // Optional — set when the 3-column Llama Guard ablation ran.
  guardrails_ablation: z.boolean().optional(),
  guard_model: z.string().nullable().optional(),
});
export type Metadata = z.infer<typeof MetadataSchema>;

export const ResultsPayloadSchema = z.object({
  metadata: MetadataSchema,
  static_results: z.array(StaticRowSchema),
  agent_results: z.array(AgentRowSchema),
});
export type ResultsPayload = z.infer<typeof ResultsPayloadSchema>;

export const RunStatusSchema = z.object({
  run_id: z.string(),
  status: z.enum(["queued", "running", "done", "failed"]),
  sections: z.array(z.string()),
  limit: z.number().int().nullable(),
  started_at: z.string(),
  completed_at: z.string().nullable(),
  total_prompts: z.number().int(),
  completed_prompts: z.number().int(),
  current_prompt_id: z.string().nullable(),
  error: z.string().nullable(),
  partial_results: z.array(z.unknown()),
});
export type RunStatus = z.infer<typeof RunStatusSchema>;

export const RunAcceptedSchema = z.object({
  run_id: z.string(),
  status: z.string(),
});
export type RunAccepted = z.infer<typeof RunAcceptedSchema>;

// ── thin fetch helpers ────────────────────────────────────────────────────

import { API_BASE } from "./api";

export async function fetchResults(): Promise<ResultsPayload | null> {
  const res = await fetch(`${API_BASE}/eval/results`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`fetchResults: ${res.status} ${res.statusText}`);
  const raw = await res.json();
  return ResultsPayloadSchema.parse(raw);
}

export async function fetchLatestRun(): Promise<RunStatus | null> {
  const res = await fetch(`${API_BASE}/eval/runs/latest`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`fetchLatestRun: ${res.status} ${res.statusText}`);
  const raw = await res.json();
  return RunStatusSchema.parse(raw);
}

export async function startEvalRun(
  sections: ("static" | "agent")[],
  limit: number | null,
): Promise<RunAccepted> {
  const res = await fetch(`${API_BASE}/eval/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sections, limit }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`startEvalRun: ${res.status} ${text}`);
  }
  return RunAcceptedSchema.parse(await res.json());
}

export function reportPdfUrl(): string {
  return `${API_BASE}/eval/report.pdf`;
}
