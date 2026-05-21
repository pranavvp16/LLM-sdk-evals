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

// ── static row (hallucination / bias / safety) ────────────────────────────

export const StaticScoresSchema = z.object({
  hallucination: z.number().int(),
  bias: z.number().int(),
  safety: z.number().int(),
  rationale: z.string(),
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
  tool_selection: z.number().int(),
  argument_correctness: z.number().int(),
  task_completion: z.number().int(),
  output_grounding: z.number().int(),
  safety_with_tools: z.number().int(),
  rationale: z.string(),
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
  judge_model: z.string(),
  static_prompts_total: z.number().int(),
  agent_prompts_total: z.number().int(),
  system_prompt_hash: z.string(),
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
