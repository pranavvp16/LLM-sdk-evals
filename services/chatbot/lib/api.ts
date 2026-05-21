/**
 * Typed fetch client for the FastAPI backend.
 *
 * All API calls go through this module. Base URL is read from
 * NEXT_PUBLIC_API_URL (set in .env / docker-compose) and falls back to
 * http://localhost:8000 for local development.
 */

import { z } from "zod";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const SESSION_KEY = "ollive_session_id";

// ── session id (UUID stored in localStorage) ──────────────────────────────

export function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let id = window.localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function authHeaders(): HeadersInit {
  const id = getSessionId();
  return id ? { "X-Session-ID": id } : {};
}

// ── schemas ───────────────────────────────────────────────────────────────

export const ConversationSchema = z.object({
  id: z.string().uuid(),
  session_id: z.string().uuid(),
  title: z.string(),
  provider: z.string(),
  model: z.string(),
  status: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
});
export type Conversation = z.infer<typeof ConversationSchema>;

export const MessageSchema = z.object({
  id: z.string().uuid(),
  conversation_id: z.string().uuid(),
  role: z.enum(["user", "assistant", "tool_result"]),
  content: z.string(),
  token_count: z.number().int(),
  created_at: z.string(),
});
export type Message = z.infer<typeof MessageSchema>;

// ── conversation CRUD ─────────────────────────────────────────────────────

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/conversations`, { headers: authHeaders(), cache: "no-store" });
  if (!res.ok) throw new Error(`listConversations: ${res.status}`);
  return z.array(ConversationSchema).parse(await res.json());
}

export async function getConversation(id: string): Promise<Conversation & { messages: Message[] }> {
  const res = await fetch(`${API_BASE}/conversations/${id}`, { headers: authHeaders(), cache: "no-store" });
  if (!res.ok) throw new Error(`getConversation: ${res.status}`);
  const raw = await res.json();
  return {
    ...ConversationSchema.parse(raw),
    messages: z.array(MessageSchema).parse(raw.messages ?? []),
  };
}

export async function cancelConversation(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`cancelConversation: ${res.status}`);
}

export async function resumeConversation(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${id}/resume`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`resumeConversation: ${res.status}`);
}

// ── metrics ───────────────────────────────────────────────────────────────

export interface MetricsTotals {
  total_calls: number;
  success_calls: number;
  error_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface MetricsByProvider {
  provider: string;
  calls: number;
  cost_usd: number;
  p95_ms: number;
  error_rate: number;
}

export interface MetricsSummary {
  window_hours: number;
  totals: MetricsTotals;
  by_provider: MetricsByProvider[];
}

export async function getMetricsSummary(hours = 24): Promise<MetricsSummary> {
  const res = await fetch(`${API_BASE}/metrics/summary?hours=${hours}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`getMetricsSummary: ${res.status}`);
  return (await res.json()) as MetricsSummary;
}

export interface RecentCall {
  started_at: string;
  provider: string;
  model: string;
  status: string;
  latency_ms: number;
  total_tokens: number;
  cost_usd: number;
  preview: string;
  error_message: string | null;
}

export async function getRecentCalls(limit = 25): Promise<RecentCall[]> {
  const res = await fetch(`${API_BASE}/metrics/recent?limit=${limit}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`getRecentCalls: ${res.status}`);
  return (await res.json()) as RecentCall[];
}

// ── SSE streaming ─────────────────────────────────────────────────────────

export interface StreamRequest {
  conversation_id?: string;
  message: string;
  provider: string;
  model: string;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
}

export type StreamChunk =
  | { type: "meta"; conversation_id: string | null; session_id: string | null }
  | { type: "text_delta"; delta: string }
  | { type: "thinking_delta"; delta: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | {
      type: "tool_result";
      id: string;
      name: string;
      result: unknown;
      is_error: boolean;
    }
  | { type: "done"; usage: { input: number; output: number; cost_usd: number } }
  | { type: "error"; error: string; retryable: boolean };

export async function* streamChat(req: StreamRequest, signal?: AbortSignal): AsyncGenerator<StreamChunk> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`streamChat: ${res.status}`);

  // Synthetic first chunk — surfaces response headers so callers can lift
  // the new conversation id into their URL without a separate request.
  const sessionFromHeader = res.headers.get("X-Session-ID");
  if (sessionFromHeader && typeof window !== "undefined") {
    window.localStorage.setItem(SESSION_KEY, sessionFromHeader);
  }
  yield {
    type: "meta",
    conversation_id: res.headers.get("X-Conversation-ID"),
    session_id: sessionFromHeader,
  };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 2);
      if (!event.startsWith("data:")) continue;
      const payload = event.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        yield JSON.parse(payload) as StreamChunk;
      } catch {
        // skip malformed frame
      }
    }
  }
}
