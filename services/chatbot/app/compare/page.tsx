"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useRef, useState } from "react";

import { TurnView } from "../../components/chat/TurnView";
import { EvalBrowser } from "../../components/eval/EvalBrowser";
import { streamChat } from "../../lib/api";
import {
  addToolCall,
  appendText,
  appendThinking,
  attachToolResult,
  finalizeLastAssistant,
  type Turn,
} from "../../lib/turns";

type View = "live" | "benchmark";

function isView(v: string | null): v is View {
  return v === "live" || v === "benchmark";
}

interface ModelOption {
  provider: string;
  model: string;
  label: string;
}

const MODEL_OPTIONS: ModelOption[] = [
  { provider: "opencode-go", model: "deepseek-v4-flash", label: "DeepSeek v4 Flash (OpenCode Go)" },
  { provider: "opencode-go", model: "glm-5", label: "GLM-5 (OpenCode Go)" },
  { provider: "opencode", model: "kimi-k2.5", label: "Kimi K2.5 (OpenCode Zen)" },
  { provider: "openai", model: "gpt-4o-mini", label: "GPT-4o mini (OpenAI)" },
  { provider: "google", model: "gemini-2.5-flash", label: "Gemini 2.5 Flash (Google)" },
  { provider: "vllm", model: "qwen2.5-0.5b-instruct", label: "Qwen 0.5B (vLLM)" },
  { provider: "huggingface", model: "qwen2.5-0.5b-instruct", label: "Qwen 0.5B (HF)" },
  { provider: "anthropic", model: "claude-sonnet-4-6", label: "Sonnet 4.6 (frontier)" },
  { provider: "anthropic", model: "claude-haiku-4-5-20251001", label: "Haiku 4.5" },
  { provider: "anthropic", model: "claude-opus-4-7", label: "Opus 4.7" },
];

function findModel(provider: string, model: string): ModelOption {
  const hit = MODEL_OPTIONS.find((o) => o.provider === provider && o.model === model);
  if (!hit) {
    throw new Error(`MODEL_OPTIONS missing ${provider}/${model}`);
  }
  return hit;
}

const DEFAULT_LEFT = findModel("opencode-go", "deepseek-v4-flash");
const DEFAULT_RIGHT = findModel("anthropic", "claude-sonnet-4-6");

interface ColumnState {
  pick: ModelOption;
  conversationId: string | null;
  turns: Turn[];
  usage: { input: number; output: number; cost_usd: number } | null;
  latencyMs: number | null;
  error: string | null;
  busy: boolean;
}

const initial = (pick: ModelOption): ColumnState => ({
  pick,
  conversationId: null,
  turns: [],
  usage: null,
  latencyMs: null,
  error: null,
  busy: false,
});

export default function ComparePage() {
  return (
    <Suspense fallback={<main className="mx-auto max-w-6xl p-4 text-sm text-neutral-500 sm:p-6">Loading…</main>}>
      <ComparePageInner />
    </Suspense>
  );
}

function ComparePageInner() {
  const search = useSearchParams();
  const router = useRouter();
  const view: View = isView(search.get("view")) ? (search.get("view") as View) : "live";

  const setView = useCallback(
    (next: View) => {
      const params = new URLSearchParams(search.toString());
      params.set("view", next);
      router.replace(`/compare?${params.toString()}`);
    },
    [router, search],
  );

  return (
    <main className="mx-auto max-w-6xl p-4 sm:p-6">
      <header className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <Link href="/" className="text-sm text-neutral-500 hover:underline">
            ← Back
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">Compare</h1>
          <p className="text-xs text-neutral-500">
            Live side-by-side prompting, or browse the saved benchmark suite.
          </p>
        </div>
        <nav className="inline-flex rounded border border-neutral-200 bg-white p-0.5 text-xs">
          {(["live", "benchmark"] as const).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              className={`rounded px-3 py-1 transition ${
                view === v
                  ? "bg-neutral-900 text-white"
                  : "text-neutral-700 hover:bg-neutral-100"
              }`}
            >
              {v === "live" ? "Live compare" : "Benchmark results"}
            </button>
          ))}
        </nav>
      </header>

      {view === "benchmark" ? <EvalBrowser /> : <LiveCompare />}
    </main>
  );
}

function LiveCompare() {
  const [prompt, setPrompt] = useState("");
  const [thinking, setThinking] = useState(false);
  const [left, setLeft] = useState<ColumnState>(initial(DEFAULT_LEFT));
  const [right, setRight] = useState<ColumnState>(initial(DEFAULT_RIGHT));
  const abortRef = useRef<AbortController | null>(null);

  function patchSide(side: "left" | "right", patch: (s: ColumnState) => ColumnState) {
    const setter = side === "left" ? setLeft : setRight;
    setter(patch);
  }

  async function runOne(
    side: "left" | "right",
    state: ColumnState,
    text: string,
    useThinking: boolean,
    signal: AbortSignal,
  ) {
    // Seed user turn + pending assistant turn, clear per-turn footer state.
    patchSide(side, (s) => ({
      ...s,
      turns: [
        ...s.turns,
        { kind: "user", content: text },
        { kind: "assistant", content: "", pending: true },
      ],
      usage: null,
      latencyMs: null,
      error: null,
      busy: true,
    }));
    const t0 = performance.now();
    try {
      for await (const chunk of streamChat(
        {
          conversation_id: state.conversationId ?? undefined,
          message: text,
          provider: state.pick.provider,
          model: state.pick.model,
          thinking: useThinking,
        },
        signal,
      )) {
        if (chunk.type === "meta") {
          const cid = chunk.conversation_id;
          if (cid) {
            patchSide(side, (s) => (s.conversationId ? s : { ...s, conversationId: cid }));
          }
        } else if (chunk.type === "text_delta") {
          patchSide(side, (s) => ({ ...s, turns: appendText(s.turns, chunk.delta) }));
        } else if (chunk.type === "thinking_delta") {
          patchSide(side, (s) => ({ ...s, turns: appendThinking(s.turns, chunk.delta) }));
        } else if (chunk.type === "tool_call") {
          patchSide(side, (s) => ({
            ...s,
            turns: addToolCall(s.turns, chunk.id, chunk.name, chunk.args),
          }));
        } else if (chunk.type === "tool_result") {
          patchSide(side, (s) => ({
            ...s,
            turns: attachToolResult(s.turns, chunk.id, chunk.result, chunk.is_error),
          }));
        } else if (chunk.type === "done") {
          patchSide(side, (s) => ({
            ...s,
            turns: finalizeLastAssistant(s.turns),
            usage: chunk.usage,
            latencyMs: performance.now() - t0,
            busy: false,
          }));
          return;
        } else if (chunk.type === "error") {
          patchSide(side, (s) => ({
            ...s,
            turns: finalizeLastAssistant(s.turns),
            error: chunk.error,
            busy: false,
          }));
          return;
        }
      }
      patchSide(side, (s) => ({
        ...s,
        turns: finalizeLastAssistant(s.turns),
        latencyMs: performance.now() - t0,
        busy: false,
      }));
    } catch (e) {
      patchSide(side, (s) => ({
        ...s,
        turns: finalizeLastAssistant(s.turns),
        error: String(e),
        busy: false,
      }));
    }
  }

  async function run() {
    const text = prompt.trim();
    if (!text) return;
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const sig = abortRef.current.signal;
    setPrompt("");
    await Promise.all([
      runOne("left", left, text, thinking, sig),
      runOne("right", right, text, thinking, sig),
    ]);
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLeft((s) => ({ ...s, turns: finalizeLastAssistant(s.turns), busy: false }));
    setRight((s) => ({ ...s, turns: finalizeLastAssistant(s.turns), busy: false }));
  }

  const busy = left.busy || right.busy;

  return (
    <div>
      <p className="mb-3 text-xs text-neutral-500">
        Send the same prompt to two models in parallel. Each column keeps its own
        conversation — follow-up questions reference earlier turns.
      </p>
      <form
        className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-end"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <textarea
          className="flex-1 resize-none rounded border p-2 text-sm"
          rows={2}
          placeholder="Prompt for both models…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void run();
            }
          }}
          disabled={busy}
        />
        <div className="flex items-center gap-2">
          <label className="flex min-h-[44px] flex-1 select-none items-center gap-1 text-xs text-neutral-600 sm:min-h-0 sm:flex-none">
            <input
              type="checkbox"
              checked={thinking}
              onChange={(e) => setThinking(e.target.checked)}
              disabled={busy}
            />
            thinking
          </label>
          {busy ? (
            <button
              type="button"
              className="min-h-[44px] flex-1 rounded bg-red-600 px-3 py-2 text-sm font-medium text-white sm:flex-none sm:min-h-0"
              onClick={stop}
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              className="min-h-[44px] flex-1 rounded bg-black px-3 py-2 text-sm font-medium text-white disabled:opacity-40 sm:flex-none sm:min-h-0"
              disabled={!prompt.trim()}
            >
              Run both
            </button>
          )}
        </div>
      </form>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Column
          title="Left"
          state={left}
          onModelChange={(opt) =>
            setLeft((s) => ({ ...s, pick: opt, usage: null, latencyMs: null, error: null }))
          }
          onReset={() => setLeft((s) => initial(s.pick))}
        />
        <Column
          title="Right"
          state={right}
          onModelChange={(opt) =>
            setRight((s) => ({ ...s, pick: opt, usage: null, latencyMs: null, error: null }))
          }
          onReset={() => setRight((s) => initial(s.pick))}
        />
      </div>
    </div>
  );
}

function Column({
  title,
  state,
  onModelChange,
  onReset,
}: {
  title: string;
  state: ColumnState;
  onModelChange: (opt: ModelOption) => void;
  onReset: () => void;
}) {
  const turnCount = state.turns.filter((t) => t.kind === "user").length;
  return (
    <section className="flex h-[50vh] flex-col rounded border sm:h-[55vh] md:h-[60vh]">
      <header className="flex flex-col gap-2 border-b bg-neutral-50 px-3 py-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-neutral-500">{title}</span>
          {turnCount > 0 && (
            <span className="rounded bg-neutral-200 px-1.5 py-0.5 text-[10px] text-neutral-700">
              {turnCount} turn{turnCount === 1 ? "" : "s"}
            </span>
          )}
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <select
            className="w-full rounded border px-2 py-1 text-xs sm:w-auto"
            value={`${state.pick.provider}/${state.pick.model}`}
            onChange={(e) => {
              const found = MODEL_OPTIONS.find(
                (o) => `${o.provider}/${o.model}` === e.target.value,
              );
              if (found) onModelChange(found);
            }}
            disabled={state.busy}
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={`${o.provider}/${o.model}`} value={`${o.provider}/${o.model}`}>
                {o.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={onReset}
            disabled={state.busy || state.turns.length === 0}
            className="rounded border px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-100 disabled:opacity-40"
            title="Start a fresh conversation for this column"
          >
            Reset
          </button>
        </div>
      </header>
      <div className="flex-1 space-y-3 overflow-y-auto p-3 text-sm">
        {state.turns.length === 0 ? (
          state.busy ? (
            <span className="text-neutral-400">…</span>
          ) : (
            <span className="text-neutral-400">Waiting for a prompt.</span>
          )
        ) : (
          state.turns.map((t, i) => <TurnView key={i} turn={t} />)
        )}
        {state.error && (
          <div className="rounded bg-red-50 px-2 py-1 text-xs text-red-700">{state.error}</div>
        )}
      </div>
      <footer className="border-t px-3 py-2 text-[11px] text-neutral-500">
        {state.usage ? (
          <>
            in {state.usage.input} · out {state.usage.output} · ${state.usage.cost_usd.toFixed(6)}
            {state.latencyMs != null && ` · ${Math.round(state.latencyMs)} ms`}
          </>
        ) : state.latencyMs != null ? (
          `${Math.round(state.latencyMs)} ms`
        ) : (
          <>&nbsp;</>
        )}
      </footer>
    </section>
  );
}
