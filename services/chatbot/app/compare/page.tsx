"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useRef, useState } from "react";

import { EvalBrowser } from "../../components/eval/EvalBrowser";
import { streamChat } from "../../lib/api";

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
  text: string;
  thinking: string;
  usage: { input: number; output: number; cost_usd: number } | null;
  latencyMs: number | null;
  error: string | null;
  busy: boolean;
}

const initial = (pick: ModelOption): ColumnState => ({
  pick,
  text: "",
  thinking: "",
  usage: null,
  latencyMs: null,
  error: null,
  busy: false,
});

export default function ComparePage() {
  return (
    <Suspense fallback={<main className="mx-auto max-w-6xl p-6 text-sm text-neutral-500">Loading…</main>}>
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
    <main className="mx-auto max-w-6xl p-6">
      <header className="mb-4 flex items-center justify-between">
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

  function setSide(side: "left" | "right", patch: Partial<ColumnState>) {
    const setter = side === "left" ? setLeft : setRight;
    setter((s) => ({ ...s, ...patch }));
  }

  async function runOne(side: "left" | "right", pick: ModelOption, text: string, signal: AbortSignal) {
    setSide(side, { text: "", thinking: "", usage: null, latencyMs: null, error: null, busy: true });
    const t0 = performance.now();
    let acc = "";
    let thinkingAcc = "";
    try {
      for await (const chunk of streamChat(
        {
          message: text,
          provider: pick.provider,
          model: pick.model,
          thinking,
        },
        signal,
      )) {
        if (chunk.type === "text_delta") {
          acc += chunk.delta;
          setSide(side, { text: acc });
        } else if (chunk.type === "thinking_delta") {
          thinkingAcc += chunk.delta;
          setSide(side, { thinking: thinkingAcc });
        } else if (chunk.type === "done") {
          setSide(side, { usage: chunk.usage, latencyMs: performance.now() - t0, busy: false });
          return;
        } else if (chunk.type === "error") {
          setSide(side, { error: chunk.error, busy: false });
          return;
        }
      }
      setSide(side, { busy: false, latencyMs: performance.now() - t0 });
    } catch (e) {
      setSide(side, { error: String(e), busy: false });
    }
  }

  async function run() {
    const text = prompt.trim();
    if (!text) return;
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const sig = abortRef.current.signal;
    await Promise.all([runOne("left", left.pick, text, sig), runOne("right", right.pick, text, sig)]);
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLeft((s) => ({ ...s, busy: false }));
    setRight((s) => ({ ...s, busy: false }));
  }

  const busy = left.busy || right.busy;

  return (
    <div>
      <p className="mb-3 text-xs text-neutral-500">
        Send the same prompt to two models in parallel. Pick OSS on the left and Claude on the
        right to mirror the eval comparison.
      </p>
      <form
        className="mb-4 flex items-end gap-2"
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
        <label className="flex select-none items-center gap-1 text-xs text-neutral-600">
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
            className="rounded bg-red-600 px-3 py-2 text-sm font-medium text-white"
            onClick={stop}
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            className="rounded bg-black px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
            disabled={!prompt.trim()}
          >
            Run both
          </button>
        )}
      </form>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Column
          title="Left"
          state={left}
          onModelChange={(opt) => setLeft((s) => ({ ...s, pick: opt }))}
        />
        <Column
          title="Right"
          state={right}
          onModelChange={(opt) => setRight((s) => ({ ...s, pick: opt }))}
        />
      </div>
    </div>
  );
}

function Column({
  title,
  state,
  onModelChange,
}: {
  title: string;
  state: ColumnState;
  onModelChange: (opt: ModelOption) => void;
}) {
  return (
    <section className="flex h-[60vh] flex-col rounded border">
      <header className="flex items-center justify-between gap-2 border-b bg-neutral-50 px-3 py-2">
        <div className="text-xs uppercase tracking-wide text-neutral-500">{title}</div>
        <select
          className="rounded border px-2 py-1 text-xs"
          value={`${state.pick.provider}/${state.pick.model}`}
          onChange={(e) => {
            const found = MODEL_OPTIONS.find(
              (o) => `${o.provider}/${o.model}` === e.target.value,
            );
            if (found) onModelChange(found);
          }}
        >
          {MODEL_OPTIONS.map((o) => (
            <option key={`${o.provider}/${o.model}`} value={`${o.provider}/${o.model}`}>
              {o.label}
            </option>
          ))}
        </select>
      </header>
      <div className="flex-1 overflow-y-auto p-3 text-sm">
        {state.thinking && <CompareThinking text={state.thinking} />}
        <div className="whitespace-pre-wrap">
          {state.error ? (
            <span className="text-red-600">{state.error}</span>
          ) : state.text ? (
            state.text
          ) : state.busy ? (
            <span className="text-neutral-400">…</span>
          ) : (
            <span className="text-neutral-400">Waiting for a prompt.</span>
          )}
        </div>
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

function CompareThinking({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2 rounded border border-amber-200 bg-amber-50 text-xs">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2 py-1 hover:bg-amber-100"
      >
        <span className="text-amber-600">{open ? "▾" : "▸"}</span>
        <span className="font-mono text-[10px] uppercase tracking-wide text-amber-700">
          thinking ({text.length} chars)
        </span>
      </button>
      {open && (
        <pre className="whitespace-pre-wrap break-words border-t border-amber-200 px-2 py-1 text-[11px] text-amber-900">
          {text}
        </pre>
      )}
    </div>
  );
}
