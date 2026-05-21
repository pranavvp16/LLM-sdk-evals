"use client";

import Link from "next/link";
import { useRef, useState } from "react";

import { streamChat } from "../../lib/api";

interface ModelOption {
  provider: string;
  model: string;
  label: string;
}

const MODEL_OPTIONS: ModelOption[] = [
  { provider: "anthropic", model: "claude-sonnet-4-6", label: "Sonnet 4.6 (frontier)" },
  { provider: "anthropic", model: "claude-haiku-4-5-20251001", label: "Haiku 4.5" },
  { provider: "anthropic", model: "claude-opus-4-7", label: "Opus 4.7" },
];

interface ColumnState {
  pick: ModelOption;
  text: string;
  usage: { input: number; output: number; cost_usd: number } | null;
  latencyMs: number | null;
  error: string | null;
  busy: boolean;
}

const initial = (pick: ModelOption): ColumnState => ({
  pick,
  text: "",
  usage: null,
  latencyMs: null,
  error: null,
  busy: false,
});

export default function ComparePage() {
  const [prompt, setPrompt] = useState("");
  const [left, setLeft] = useState<ColumnState>(initial(MODEL_OPTIONS[0]));
  const [right, setRight] = useState<ColumnState>(initial(MODEL_OPTIONS[1]));
  const abortRef = useRef<AbortController | null>(null);

  function setSide(side: "left" | "right", patch: Partial<ColumnState>) {
    const setter = side === "left" ? setLeft : setRight;
    setter((s) => ({ ...s, ...patch }));
  }

  async function runOne(side: "left" | "right", pick: ModelOption, text: string, signal: AbortSignal) {
    setSide(side, { text: "", usage: null, latencyMs: null, error: null, busy: true });
    const t0 = performance.now();
    let acc = "";
    try {
      for await (const chunk of streamChat(
        { message: text, provider: pick.provider, model: pick.model },
        signal,
      )) {
        if (chunk.type === "text_delta") {
          acc += chunk.delta;
          setSide(side, { text: acc });
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
    <main className="mx-auto max-w-6xl p-6">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <Link href="/" className="text-sm text-neutral-500 hover:underline">
            ← Back
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">Side-by-side compare</h1>
          <p className="text-xs text-neutral-500">
            Send the same prompt to two models in parallel. Once vLLM/Qwen is hosted, swap the left
            column to OSS to reproduce the eval head-to-head.
          </p>
        </div>
      </header>

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
    </main>
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
      <div className="flex-1 overflow-y-auto whitespace-pre-wrap p-3 text-sm">
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
