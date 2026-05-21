"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import {
  cancelConversation,
  getConversation,
  type Message,
  streamChat,
} from "../../../lib/api";

interface PageProps {
  params: { id: string };
}

type Turn =
  | { kind: "user"; content: string }
  | { kind: "assistant"; content: string; thinking?: string; pending?: boolean }
  | {
      kind: "tool";
      id: string;
      name: string;
      args: Record<string, unknown>;
      result?: unknown;
      isError?: boolean;
    }
  | { kind: "system"; content: string };

export default function ChatPage({ params }: PageProps) {
  const { id } = params;
  const isNew = id === "new";
  const search = useSearchParams();
  const router = useRouter();

  const [conversationId, setConversationId] = useState<string | null>(isNew ? null : id);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [usage, setUsage] = useState<{ input: number; output: number; cost_usd: number } | null>(
    null,
  );
  const abortRef = useRef<AbortController | null>(null);
  const provider = search.get("provider") ?? "anthropic";
  const model = search.get("model") ?? "claude-sonnet-4-6";

  // Load history for an existing conversation.
  useEffect(() => {
    if (isNew) return;
    let cancelled = false;
    (async () => {
      try {
        const conv = await getConversation(id);
        if (cancelled) return;
        const loaded: Turn[] = conv.messages
          .filter((m: Message) => m.role === "user" || m.role === "assistant")
          .map((m): Turn =>
            m.role === "user"
              ? { kind: "user", content: m.content }
              : { kind: "assistant", content: m.content },
          );
        setTurns(loaded);
      } catch (err) {
        console.error(err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id, isNew]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setBusy(true);
    setInput("");
    setTurns((t) => [
      ...t,
      { kind: "user", content: text },
      { kind: "assistant", content: "", pending: true },
    ]);
    setUsage(null);

    abortRef.current = new AbortController();

    try {
      for await (const chunk of streamChat(
        {
          conversation_id: conversationId ?? undefined,
          message: text,
          provider,
          model,
          thinking: thinking ? true : undefined,
        },
        abortRef.current.signal,
      )) {
        if (chunk.type === "meta") {
          if (chunk.conversation_id && chunk.conversation_id !== conversationId) {
            setConversationId(chunk.conversation_id);
            if (isNew) {
              const qs = new URLSearchParams({ provider, model });
              router.replace(`/chat/${chunk.conversation_id}?${qs.toString()}`);
            }
          }
        } else if (chunk.type === "text_delta") {
          setTurns((t) => appendText(t, chunk.delta));
        } else if (chunk.type === "thinking_delta") {
          setTurns((t) => appendThinking(t, chunk.delta));
        } else if (chunk.type === "tool_call") {
          setTurns((t) => addToolCall(t, chunk.id, chunk.name, chunk.args));
        } else if (chunk.type === "tool_result") {
          setTurns((t) => attachToolResult(t, chunk.id, chunk.result, chunk.is_error));
        } else if (chunk.type === "done") {
          setUsage(chunk.usage);
          setTurns((t) => finalizeLastAssistant(t));
        } else if (chunk.type === "error") {
          setTurns((t) => [...t, { kind: "system", content: `error: ${chunk.error}` }]);
        }
      }
    } catch (err) {
      console.error(err);
      setTurns((t) => [...t, { kind: "system", content: `error: ${String(err)}` }]);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function onCancel() {
    if (abortRef.current) abortRef.current.abort();
    if (!isNew && conversationId) {
      await cancelConversation(conversationId);
    }
    router.push("/");
  }

  return (
    <main className="mx-auto flex h-screen max-w-3xl flex-col p-6">
      <header className="flex items-center justify-between border-b pb-4">
        <div>
          <Link href="/" className="text-sm text-neutral-500 hover:underline">
            ← All conversations
          </Link>
          <h1 className="text-lg font-semibold">
            {isNew ? "New conversation" : `Conversation ${id.slice(0, 8)}`}
          </h1>
          <p className="text-xs text-neutral-500">
            {provider} / {model}
          </p>
        </div>
        <button className="text-sm text-red-600 hover:underline" onClick={onCancel}>
          {busy ? "Stop" : "Cancel"}
        </button>
      </header>

      <section className="flex-1 space-y-3 overflow-y-auto py-4 text-sm">
        {turns.length === 0 && <p className="text-neutral-500">Say hi to get started…</p>}
        {turns.map((t, i) => (
          <TurnView key={i} turn={t} />
        ))}
        {usage && (
          <div className="text-right text-[10px] text-neutral-400">
            in {usage.input} · out {usage.output} · ${usage.cost_usd.toFixed(6)}
          </div>
        )}
      </section>

      <form
        className="flex items-end gap-2 border-t pt-4"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <textarea
          className="flex-1 resize-none rounded border p-2 text-sm"
          rows={2}
          placeholder="Message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
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
        <button
          type="submit"
          className="rounded bg-black px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
          disabled={busy || !input.trim()}
        >
          Send
        </button>
      </form>
    </main>
  );
}

// ── Turn-rendering helpers ─────────────────────────────────────────────────

function TurnView({ turn }: { turn: Turn }) {
  if (turn.kind === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-lg bg-blue-600 px-3 py-2 text-white">
          {turn.content}
        </div>
      </div>
    );
  }
  if (turn.kind === "system") {
    return (
      <div className="flex justify-center">
        <div className="rounded bg-red-50 px-3 py-1 text-xs text-red-700">{turn.content}</div>
      </div>
    );
  }
  if (turn.kind === "tool") {
    return (
      <div className="flex justify-start">
        <ToolCard turn={turn} />
      </div>
    );
  }
  // assistant
  return (
    <div className="flex flex-col items-start gap-1">
      {turn.thinking && <ThinkingBlock text={turn.thinking} />}
      <div className="max-w-[80%] whitespace-pre-wrap rounded-lg bg-neutral-100 px-3 py-2">
        {turn.content || (turn.pending ? "…" : "")}
      </div>
    </div>
  );
}

function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="max-w-[80%] rounded border border-amber-200 bg-amber-50 text-xs">
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

function ToolCard({
  turn,
}: {
  turn: Extract<Turn, { kind: "tool" }>;
}) {
  const [open, setOpen] = useState(false);
  const status = turn.result === undefined ? "running" : turn.isError ? "error" : "ok";
  const pillClass =
    status === "running"
      ? "bg-amber-100 text-amber-800"
      : status === "error"
      ? "bg-red-100 text-red-700"
      : "bg-green-100 text-green-800";
  return (
    <div className="w-full max-w-[80%] rounded-lg border border-neutral-200 bg-white text-xs shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-t-lg px-3 py-2 hover:bg-neutral-50"
      >
        <span className="flex items-center gap-2">
          <span className="text-neutral-400">{open ? "▾" : "▸"}</span>
          <span className="font-mono text-[11px] text-neutral-500">tool</span>
          <span className="font-medium">{turn.name}</span>
        </span>
        <span className={`rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${pillClass}`}>
          {status}
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-neutral-100 px-3 py-2">
          <div>
            <p className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">args</p>
            <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-neutral-50 p-2 text-[11px] text-neutral-700">
              {JSON.stringify(turn.args, null, 2)}
            </pre>
          </div>
          <div>
            <p className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">result</p>
            {turn.result === undefined ? (
              <p className="text-neutral-400">…running</p>
            ) : (
              <pre
                className={`overflow-x-auto whitespace-pre-wrap break-words rounded p-2 text-[11px] ${
                  turn.isError ? "bg-red-50 text-red-700" : "bg-neutral-50 text-neutral-700"
                }`}
              >
                {JSON.stringify(turn.result, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── reducers ──────────────────────────────────────────────────────────────

function appendText(turns: Turn[], delta: string): Turn[] {
  const last = turns[turns.length - 1];
  if (last && last.kind === "assistant" && last.pending) {
    const next = [...turns];
    next[next.length - 1] = { ...last, content: last.content + delta };
    return next;
  }
  // A tool round-trip closed the previous bubble — open a new one.
  return [...turns, { kind: "assistant", content: delta, pending: true }];
}

function appendThinking(turns: Turn[], delta: string): Turn[] {
  const last = turns[turns.length - 1];
  if (last && last.kind === "assistant" && last.pending) {
    const next = [...turns];
    next[next.length - 1] = { ...last, thinking: (last.thinking ?? "") + delta };
    return next;
  }
  // Open a new pending assistant turn to host the thinking stream.
  return [...turns, { kind: "assistant", content: "", thinking: delta, pending: true }];
}

function addToolCall(
  turns: Turn[],
  id: string,
  name: string,
  args: Record<string, unknown>,
): Turn[] {
  const next = [...turns];
  // Close the current pending assistant bubble (next text starts a new one).
  const lastIdx = next.length - 1;
  if (lastIdx >= 0 && next[lastIdx].kind === "assistant") {
    const a = next[lastIdx] as Extract<Turn, { kind: "assistant" }>;
    if (!a.content && !a.thinking) {
      // Drop empty placeholder so we don't render a stray "…".
      next.pop();
    } else {
      next[lastIdx] = { ...a, pending: false };
    }
  }
  next.push({ kind: "tool", id, name, args });
  return next;
}

function attachToolResult(
  turns: Turn[],
  id: string,
  result: unknown,
  isError: boolean,
): Turn[] {
  return turns.map((t) =>
    t.kind === "tool" && t.id === id ? { ...t, result, isError } : t,
  );
}

function finalizeLastAssistant(turns: Turn[]): Turn[] {
  const next = [...turns];
  const lastIdx = next.length - 1;
  if (lastIdx >= 0 && next[lastIdx].kind === "assistant") {
    const a = next[lastIdx] as Extract<Turn, { kind: "assistant" }>;
    next[lastIdx] = { ...a, pending: false };
  }
  return next;
}
