"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { TurnView } from "../../../components/chat/TurnView";
import {
  cancelConversation,
  getConversation,
  type Message,
  streamChat,
} from "../../../lib/api";
import {
  addToolCall,
  appendText,
  appendThinking,
  attachToolResult,
  finalizeLastAssistant,
  type Turn,
} from "../../../lib/turns";

interface PageProps {
  params: { id: string };
}

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
          thinking,
        },
        abortRef.current.signal,
      )) {
        if (chunk.type === "meta") {
          if (chunk.conversation_id && chunk.conversation_id !== conversationId) {
            setConversationId(chunk.conversation_id);
            if (isNew && typeof window !== "undefined") {
              // Silent URL swap — router.replace would re-fire the history-loading
              // useEffect mid-stream and overwrite the live turns.
              const qs = new URLSearchParams({ provider, model });
              window.history.replaceState(
                null,
                "",
                `/chat/${chunk.conversation_id}?${qs.toString()}`,
              );
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
    // history.replaceState (used after the first turn on /chat/new) doesn't
    // update params.id, so `isNew` stays true even once we have a real
    // conversationId. Gate on conversationId alone to keep server-side
    // cancel working for conversations started from /chat/new.
    if (conversationId) {
      await cancelConversation(conversationId);
    }
    router.push("/");
  }

  return (
    <main className="mx-auto flex h-dvh max-h-dvh max-w-3xl flex-col p-4 pb-safe sm:p-6">
      <header className="flex flex-col items-start gap-2 border-b pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <Link href="/" className="text-sm text-neutral-500 hover:underline">
            ← All conversations
          </Link>
          <h1 className="text-lg font-semibold">
            {conversationId
              ? `Conversation ${conversationId.slice(0, 8)}`
              : "New conversation"}
          </h1>
          <p className="text-xs text-neutral-500">
            {provider} / {model}
          </p>
        </div>
        <button className="min-h-[44px] text-sm text-red-600 hover:underline sm:min-h-0" onClick={onCancel}>
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
        className="flex flex-col gap-2 border-t pt-4 sm:flex-row sm:items-end"
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
          <button
            type="submit"
            className="min-h-[44px] flex-1 rounded bg-black px-3 py-2 text-sm font-medium text-white disabled:opacity-40 sm:flex-none sm:min-h-0"
            disabled={busy || !input.trim()}
          >
            Send
          </button>
        </div>
      </form>
    </main>
  );
}

