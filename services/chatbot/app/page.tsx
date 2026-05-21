"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  cancelConversation,
  type Conversation,
  listConversations,
  resumeConversation,
} from "../lib/api";

const PROVIDERS: { provider: string; model: string; label: string }[] = [
  { provider: "anthropic", model: "claude-sonnet-4-6", label: "Sonnet 4.6 (frontier)" },
  { provider: "anthropic", model: "claude-haiku-4-5-20251001", label: "Haiku 4.5 (cheap)" },
  { provider: "anthropic", model: "claude-opus-4-7", label: "Opus 4.7" },
  { provider: "vllm", model: "qwen2.5-0.5b-instruct", label: "Qwen 0.5B (vLLM)" },
  { provider: "opencode", model: "kimi-k2.5", label: "Kimi K2.5 (OpenCode Zen)" },
  { provider: "opencode-go", model: "glm-5", label: "GLM-5 (OpenCode Go)" },
  { provider: "huggingface", model: "qwen2.5-0.5b-instruct", label: "Qwen 0.5B (HF fallback)" },
];

export default function HomePage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [picker, setPicker] = useState(PROVIDERS[0]);

  async function refresh() {
    try {
      const list = await listConversations();
      setConversations(list);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function onCancel(id: string) {
    await cancelConversation(id);
    await refresh();
  }

  async function onResume(id: string) {
    await resumeConversation(id);
    await refresh();
  }

  return (
    <main className="mx-auto max-w-3xl p-6">
      <header className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Ollive Chat</h1>
        <div className="flex items-center gap-2">
          <select
            className="rounded border px-2 py-1.5 text-sm"
            value={`${picker.provider}/${picker.model}`}
            onChange={(e) => {
              const found = PROVIDERS.find(
                (p) => `${p.provider}/${p.model}` === e.target.value,
              );
              if (found) setPicker(found);
            }}
          >
            {PROVIDERS.map((p) => (
              <option key={`${p.provider}/${p.model}`} value={`${p.provider}/${p.model}`}>
                {p.label}
              </option>
            ))}
          </select>
          <Link
            href={`/chat/new?provider=${picker.provider}&model=${picker.model}`}
            className="rounded bg-black px-3 py-1.5 text-sm font-medium text-white hover:bg-neutral-800"
          >
            New conversation
          </Link>
        </div>
      </header>

      <p className="mb-4 flex gap-4 text-sm">
        <Link className="underline" href="/dashboard">
          Open dashboard →
        </Link>
        <Link className="underline" href="/compare">
          Side-by-side compare →
        </Link>
      </p>

      {loading ? (
        <p className="text-sm text-neutral-500">Loading…</p>
      ) : conversations.length === 0 ? (
        <p className="text-sm text-neutral-500">
          No conversations yet. Click <strong>New conversation</strong> to start.
        </p>
      ) : (
        <ul className="space-y-2">
          {conversations.map((c) => (
            <li
              key={c.id}
              className="flex items-center justify-between rounded border p-3 hover:bg-neutral-50"
            >
              <Link href={`/chat/${c.id}`} className="flex-1">
                <div className="text-sm font-medium">{c.title || "(untitled)"}</div>
                <div className="text-xs text-neutral-500">
                  {c.provider} / {c.model} ·{" "}
                  <span
                    className={
                      c.status === "active"
                        ? "text-green-600"
                        : c.status === "cancelled"
                        ? "text-red-600"
                        : ""
                    }
                  >
                    {c.status}
                  </span>{" "}
                  · {new Date(c.updated_at).toLocaleString()}
                </div>
              </Link>
              {c.status === "cancelled" ? (
                <button
                  className="ml-2 text-xs text-blue-600 hover:underline"
                  onClick={() => onResume(c.id)}
                >
                  Resume
                </button>
              ) : (
                <button
                  className="ml-2 text-xs text-red-600 hover:underline"
                  onClick={() => onCancel(c.id)}
                >
                  Cancel
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
