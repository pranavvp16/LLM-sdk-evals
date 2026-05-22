"use client";

import { useState } from "react";

import type { Turn } from "../../lib/turns";

export function TurnView({ turn }: { turn: Turn }) {
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

export function ThinkingBlock({ text }: { text: string }) {
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

export function ToolCard({
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
