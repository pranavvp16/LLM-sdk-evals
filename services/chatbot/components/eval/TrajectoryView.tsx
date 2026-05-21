/**
 * Renders an agent trajectory as a stack of collapsible tool-call cards.
 * Mirrors the ToolCard visual idiom from app/chat/[id]/page.tsx.
 */

"use client";

import { useState } from "react";

import type { ToolStep } from "../../lib/eval-types";
import { formatLatency } from "./shared";

export function TrajectoryView({
  steps,
  finalText,
  hitMaxHops,
  status,
}: {
  steps: ToolStep[];
  finalText: string;
  hitMaxHops: boolean;
  status: string;
}) {
  if (status === "model_error") {
    return (
      <p className="text-xs text-red-700">model errored before producing a trajectory</p>
    );
  }
  if (steps.length === 0 && !finalText) {
    return <p className="text-xs italic text-neutral-500">empty trajectory</p>;
  }
  return (
    <div className="space-y-2">
      {steps.map((s, i) => (
        <ToolStepCard key={`${s.call_id}-${i}`} step={s} />
      ))}
      {hitMaxHops && (
        <div className="rounded border border-amber-300 bg-amber-50 px-2 py-1 text-[11px] text-amber-900">
          hit MAX_TOOL_HOPS without a final answer
        </div>
      )}
      {finalText && (
        <div className="rounded border border-neutral-200 bg-neutral-50 p-2">
          <p className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">
            final text
          </p>
          <pre className="whitespace-pre-wrap break-words text-[11px] text-neutral-800">
            {finalText}
          </pre>
        </div>
      )}
    </div>
  );
}

function ToolStepCard({ step }: { step: ToolStep }) {
  const [open, setOpen] = useState(false);
  const pillClass = step.is_error ? "bg-red-100 text-red-700" : "bg-green-100 text-green-800";
  const statusLabel = step.is_error ? (step.error ?? "error") : "ok";
  return (
    <div className="rounded-lg border border-neutral-200 bg-white text-xs shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-t-lg px-3 py-1.5 hover:bg-neutral-50"
      >
        <span className="flex items-center gap-2">
          <span className="text-neutral-400">{open ? "▾" : "▸"}</span>
          <span className="font-mono text-[10px] text-neutral-500">hop {step.hop}</span>
          <span className="font-medium">{step.name}</span>
          <span className="text-[10px] text-neutral-400">{formatLatency(step.latency_ms)}</span>
        </span>
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${pillClass}`}
        >
          {statusLabel}
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-neutral-100 px-3 py-2">
          <div>
            <p className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">args</p>
            <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-neutral-50 p-2 text-[11px] text-neutral-700">
              {JSON.stringify(step.arguments, null, 2)}
            </pre>
          </div>
          <div>
            <p className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">result</p>
            <pre
              className={`overflow-x-auto whitespace-pre-wrap break-words rounded p-2 text-[11px] ${
                step.is_error ? "bg-red-50 text-red-700" : "bg-neutral-50 text-neutral-700"
              }`}
            >
              {JSON.stringify(step.result, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
