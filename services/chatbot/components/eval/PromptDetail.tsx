/**
 * Right-pane detail view for a single eval row. Renders OSS and Frontier
 * side-by-side with response/trajectory, heuristic checklist, judge scores,
 * latency and cost.
 */

"use client";

import { useState } from "react";

import type {
  AgentRow,
  AgentScores,
  StaticRow,
  StaticScores,
  Trajectory,
  StaticSide,
} from "../../lib/eval-types";
import { CategoryBadge, HeuristicChecklist, ScoreChip, formatCost, formatLatency } from "./shared";
import { TrajectoryView } from "./TrajectoryView";

type EvalRow = StaticRow | AgentRow;

const STATIC_AXES: { key: keyof StaticScores; label: string }[] = [
  { key: "hallucination", label: "Halluc" },
  { key: "bias", label: "Bias" },
  { key: "safety", label: "Safety" },
];

const AGENT_AXES: { key: keyof AgentScores; label: string }[] = [
  { key: "tool_selection", label: "Tool sel" },
  { key: "argument_correctness", label: "Args" },
  { key: "task_completion", label: "Complete" },
  { key: "output_grounding", label: "Ground" },
  { key: "safety_with_tools", label: "Safety" },
];

export function PromptDetail({ row }: { row: EvalRow }) {
  return (
    <section className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-neutral-200 px-4 py-4 sm:px-5">
        <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-500">
          <CategoryBadge category={row.category} />
          <span className="font-mono">{row.prompt_id}</span>
          <span>·</span>
          <span>{row.category}</span>
          <span>·</span>
          <span>{row.kind}</span>
        </div>
        <p className="mt-2 whitespace-pre-wrap text-sm text-neutral-900">{row.prompt}</p>
        <details className="mt-2 text-xs text-neutral-500">
          <summary className="cursor-pointer">expected behaviour</summary>
          <p className="mt-1 whitespace-pre-wrap pl-4 text-neutral-700">
            {row.expected_behavior}
          </p>
          {row.kind === "agent" && row.expected_tools && row.expected_tools.length > 0 && (
            <p className="mt-1 pl-4 text-neutral-700">
              expected tools: <code>{row.expected_tools.join(", ")}</code>
            </p>
          )}
        </details>
      </header>

      <div className="grid flex-1 grid-cols-1 divide-y divide-neutral-200 overflow-auto lg:grid-cols-2 lg:divide-x lg:divide-y-0">
        <SidePanel row={row} side="oss" />
        <SidePanel row={row} side="frontier" />
      </div>
    </section>
  );
}

function SidePanel({ row, side }: { row: EvalRow; side: "oss" | "frontier" }) {
  const data = row[side];
  const isAgent = row.kind === "agent";
  const provider = isAgent ? (data as Trajectory).provider : side === "oss" ? "OSS" : "Frontier";
  const model = isAgent ? (data as Trajectory).model : "";
  return (
    <div className="flex flex-col overflow-auto">
      <header className="sticky top-0 z-10 border-b border-neutral-200 bg-white px-4 py-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-neutral-500">
            {side === "oss" ? "OSS" : "Frontier"}
            {model && <span className="ml-2 font-mono text-[10px] text-neutral-400">{model}</span>}
          </span>
          <span className="text-[11px] text-neutral-400">
            {formatLatency(data.latency_ms)} · {formatCost(data.cost_usd)}
          </span>
        </div>
        <div className="mt-1 flex flex-wrap gap-2">
          {isAgent
            ? AGENT_AXES.map((a) => (
                <span key={a.key} className="flex items-center gap-1 text-[11px] text-neutral-600">
                  {a.label}{" "}
                  <ScoreChip
                    value={(data as Trajectory).scores[a.key] as number}
                    label={a.label}
                  />
                </span>
              ))
            : STATIC_AXES.map((a) => (
                <span key={a.key} className="flex items-center gap-1 text-[11px] text-neutral-600">
                  {a.label}{" "}
                  <ScoreChip
                    value={(data as StaticSide).scores[a.key] as number}
                    label={a.label}
                  />
                </span>
              ))}
        </div>
      </header>

      <div className="space-y-4 px-4 py-3">
        {data.thinking && <ThinkingBlock text={data.thinking} />}
        {isAgent ? (
          <TrajectoryView
            steps={(data as Trajectory).tool_calls}
            finalText={(data as Trajectory).final_text}
            hitMaxHops={(data as Trajectory).hit_max_hops}
            status={(data as Trajectory).status}
          />
        ) : (
          <ResponseBlock text={(data as StaticSide).response} status={(data as StaticSide).status} />
        )}

        <section>
          <h4 className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">
            Heuristic compliance
          </h4>
          <HeuristicChecklist h={data.heuristic} />
        </section>

        <section>
          <h4 className="mb-1 text-[10px] uppercase tracking-wide text-neutral-400">
            Judge rationale
          </h4>
          <p className="whitespace-pre-wrap text-xs text-neutral-700">
            {data.scores.rationale || "—"}
          </p>
        </section>

        {isAgent && (
          <section className="text-[11px] text-neutral-500">
            hops: {(data as Trajectory).hops_used} · status: {(data as Trajectory).status}
            {(data as Trajectory).error && (
              <span className="ml-2 text-red-600">err: {(data as Trajectory).error}</span>
            )}
          </section>
        )}
      </div>
    </div>
  );
}

function ResponseBlock({ text, status }: { text: string; status: string }) {
  if (status === "error") {
    return <p className="text-xs text-red-700">model errored — no response</p>;
  }
  if (!text) {
    return <p className="text-xs italic text-neutral-500">empty response</p>;
  }
  return (
    <pre className="whitespace-pre-wrap break-words rounded border border-neutral-200 bg-neutral-50 p-3 text-xs text-neutral-800">
      {text}
    </pre>
  );
}

function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded border border-amber-200 bg-amber-50 text-xs">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex min-h-[44px] w-full items-center gap-2 px-2 py-1 hover:bg-amber-100"
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
