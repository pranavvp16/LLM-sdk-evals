/**
 * Right-pane detail view for a single eval row. Renders OSS and Frontier
 * side-by-side with response/trajectory, heuristic checklist, the
 * 3-judge DAG panel per axis (aggregated chip + expandable stack), and
 * cost / latency.
 */

"use client";

import { useState } from "react";

import type {
  AgentRow,
  AgentScores,
  PanelScore,
  StaticRow,
  StaticScores,
  Trajectory,
  StaticSide,
} from "../../lib/eval-types";
import {
  AgreementChip,
  AxisNAChip,
  CategoryBadge,
  HeuristicChecklist,
  JudgeStack,
  PanelStatusBadge,
  ScoreChip,
  formatCost,
  formatLatency,
} from "./shared";
import { TrajectoryView } from "./TrajectoryView";

type EvalRow = StaticRow | AgentRow;

const STATIC_AXES: { key: keyof StaticScores; label: string }[] = [
  { key: "hallucination",  label: "Halluc" },
  { key: "bias",           label: "Bias" },
  { key: "safety",         label: "Safety" },
  { key: "role_violation", label: "Role" },
];

const AGENT_AXES: { key: keyof AgentScores; label: string }[] = [
  { key: "tool_selection",       label: "Tool sel" },
  { key: "argument_correctness", label: "Args" },
  { key: "task_completion",      label: "Complete" },
  { key: "output_grounding",     label: "Ground" },
  { key: "safety_with_tools",    label: "Safety" },
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

      <div
        className={`grid min-h-0 flex-1 grid-cols-1 divide-y divide-neutral-200 overflow-auto lg:divide-x lg:divide-y-0 lg:overflow-hidden lg:grid-rows-[1fr] ${
          row.oss_guarded ? "lg:grid-cols-3" : "lg:grid-cols-2"
        }`}
      >
        <SidePanel row={row} side="oss" />
        {row.oss_guarded && <SidePanel row={row} side="oss_guarded" />}
        <SidePanel row={row} side="frontier" />
      </div>
    </section>
  );
}

type SideKey = "oss" | "oss_guarded" | "frontier";

const SIDE_LABEL: Record<SideKey, string> = {
  oss: "OSS",
  oss_guarded: "OSS + Llama Guard",
  frontier: "Frontier",
};

function SidePanel({ row, side }: { row: EvalRow; side: SideKey }) {
  const data = side === "oss_guarded" ? row.oss_guarded : row[side];
  if (!data) return null;
  const isAgent = row.kind === "agent";
  const model = isAgent ? (data as Trajectory).model : "";
  const axes = isAgent ? AGENT_AXES : STATIC_AXES;
  const scores = data.scores as StaticScores | AgentScores;

  return (
    <div className="flex min-h-0 min-w-0 flex-col overflow-auto">
      <header className="sticky top-0 z-10 border-b border-neutral-200 bg-white px-4 py-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-neutral-500">
            {SIDE_LABEL[side]}
            {model && <span className="ml-2 font-mono text-[10px] text-neutral-400">{model}</span>}
          </span>
          <span className="text-[11px] text-neutral-400">
            {formatLatency(data.latency_ms)} · {formatCost(data.cost_usd)}
          </span>
        </div>
        <div className="mt-1 flex flex-wrap gap-2">
          {axes.map((a) => {
            const panel = scores[a.key as keyof typeof scores] as PanelScore | undefined;
            if (!panel) return null;
            return (
              <span key={String(a.key)} className="flex items-center gap-1 text-[11px] text-neutral-600">
                {a.label}{" "}
                <ScoreChip value={panel.aggregated_score} label={a.label} />
                <AgreementChip panel={panel} />
                <PanelStatusBadge panel={panel} />
                <AxisNAChip panel={panel} />
              </span>
            );
          })}
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
            Judge panel — per axis
          </h4>
          <div className="space-y-2">
            {axes.map((a) => {
              const panel = scores[a.key as keyof typeof scores] as PanelScore | undefined;
              if (!panel) return null;
              return <AxisPanelBlock key={String(a.key)} axisLabel={a.label} panel={panel} />;
            })}
          </div>
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

function AxisPanelBlock({ axisLabel, panel }: { axisLabel: string; panel: PanelScore }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded border border-neutral-200">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 px-2 py-1 text-left text-xs hover:bg-neutral-50"
      >
        <span className="flex items-center gap-2">
          <span className="text-neutral-500">{open ? "▾" : "▸"}</span>
          <span className="font-medium text-neutral-700">{axisLabel}</span>
          <ScoreChip value={panel.aggregated_score} />
          <AgreementChip panel={panel} />
          <PanelStatusBadge panel={panel} />
          <AxisNAChip panel={panel} />
          {panel.category_majority && panel.category_majority !== "none" && (
            <span className="rounded bg-purple-100 px-1 text-[9px] uppercase tracking-wide text-purple-800">
              {panel.category_majority}
            </span>
          )}
          {panel.toxicity_flagged && (
            <span className="rounded bg-red-100 px-1 text-[9px] uppercase tracking-wide text-red-700">toxic</span>
          )}
          {panel.violations && panel.violations.length > 0 && (
            <span className="rounded bg-rose-100 px-1 text-[9px] uppercase tracking-wide text-rose-700">
              {panel.violations.length} violation{panel.violations.length === 1 ? "" : "s"}
            </span>
          )}
        </span>
        <span className="text-[10px] text-neutral-400">{panel.judges.length} judges</span>
      </button>
      {open && (
        <div className="border-t border-neutral-200 p-2">
          {panel.verdict_path_majority.length > 0 && (
            <p className="mb-2 text-[10px] text-neutral-600">
              <span className="font-medium">Consensus DAG path:</span>{" "}
              <code className="break-all">{panel.verdict_path_majority.join(" → ")}</code>
            </p>
          )}
          {panel.violations && panel.violations.length > 0 && (
            <p className="mb-2 text-[10px] text-neutral-600">
              <span className="font-medium">Triggered categories:</span>{" "}
              <code>{panel.violations.join(", ")}</code>
            </p>
          )}
          {panel.llamaguard_pre_signal && (
            <p className="mb-2 text-[10px] text-neutral-600">
              <span className="font-medium">Llama Guard pre-signal:</span>{" "}
              <code>{panel.llamaguard_pre_signal}</code>
            </p>
          )}
          <JudgeStack judges={panel.judges} />
        </div>
      )}
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
