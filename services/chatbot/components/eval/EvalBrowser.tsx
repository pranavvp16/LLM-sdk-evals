/**
 * Top-level eval results browser. Left column lists every prompt with a
 * compact score summary; right column shows the selected prompt's detail.
 *
 * Includes a "Run eval" button that kicks off a background job in the API
 * and polls /eval/runs/latest every 2 s until completion.
 */

"use client";

import { useEffect, useMemo, useState } from "react";

import {
  fetchLatestRun,
  fetchResults,
  reportPdfUrl,
  startEvalRun,
  type AgentRow,
  type ResultsPayload,
  type RunStatus,
  type StaticRow,
} from "../../lib/eval-types";
import { PromptDetail } from "./PromptDetail";
import { CategoryBadge, ScoreChip } from "./shared";

type EvalRow = StaticRow | AgentRow;

export function EvalBrowser() {
  const [payload, setPayload] = useState<ResultsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mobileShowDetail, setMobileShowDetail] = useState(false);
  const [filter, setFilter] = useState<"all" | "static" | "agent" | "failures" | "disagreement">("all");
  const [run, setRun] = useState<RunStatus | null>(null);
  const [runErr, setRunErr] = useState<string | null>(null);

  // initial load
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const p = await fetchResults();
        if (!cancelled) {
          setPayload(p);
          if (p) {
            const first = (p.static_results[0] ?? p.agent_results[0])?.prompt_id ?? null;
            setSelectedId(first);
          }
        }
      } catch (e) {
        if (!cancelled) setLoadErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // poll run status every 2s while a run is active
  useEffect(() => {
    if (!run || run.status === "done" || run.status === "failed") return;
    const t = setInterval(async () => {
      try {
        const next = await fetchLatestRun();
        if (next) setRun(next);
        if (next?.status === "done") {
          const fresh = await fetchResults();
          if (fresh) setPayload(fresh);
        }
      } catch (e) {
        console.error(e);
      }
    }, 2000);
    return () => clearInterval(t);
  }, [run?.run_id, run?.status]);

  const rows: EvalRow[] = useMemo(() => {
    if (!payload) return [];
    const all: EvalRow[] = [...payload.static_results, ...payload.agent_results];
    if (filter === "static") return all.filter((r) => r.kind === "static");
    if (filter === "agent") return all.filter((r) => r.kind === "agent");
    if (filter === "failures") return all.filter(rowIsFailure);
    if (filter === "disagreement") return all.filter(rowHasDisagreement);
    return all;
  }, [payload, filter]);

  const selected = rows.find((r) => r.prompt_id === selectedId) ?? rows[0];

  // Returning to the list when a filter clears all rows avoids a mobile dead-end.
  useEffect(() => {
    if (rows.length === 0) {
      setMobileShowDetail(false);
    }
  }, [rows.length]);

  async function onRun() {
    setRunErr(null);
    try {
      const accepted = await startEvalRun(["static", "agent"], null);
      setRun({
        run_id: accepted.run_id,
        status: "running",
        sections: ["static", "agent"],
        limit: null,
        started_at: new Date().toISOString(),
        completed_at: null,
        total_prompts: 50,
        completed_prompts: 0,
        current_prompt_id: null,
        error: null,
        partial_results: [],
      });
    } catch (e) {
      setRunErr(e instanceof Error ? e.message : String(e));
    }
  }

  if (loading) {
    return <div className="p-6 text-sm text-neutral-500">Loading eval results…</div>;
  }
  if (loadErr) {
    return <div className="p-6 text-sm text-red-700">Failed to load: {loadErr}</div>;
  }

  return (
    <div className="flex min-h-[60dvh] flex-col gap-3 lg:h-[calc(100vh-8rem)]">
      <Header
        payload={payload}
        run={run}
        runErr={runErr}
        filter={filter}
        setFilter={setFilter}
        onRun={onRun}
      />

      {!payload ? (
        <EmptyState onRun={onRun} runErr={runErr} run={run} />
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden rounded-lg border border-neutral-200 lg:grid-cols-[18rem_1fr] lg:grid-rows-[minmax(0,1fr)]">
          <div className={`min-h-0 ${mobileShowDetail ? "hidden lg:block" : "block"}`}>
            <PromptList
              rows={rows}
              selectedId={selected?.prompt_id ?? null}
              onSelect={(id) => {
                setSelectedId(id);
                setMobileShowDetail(true);
              }}
            />
          </div>
          <div className={`min-h-0 overflow-hidden bg-white ${mobileShowDetail ? "block" : "hidden lg:block"}`}>
            {selected ? (
              <>
                <MobileBackButton onClick={() => setMobileShowDetail(false)} />
                <PromptDetail row={selected} />
              </>
            ) : (
              <>
                {mobileShowDetail && (
                  <MobileBackButton onClick={() => setMobileShowDetail(false)} />
                )}
                <div className="p-6 text-sm text-neutral-500">No prompts in filter.</div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MobileBackButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="border-b border-neutral-200 px-4 py-2 text-left text-xs text-neutral-600 hover:bg-neutral-50 lg:hidden"
    >
      ← Back to prompts
    </button>
  );
}

function Header({
  payload,
  run,
  runErr,
  filter,
  setFilter,
  onRun,
}: {
  payload: ResultsPayload | null;
  run: RunStatus | null;
  runErr: string | null;
  filter: "all" | "static" | "agent" | "failures" | "disagreement";
  setFilter: (f: "all" | "static" | "agent" | "failures" | "disagreement") => void;
  onRun: () => void;
}) {
  const meta = payload?.metadata;
  const running = run && run.status === "running";
  return (
    <header className="rounded-lg border border-neutral-200 bg-white px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">Benchmark results</h2>
          {meta && (
            <p className="text-xs text-neutral-500">
              {new Date(meta.completed_at).toLocaleString()} · OSS{" "}
              <code>{meta.oss_provider}/{meta.oss_model}</code>
              {meta.guardrails_ablation && meta.guard_model && (
                <>
                  {" "}· OSS + <code>{meta.guard_model}</code>
                </>
              )}
              {" "}· Frontier <code>{meta.frontier_model}</code>
              {meta.judge_panel && (
                <>
                  <br />
                  judge panel: {meta.judge_panel.map((j) => <code key={j} className="mr-1">{j.split("/").slice(-1)[0]}</code>)}
                  {typeof meta.judge_total_cost_usd === "number" && (
                    <> · panel cost ${meta.judge_total_cost_usd.toFixed(4)}</>
                  )}
                </>
              )}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 overflow-x-auto">
          {(["all", "static", "agent", "failures", "disagreement"] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`rounded px-2 py-1 text-xs ${
                filter === f
                  ? "bg-neutral-900 text-white"
                  : "border border-neutral-200 text-neutral-700 hover:bg-neutral-50"
              }`}
            >
              {f}
            </button>
          ))}
          <a
            href={reportPdfUrl()}
            target="_blank"
            rel="noreferrer"
            className="rounded border border-neutral-200 px-3 py-1 text-xs text-neutral-700 hover:bg-neutral-50"
          >
            ⬇ PDF
          </a>
          <button
            type="button"
            onClick={onRun}
            disabled={Boolean(running)}
            className="rounded bg-neutral-900 px-3 py-1 text-xs font-medium text-white hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {running ? "Running…" : payload ? "Re-run eval" : "Run eval"}
          </button>
        </div>
      </div>
      {run && (
        <div className="mt-2 flex items-center gap-2 text-xs">
          <span
            className={`rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
              run.status === "done"
                ? "bg-green-100 text-green-800"
                : run.status === "failed"
                ? "bg-red-100 text-red-700"
                : "bg-amber-100 text-amber-800"
            }`}
          >
            {run.status}
          </span>
          <span className="text-neutral-600">
            {run.completed_prompts}/{run.total_prompts}
            {run.current_prompt_id && ` · current: ${run.current_prompt_id}`}
          </span>
          {run.error && <span className="text-red-700">err: {run.error}</span>}
        </div>
      )}
      {runErr && <p className="mt-1 text-xs text-red-700">{runErr}</p>}
    </header>
  );
}

function PromptList({
  rows,
  selectedId,
  onSelect,
}: {
  rows: EvalRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <nav className="overflow-y-auto border-r border-neutral-200 bg-neutral-50">
      {rows.length === 0 ? (
        <p className="p-4 text-xs text-neutral-500">No prompts match this filter.</p>
      ) : (
        <ul>
          {rows.map((r) => {
            const active = r.prompt_id === selectedId;
            return (
              <li key={r.prompt_id}>
                <button
                  type="button"
                  onClick={() => onSelect(r.prompt_id)}
                  className={`flex w-full items-center justify-between gap-2 border-b border-neutral-200 px-3 py-2 text-left text-xs ${
                    active ? "bg-white" : "hover:bg-white"
                  }`}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <CategoryBadge category={r.category} />
                    <span className="truncate font-mono">{r.prompt_id}</span>
                  </span>
                  <span className="hidden shrink-0 sm:flex">
                    <RowScoreSummary row={r} />
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </nav>
  );
}

function RowScoreSummary({ row }: { row: EvalRow }) {
  const sep = <span className="mx-1 text-neutral-300">|</span>;
  if (row.kind === "static") {
    return (
      <span className="flex items-center gap-0.5">
        <ScoreChip value={row.oss.scores.hallucination.aggregated_score} />
        <ScoreChip value={row.oss.scores.bias.aggregated_score} />
        <ScoreChip value={row.oss.scores.safety.aggregated_score} />
        <ScoreChip value={row.oss.scores.role_violation.aggregated_score} />
        {row.oss_guarded && (
          <>
            {sep}
            <ScoreChip value={row.oss_guarded.scores.hallucination.aggregated_score} />
            <ScoreChip value={row.oss_guarded.scores.bias.aggregated_score} />
            <ScoreChip value={row.oss_guarded.scores.safety.aggregated_score} />
            <ScoreChip value={row.oss_guarded.scores.role_violation.aggregated_score} />
          </>
        )}
        {sep}
        <ScoreChip value={row.frontier.scores.hallucination.aggregated_score} />
        <ScoreChip value={row.frontier.scores.bias.aggregated_score} />
        <ScoreChip value={row.frontier.scores.safety.aggregated_score} />
        <ScoreChip value={row.frontier.scores.role_violation.aggregated_score} />
      </span>
    );
  }
  return (
    <span className="flex items-center gap-0.5">
      <ScoreChip value={row.oss.scores.tool_selection.aggregated_score} />
      <ScoreChip value={row.oss.scores.argument_correctness.aggregated_score} />
      <ScoreChip value={row.oss.scores.task_completion.aggregated_score} />
      {row.oss_guarded && (
        <>
          {sep}
          <ScoreChip value={row.oss_guarded.scores.tool_selection.aggregated_score} />
          <ScoreChip value={row.oss_guarded.scores.argument_correctness.aggregated_score} />
          <ScoreChip value={row.oss_guarded.scores.task_completion.aggregated_score} />
        </>
      )}
      {sep}
      <ScoreChip value={row.frontier.scores.tool_selection.aggregated_score} />
      <ScoreChip value={row.frontier.scores.argument_correctness.aggregated_score} />
      <ScoreChip value={row.frontier.scores.task_completion.aggregated_score} />
    </span>
  );
}

function EmptyState({
  onRun,
  runErr,
  run,
}: {
  onRun: () => void;
  runErr: string | null;
  run: RunStatus | null;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-neutral-300 bg-white p-12 text-center">
      <p className="text-sm text-neutral-700">No eval has been run yet.</p>
      <p className="max-w-md text-xs text-neutral-500">
        Click <strong>Run eval</strong> to kick off a background job, or run{" "}
        <code className="rounded bg-neutral-100 px-1">python eval/run_eval.py</code> from the CLI.
        A full run is ~50 prompts × 2 models × judge — roughly $1 and ~5 minutes.
      </p>
      <button
        type="button"
        onClick={onRun}
        disabled={Boolean(run && run.status === "running")}
        className="rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-800 disabled:opacity-60"
      >
        {run && run.status === "running" ? "Running…" : "Run eval"}
      </button>
      {runErr && <p className="text-xs text-red-700">{runErr}</p>}
    </div>
  );
}

function _agg(score: { aggregated_score: number | null }): number {
  return typeof score.aggregated_score === "number" ? score.aggregated_score : 99;
}

function rowIsFailure(row: EvalRow): boolean {
  if (row.kind === "static") {
    const o = row.oss.scores;
    const f = row.frontier.scores;
    const minOss = Math.min(_agg(o.hallucination), _agg(o.bias), _agg(o.safety), _agg(o.role_violation));
    const minFr = Math.min(_agg(f.hallucination), _agg(f.bias), _agg(f.safety), _agg(f.role_violation));
    return (
      minOss <= 2 ||
      minFr <= 2 ||
      !row.oss.heuristic.overall_pass ||
      !row.frontier.heuristic.overall_pass
    );
  }
  const o = row.oss.scores;
  const f = row.frontier.scores;
  const minOss = Math.min(
    _agg(o.tool_selection),
    _agg(o.argument_correctness),
    _agg(o.task_completion),
    _agg(o.output_grounding),
    _agg(o.safety_with_tools),
  );
  const minFr = Math.min(
    _agg(f.tool_selection),
    _agg(f.argument_correctness),
    _agg(f.task_completion),
    _agg(f.output_grounding),
    _agg(f.safety_with_tools),
  );
  return minOss <= 2 || minFr <= 2;
}

function rowHasDisagreement(row: EvalRow): boolean {
  function check(scores: Record<string, unknown>): boolean {
    for (const v of Object.values(scores)) {
      const panel = v as { agreement?: { binary_unanimous: boolean; geval_stdev: number | null } };
      if (!panel?.agreement) continue;
      if (!panel.agreement.binary_unanimous) return true;
      if (panel.agreement.geval_stdev !== null && panel.agreement.geval_stdev > 1.0) return true;
    }
    return false;
  }
  return check(row.oss.scores as unknown as Record<string, unknown>)
    || check(row.frontier.scores as unknown as Record<string, unknown>)
    || (row.oss_guarded ? check(row.oss_guarded.scores as unknown as Record<string, unknown>) : false);
}
