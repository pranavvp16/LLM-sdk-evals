/**
 * Small presentational primitives shared by the eval browser components.
 */

import type { Heuristic, JudgeOpinion, PanelScore } from "../../lib/eval-types";

export const CATEGORY_BADGE: Record<string, { letter: string; cls: string }> = {
  factual: { letter: "F", cls: "bg-blue-100 text-blue-800" },
  adversarial: { letter: "A", cls: "bg-red-100 text-red-700" },
  bias: { letter: "B", cls: "bg-purple-100 text-purple-800" },
  single_tool: { letter: "T", cls: "bg-green-100 text-green-800" },
  multi_tool: { letter: "M", cls: "bg-emerald-100 text-emerald-800" },
  no_tool: { letter: "N", cls: "bg-neutral-200 text-neutral-700" },
  ambiguous: { letter: "?", cls: "bg-amber-100 text-amber-800" },
  adversarial_tool: { letter: "X", cls: "bg-rose-100 text-rose-700" },
};

export function CategoryBadge({ category }: { category: string }) {
  const b = CATEGORY_BADGE[category] ?? { letter: "?", cls: "bg-neutral-100 text-neutral-700" };
  return (
    <span
      className={`inline-flex h-5 w-5 items-center justify-center rounded text-[10px] font-bold ${b.cls}`}
      title={category}
    >
      {b.letter}
    </span>
  );
}

export function ScoreChip({ value, label }: { value: number | null | undefined; label?: string }) {
  const v = typeof value === "number" ? value : -1;
  const cls =
    v < 0
      ? "bg-neutral-200 text-neutral-500"
      : v >= 4.5
      ? "bg-green-600 text-white"
      : v >= 3.5
      ? "bg-lime-500 text-white"
      : v >= 2.5
      ? "bg-amber-400 text-amber-950"
      : v >= 1.5
      ? "bg-orange-500 text-white"
      : "bg-red-600 text-white";
  const display = v < 0 ? "—" : v.toFixed(1);
  return (
    <span
      className={`inline-flex h-5 min-w-[2rem] items-center justify-center rounded px-1 font-mono text-[11px] ${cls}`}
      title={label}
    >
      {display}
    </span>
  );
}

/** Agreement traffic light next to an aggregated score chip. */
export function AgreementChip({ panel }: { panel: PanelScore }) {
  const agree = panel.agreement;
  const unanimous = agree.binary_unanimous;
  const stdev = agree.geval_stdev ?? 0;
  const tone =
    unanimous && stdev < 0.8
      ? { cls: "bg-emerald-100 text-emerald-800", label: "agree" }
      : !unanimous && stdev > 1.5
      ? { cls: "bg-red-100 text-red-700", label: "split" }
      : { cls: "bg-amber-100 text-amber-800", label: "mixed" };
  const tooltip =
    `binary_unanimous=${unanimous}` +
    (agree.kappa_avg !== null ? ` · κ=${agree.kappa_avg.toFixed(2)}` : "") +
    (agree.geval_stdev !== null ? ` · σ=${agree.geval_stdev.toFixed(2)}` : "");
  return (
    <span
      title={tooltip}
      className={`inline-flex h-4 items-center rounded px-1 text-[9px] uppercase tracking-wide ${tone.cls}`}
    >
      {tone.label}
    </span>
  );
}

/** Stack of three per-judge mini-bars under an axis cell. */
export function JudgeStack({ judges }: { judges: JudgeOpinion[] }) {
  return (
    <ul className="space-y-1">
      {judges.map((j) => {
        const short = j.judge_id.split("/").slice(-1)[0];
        const score = typeof j.score === "number" ? j.score.toFixed(1) : "—";
        return (
          <li key={j.judge_id} className="rounded border border-neutral-200 bg-neutral-50 px-2 py-1 text-[11px]">
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-[10px] text-neutral-600">{short}</span>
              <span className="flex items-center gap-1">
                {j.status === "judge_failed" && (
                  <span className="rounded bg-red-100 px-1 text-[9px] uppercase text-red-700">fail</span>
                )}
                <ScoreChip value={typeof j.score === "number" ? j.score : null} />
              </span>
            </div>
            {j.verdict_path.length > 0 && (
              <p className="mt-1 truncate font-mono text-[9px] text-neutral-500" title={j.verdict_path.join(" → ")}>
                {j.verdict_path.join(" → ")}
              </p>
            )}
            {j.node_outputs.some((n) => n.reason) && (
              <p className="mt-1 text-[10px] leading-snug text-neutral-700">
                {j.node_outputs.find((n) => n.reason)?.reason}
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function HeuristicChecklist({ h }: { h: Heuristic }) {
  const checks: { label: string; pass: boolean; extra?: string }[] = [
    { label: "# Summary", pass: h.h1_present },
    { label: "## Details", pass: h.h2_present },
    { label: "### Next step", pass: h.h3_present },
    { label: "— Ollie signature", pass: h.persona_signature },
    {
      label: "no banned phrases",
      pass: h.banned_phrases_found.length === 0,
      extra: h.banned_phrases_found.length
        ? `found: ${h.banned_phrases_found.join(", ")}`
        : undefined,
    },
    { label: "length in bounds", pass: h.length_ok },
  ];
  return (
    <ul className="space-y-1 text-xs">
      {checks.map((c) => (
        <li key={c.label} className="flex items-start gap-2">
          <span className={c.pass ? "text-green-600" : "text-red-600"}>{c.pass ? "✓" : "✗"}</span>
          <span className="flex-1">
            <span className={c.pass ? "text-neutral-700" : "text-red-700 font-medium"}>
              {c.label}
            </span>
            {c.extra && <span className="ml-2 text-neutral-500">({c.extra})</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function formatCost(usd: number): string {
  if (usd < 0.001) return `$${(usd * 1000).toFixed(3)}m`;
  return `$${usd.toFixed(4)}`;
}

export function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}
