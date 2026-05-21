/**
 * Small presentational primitives shared by the eval browser components.
 */

import type { Heuristic } from "../../lib/eval-types";

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

export function ScoreChip({ value, label }: { value: number; label?: string }) {
  const cls =
    value < 0
      ? "bg-neutral-200 text-neutral-500"
      : value >= 5
      ? "bg-green-600 text-white"
      : value === 4
      ? "bg-lime-500 text-white"
      : value === 3
      ? "bg-amber-400 text-amber-950"
      : value === 2
      ? "bg-orange-500 text-white"
      : "bg-red-600 text-white";
  const display = value < 0 ? "—" : String(value);
  return (
    <span
      className={`inline-flex h-5 min-w-[2rem] items-center justify-center rounded px-1 font-mono text-[11px] ${cls}`}
      title={label}
    >
      {display}
    </span>
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
