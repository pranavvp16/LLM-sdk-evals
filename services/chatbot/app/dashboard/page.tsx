"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getMetricsSummary,
  getRecentCalls,
  type MetricsSummary,
  type RecentCall,
} from "../../lib/api";

function fmtCost(n: number): string {
  if (n === 0) return "$0";
  if (n < 0.01) return `$${n.toFixed(6)}`;
  return `$${n.toFixed(4)}`;
}

function fmtMs(n: number): string {
  return `${Math.round(n)} ms`;
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [recent, setRecent] = useState<RecentCall[]>([]);
  const [hours, setHours] = useState(24);
  const [err, setErr] = useState<string | null>(null);

  async function refresh() {
    try {
      const [s, r] = await Promise.all([getMetricsSummary(hours), getRecentCalls(25)]);
      setSummary(s);
      setRecent(r);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 10_000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hours]);

  const t = summary?.totals;
  const errorRate = t && t.total_calls > 0 ? (t.error_calls / t.total_calls) * 100 : 0;

  return (
    <main className="mx-auto max-w-7xl p-4 sm:p-6">
      <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <Link href="/" className="text-sm text-neutral-500 hover:underline">
            ← Back
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">Inference Dashboard</h1>
        </div>
        <select
          className="w-full rounded border px-2 py-1.5 text-sm sm:w-auto"
          value={hours}
          onChange={(e) => setHours(Number(e.target.value))}
        >
          <option value={1}>last 1h</option>
          <option value={6}>last 6h</option>
          <option value={24}>last 24h</option>
          <option value={24 * 7}>last 7d</option>
        </select>
      </header>

      {err && <div className="mb-4 rounded bg-red-50 p-3 text-sm text-red-700">{err}</div>}

      <section className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Calls" value={t ? t.total_calls.toLocaleString() : "—"} />
        <Stat label="Error rate" value={t ? `${errorRate.toFixed(1)}%` : "—"} />
        <Stat label="p95 latency" value={t ? fmtMs(t.p95_ms) : "—"} />
        <Stat label="Total cost" value={t ? fmtCost(t.cost_usd) : "—"} />
        <Stat label="p50 latency" value={t ? fmtMs(t.p50_ms) : "—"} subtle />
        <Stat label="p99 latency" value={t ? fmtMs(t.p99_ms) : "—"} subtle />
        <Stat label="Input tokens" value={t ? t.input_tokens.toLocaleString() : "—"} subtle />
        <Stat label="Output tokens" value={t ? t.output_tokens.toLocaleString() : "—"} subtle />
      </section>

      <section className="mb-8">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
          By provider
        </h2>
        <div className="overflow-x-auto rounded border">
          <table className="w-full text-sm">
            <thead className="bg-neutral-50 text-left text-xs uppercase text-neutral-500">
              <tr>
                <th className="px-3 py-2">Provider</th>
                <th className="px-3 py-2 text-right">Calls</th>
                <th className="px-3 py-2 text-right">p95</th>
                <th className="px-3 py-2 text-right">Error %</th>
                <th className="px-3 py-2 text-right">Cost</th>
              </tr>
            </thead>
            <tbody>
              {summary?.by_provider.length ? (
                summary.by_provider.map((row) => (
                  <tr key={row.provider} className="border-t">
                    <td className="px-3 py-2 font-medium">{row.provider}</td>
                    <td className="px-3 py-2 text-right">{row.calls.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right">{fmtMs(row.p95_ms)}</td>
                    <td className="px-3 py-2 text-right">{(row.error_rate * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-right">{fmtCost(row.cost_usd)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="px-3 py-4 text-center text-neutral-500">
                    No data in this window.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
          Recent calls
        </h2>
        <div className="overflow-x-auto rounded border">
          <table className="w-full text-sm">
            <thead className="bg-neutral-50 text-left text-xs uppercase text-neutral-500">
              <tr>
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Provider / model</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2 text-right">Latency</th>
                <th className="px-3 py-2 text-right">Tokens</th>
                <th className="px-3 py-2 text-right">Cost</th>
                <th className="px-3 py-2">Preview</th>
              </tr>
            </thead>
            <tbody>
              {recent.length ? (
                recent.map((row, i) => (
                  <tr key={i} className="border-t align-top">
                    <td className="whitespace-nowrap px-3 py-2 text-xs text-neutral-500">
                      {new Date(row.started_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-xs">
                      <div className="font-medium">{row.provider}</div>
                      <div className="text-neutral-500">{row.model}</div>
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={
                          row.status === "success"
                            ? "rounded bg-green-100 px-2 py-0.5 text-xs text-green-800"
                            : "rounded bg-red-100 px-2 py-0.5 text-xs text-red-800"
                        }
                      >
                        {row.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-xs">{fmtMs(row.latency_ms)}</td>
                    <td className="px-3 py-2 text-right text-xs">{row.total_tokens}</td>
                    <td className="px-3 py-2 text-right text-xs">{fmtCost(row.cost_usd)}</td>
                    <td className="px-3 py-2 text-xs text-neutral-700">
                      {row.error_message ? (
                        <span className="text-red-600">{row.error_message.slice(0, 80)}</span>
                      ) : (
                        row.preview
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="px-3 py-4 text-center text-neutral-500">
                    No calls yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function Stat({ label, value, subtle }: { label: string; value: string; subtle?: boolean }) {
  return (
    <div className={`rounded border p-4 ${subtle ? "bg-neutral-50" : ""}`}>
      <p className="text-xs uppercase tracking-wide text-neutral-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
    </div>
  );
}
