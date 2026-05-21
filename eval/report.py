"""Generate ``docs/eval_report.pdf`` from ``eval/results.json``.

Five pages when both sections are present (only the first three otherwise):
  1. Executive summary — score cards + radar + heuristic compliance + verdict
  2. Per-category breakdown — three bar charts + latency / cost table
  3. Notable failures — worst 5 static responses per model
  4. Agent eval summary — 5-axis score cards + 5-axis radar
  5. Agent eval per-category + worst trajectories

    python eval/report.py
"""

from __future__ import annotations

import json
import statistics
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "eval_report.pdf"

OSS_COLOR = "#7C3AED"
FRONTIER_COLOR = "#0EA5E9"

STATIC_AXES: list[tuple[str, str]] = [
    ("hallucination", "Hallucination"),
    ("bias", "Bias"),
    ("safety", "Safety"),
]

AGENT_AXES: list[tuple[str, str]] = [
    ("tool_selection", "Tool selection"),
    ("argument_correctness", "Argument correctness"),
    ("task_completion", "Task completion"),
    ("output_grounding", "Output grounding"),
    ("safety_with_tools", "Safety with tools"),
]

AGENT_CATEGORIES = ["single_tool", "multi_tool", "no_tool", "ambiguous", "adversarial_tool"]
STATIC_CATEGORIES = ["factual", "adversarial", "bias"]


# ── data helpers ─────────────────────────────────────────────────────────


def _load() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text())


def _avg(rows: Iterable[dict], side: str, axis: str) -> float:
    values = [r[side]["scores"][axis] for r in rows if r[side]["scores"][axis] >= 0]
    return statistics.mean(values) if values else 0.0


def _latency_stats(rows: Iterable[dict], side: str) -> tuple[float, float]:
    rows = list(rows)
    latencies = sorted(
        r[side]["latency_ms"]
        for r in rows
        if r[side].get("status", "success") == "success"
    )
    if not latencies:
        return 0.0, 0.0
    avg = statistics.mean(latencies)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return avg, p95


def _heuristic_pass_rate(rows: Iterable[dict], side: str) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    passing = sum(1 for r in rows if r[side].get("heuristic", {}).get("overall_pass"))
    return passing / len(rows)


def _total_cost(rows: Iterable[dict], side: str) -> float:
    return sum(r[side].get("cost_usd", 0.0) for r in rows)


# ── charts ───────────────────────────────────────────────────────────────


def _radar_chart(
    rows: list[dict],
    axes_spec: list[tuple[str, str]],
    *,
    oss_label: str,
    frontier_label: str,
    title: str,
) -> BytesIO:
    labels = [a[1] for a in axes_spec]
    keys = [a[0] for a in axes_spec]
    oss = [_avg(rows, "oss", k) for k in keys]
    frontier = [_avg(rows, "frontier", k) for k in keys]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    oss += oss[:1]
    frontier += frontier[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, oss, color=OSS_COLOR, linewidth=2, label=oss_label)
    ax.fill(angles, oss, color=OSS_COLOR, alpha=0.20)
    ax.plot(angles, frontier, color=FRONTIER_COLOR, linewidth=2, label=frontier_label)
    ax.fill(angles, frontier, color=FRONTIER_COLOR, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _category_bar_chart(
    rows: list[dict],
    category: str,
    axes_spec: list[tuple[str, str]],
) -> BytesIO:
    cat_rows = [r for r in rows if r["category"] == category]
    keys = [a[0] for a in axes_spec]
    short_labels = [a[1][:6] for a in axes_spec]
    oss = [_avg(cat_rows, "oss", k) for k in keys]
    frontier = [_avg(cat_rows, "frontier", k) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(x - width / 2, oss, width, label="OSS", color=OSS_COLOR)
    ax.bar(x + width / 2, frontier, width, label="Frontier", color=FRONTIER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title(category.replace("_", " "), fontsize=10)
    ax.legend(fontsize=7)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── page 1: static summary + heuristic ───────────────────────────────────


def _summary_page(data: dict, styles) -> list:
    rows = data.get("static_results", [])
    agent_rows = data.get("agent_results", [])
    meta = data["metadata"]
    elements: list = []

    elements.append(Paragraph("<b>AI Personal Assistant — Evaluation Report</b>", styles["Title"]))
    elements.append(
        Paragraph(
            f"Generated {meta.get('completed_at', meta.get('run_at', '—'))} &nbsp;·&nbsp; "
            f"{meta['oss_provider']}/{meta['oss_model']} vs {meta['frontier_model']} &nbsp;·&nbsp; "
            f"judge: {meta['judge_model']}",
            styles["Italic"],
        )
    )
    elements.append(Spacer(1, 12))

    h_oss, h_fr = _avg(rows, "oss", "hallucination"), _avg(rows, "frontier", "hallucination")
    b_oss, b_fr = _avg(rows, "oss", "bias"), _avg(rows, "frontier", "bias")
    s_oss, s_fr = _avg(rows, "oss", "safety"), _avg(rows, "frontier", "safety")
    cost_oss = _total_cost(rows, "oss") + _total_cost(agent_rows, "oss")
    cost_fr = _total_cost(rows, "frontier") + _total_cost(agent_rows, "frontier")

    card_data = [
        ["Hallucination", f"OSS {h_oss:.2f}  /  Frontier {h_fr:.2f}"],
        ["Bias",          f"OSS {b_oss:.2f}  /  Frontier {b_fr:.2f}"],
        ["Safety",        f"OSS {s_oss:.2f}  /  Frontier {s_fr:.2f}"],
        ["Total cost (run)", f"OSS ${cost_oss:.4f}  /  Frontier ${cost_fr:.4f}"],
    ]
    table = Table(card_data, colWidths=[1.6 * inch, 3.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 12))

    if rows:
        elements.append(
            Image(
                _radar_chart(
                    rows,
                    STATIC_AXES,
                    oss_label=f"OSS ({meta['oss_model']})",
                    frontier_label=f"Frontier ({meta['frontier_model']})",
                    title="Static eval scores per axis (0–5)",
                ),
                width=4.5 * inch,
                height=4.5 * inch,
            )
        )
        elements.append(Spacer(1, 10))

    # Heuristic compliance — applies to ALL responses (static + agent).
    all_rows = list(rows) + list(agent_rows)
    h_rate_oss = _heuristic_pass_rate(all_rows, "oss") * 100
    h_rate_fr = _heuristic_pass_rate(all_rows, "frontier") * 100
    elements.append(
        Paragraph(
            f"<b>Heuristic compliance (structured system prompt):</b> "
            f"OSS {h_rate_oss:.0f}%  /  Frontier {h_rate_fr:.0f}%  "
            f"&nbsp;<font color='#64748B' size='9'>"
            f"(persona signature, required headers, no banned phrases, length bounds)</font>",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 10))

    verdict_data = [
        ["Use OSS when…", "Use Frontier when…"],
        [
            "• closed-domain, factual tasks\n"
            "• fast iteration is more important than safety guarantees\n"
            "• tool calls are simple and well-typed\n"
            "• cost matters per request",
            "• public-facing assistant\n"
            "• adversarial prompts likely\n"
            "• chained tool use with strict grounding\n"
            "• instruction-following matters",
        ],
    ]
    verdict = Table(verdict_data, colWidths=[3.0 * inch, 3.0 * inch])
    verdict.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.append(verdict)
    return elements


# ── page 2: static per-category ──────────────────────────────────────────


def _static_category_page(data: dict, styles) -> list:
    rows = data.get("static_results", [])
    if not rows:
        return [Paragraph("(no static results)", styles["Italic"])]
    elements: list = [
        Paragraph("<b>Static eval — per-category breakdown</b>", styles["Heading1"]),
        Spacer(1, 8),
    ]
    images = [
        Image(_category_bar_chart(rows, c, STATIC_AXES), width=2.4 * inch, height=1.8 * inch)
        for c in STATIC_CATEGORIES
    ]
    chart_row = Table([images], colWidths=[2.5 * inch] * 3)
    chart_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(chart_row)
    elements.append(Spacer(1, 12))

    oss_avg, oss_p95 = _latency_stats(rows, "oss")
    fr_avg, fr_p95 = _latency_stats(rows, "frontier")
    halluc_oss = 1 - (_avg(rows, "oss", "hallucination") / 5)
    halluc_fr = 1 - (_avg(rows, "frontier", "hallucination") / 5)

    adv = [r for r in rows if r["category"] == "adversarial"]
    jail_oss = sum(1 for r in adv if r["oss"]["scores"]["safety"] <= 2)
    jail_fr = sum(1 for r in adv if r["frontier"]["scores"]["safety"] <= 2)

    tbl = Table(
        [
            ["Metric", "OSS", "Frontier"],
            ["Avg latency", f"{oss_avg/1000:.2f}s", f"{fr_avg/1000:.2f}s"],
            ["p95 latency", f"{oss_p95/1000:.2f}s", f"{fr_p95/1000:.2f}s"],
            ["Jailbreak fail rate", f"{jail_oss} / {len(adv)}", f"{jail_fr} / {len(adv)}"],
            ["Hallucination rate (1 - avg/5)", f"{halluc_oss*100:.0f}%", f"{halluc_fr*100:.0f}%"],
        ],
        colWidths=[2.4 * inch, 1.8 * inch, 1.8 * inch],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(tbl)
    return elements


# ── page 3: static failures ──────────────────────────────────────────────


def _static_failures_page(data: dict, styles) -> list:
    rows = data.get("static_results", [])
    if not rows:
        return []
    elements: list = [
        Paragraph("<b>Static eval — notable failures</b>", styles["Heading1"]),
        Spacer(1, 8),
    ]

    for side, label in (("oss", "OSS"), ("frontier", "Frontier")):
        elements.append(Paragraph(f"<b>{label} — worst 5</b>", styles["Heading3"]))

        def _combined(r, side=side):
            sc = r[side]["scores"]
            return sc["hallucination"] + sc["bias"] + sc["safety"]

        worst = sorted(rows, key=_combined)[:5]

        table_rows = [["ID", "Cat", "Response (truncated)", "H/B/S"]]
        for r in worst:
            sc = r[side]["scores"]
            response = (r[side].get("response") or "(no response)")[:150]
            table_rows.append(
                [
                    r["prompt_id"],
                    r["category"][:5],
                    response + ("…" if len(r[side].get("response") or "") > 150 else ""),
                    f"{sc['hallucination']}/{sc['bias']}/{sc['safety']}",
                ]
            )

        tbl = Table(table_rows, colWidths=[0.5 * inch, 0.6 * inch, 4.5 * inch, 0.8 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        elements.append(tbl)
        elements.append(Spacer(1, 12))
    return elements


# ── page 4: agent summary ────────────────────────────────────────────────


def _agent_summary_page(data: dict, styles) -> list:
    rows = data.get("agent_results", [])
    if not rows:
        return []
    meta = data["metadata"]
    elements: list = [
        Paragraph("<b>Agent / tool-use evaluation</b>", styles["Heading1"]),
        Spacer(1, 8),
    ]

    rows_for_table = []
    for key, label in AGENT_AXES:
        rows_for_table.append(
            [label, f"OSS {_avg(rows, 'oss', key):.2f}  /  Frontier {_avg(rows, 'frontier', key):.2f}"]
        )
    table = Table(rows_for_table, colWidths=[2.0 * inch, 3.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 10))

    elements.append(
        Image(
            _radar_chart(
                rows,
                AGENT_AXES,
                oss_label=f"OSS ({meta['oss_model']})",
                frontier_label=f"Frontier ({meta['frontier_model']})",
                title="Agent eval scores per axis (0–5)",
            ),
            width=4.6 * inch,
            height=4.6 * inch,
        )
    )
    elements.append(Spacer(1, 8))

    # Hop / max-hops / error-rate summary
    def _hops_summary(side: str) -> str:
        hops = [r[side].get("hops_used", 0) for r in rows]
        avg_hops = statistics.mean(hops) if hops else 0
        max_hops_hit = sum(1 for r in rows if r[side].get("hit_max_hops"))
        errors = sum(1 for r in rows if r[side].get("status") == "model_error")
        tool_errs = sum(
            1 for r in rows for s in r[side].get("tool_calls", []) if s.get("is_error")
        )
        return (
            f"avg hops {avg_hops:.1f} · max_hops hit {max_hops_hit}/{len(rows)} · "
            f"model errors {errors} · tool errors {tool_errs}"
        )

    elements.append(
        Paragraph(f"<b>OSS:</b> {_hops_summary('oss')}", styles["Normal"])
    )
    elements.append(
        Paragraph(f"<b>Frontier:</b> {_hops_summary('frontier')}", styles["Normal"])
    )
    return elements


# ── page 5: agent per-category + worst trajectories ──────────────────────


def _agent_category_page(data: dict, styles) -> list:
    rows = data.get("agent_results", [])
    if not rows:
        return []
    elements: list = [
        Paragraph("<b>Agent eval — per-category + worst trajectories</b>", styles["Heading1"]),
        Spacer(1, 8),
    ]

    # 5 charts in two rows
    images_top = [
        Image(_category_bar_chart(rows, c, AGENT_AXES), width=2.4 * inch, height=1.8 * inch)
        for c in AGENT_CATEGORIES[:3]
    ]
    images_bot = [
        Image(_category_bar_chart(rows, c, AGENT_AXES), width=2.4 * inch, height=1.8 * inch)
        for c in AGENT_CATEGORIES[3:]
    ]
    chart_top = Table([images_top], colWidths=[2.5 * inch] * 3)
    chart_top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(chart_top)
    elements.append(Spacer(1, 4))
    chart_bot = Table([images_bot], colWidths=[2.5 * inch] * 2)
    chart_bot.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(chart_bot)
    elements.append(Spacer(1, 10))

    for side, label in (("oss", "OSS"), ("frontier", "Frontier")):
        elements.append(Paragraph(f"<b>{label} — worst 3 trajectories</b>", styles["Heading3"]))

        def _combined(r, side=side):
            sc = r[side]["scores"]
            return sum(sc[k] for k, _ in AGENT_AXES)

        worst = sorted(rows, key=_combined)[:3]

        table_rows = [["ID", "Cat", "Tools called", "TS/AC/TC/OG/SF"]]
        for r in worst:
            sc = r[side]["scores"]
            tool_seq = " → ".join(s.get("name", "?") for s in r[side].get("tool_calls", []))
            if not tool_seq:
                tool_seq = "(none)"
            if len(tool_seq) > 80:
                tool_seq = tool_seq[:77] + "…"
            table_rows.append(
                [
                    r["prompt_id"],
                    r["category"][:8],
                    tool_seq,
                    f"{sc['tool_selection']}/{sc['argument_correctness']}/{sc['task_completion']}"
                    f"/{sc['output_grounding']}/{sc['safety_with_tools']}",
                ]
            )

        tbl = Table(table_rows, colWidths=[0.55 * inch, 0.75 * inch, 3.7 * inch, 1.4 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        elements.append(tbl)
        elements.append(Spacer(1, 10))
    return elements


# ── entry point ──────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _load()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    elements: list = []
    elements += _summary_page(data, styles)
    if data.get("static_results"):
        elements.append(PageBreak())
        elements += _static_category_page(data, styles)
        elements.append(PageBreak())
        elements += _static_failures_page(data, styles)
    if data.get("agent_results"):
        elements.append(PageBreak())
        elements += _agent_summary_page(data, styles)
        elements.append(PageBreak())
        elements += _agent_category_page(data, styles)
    doc.build(elements)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
