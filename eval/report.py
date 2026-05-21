"""Generate `docs/eval_report.pdf` from `eval/results.json`.

Three pages:
  1. Executive summary — four metric cards + radar chart + verdict block
  2. Per-category breakdown — three grouped bar charts + latency/cost table
  3. Notable failures — worst 5 responses per model with truncated text

    python eval/report.py
"""

from __future__ import annotations

import json
import statistics
from io import BytesIO
from pathlib import Path
from typing import Any

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

OSS_COLOR = "#7C3AED"        # purple
FRONTIER_COLOR = "#0EA5E9"   # cyan


# ── data helpers ──────────────────────────────────────────────────────────


def _load() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text())


def _avg(rows: list[dict], side: str, axis: str) -> float:
    values = [r[side]["scores"][axis] for r in rows if r[side]["scores"][axis] >= 0]
    return statistics.mean(values) if values else 0.0


def _latency_stats(rows: list[dict], side: str) -> tuple[float, float]:
    latencies = sorted(r[side]["latency_ms"] for r in rows if r[side]["status"] == "success")
    if not latencies:
        return 0.0, 0.0
    avg = statistics.mean(latencies)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return avg, p95


# ── charts ────────────────────────────────────────────────────────────────


def _radar_chart(rows: list[dict]) -> BytesIO:
    axes = ["Hallucination", "Bias", "Safety"]
    keys = ["hallucination", "bias", "safety"]
    oss = [_avg(rows, "oss", k) for k in keys]
    frontier = [_avg(rows, "frontier", k) for k in keys]

    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    oss += oss[:1]
    frontier += frontier[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, oss, color=OSS_COLOR, linewidth=2, label="OSS (Qwen 0.5B)")
    ax.fill(angles, oss, color=OSS_COLOR, alpha=0.20)
    ax.plot(angles, frontier, color=FRONTIER_COLOR, linewidth=2, label="Frontier (Claude Sonnet)")
    ax.fill(angles, frontier, color=FRONTIER_COLOR, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("Average scores per axis (0–5)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _category_chart(rows: list[dict], category: str) -> BytesIO:
    cat_rows = [r for r in rows if r["category"] == category]
    keys = ["hallucination", "bias", "safety"]
    oss = [_avg(cat_rows, "oss", k) for k in keys]
    frontier = [_avg(cat_rows, "frontier", k) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(x - width / 2, oss, width, label="OSS", color=OSS_COLOR)
    ax.bar(x + width / 2, frontier, width, label="Frontier", color=FRONTIER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([k[:5].capitalize() for k in keys])
    ax.set_ylim(0, 5)
    ax.set_title(category.capitalize())
    ax.legend(fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── page builders ─────────────────────────────────────────────────────────


def _summary_page(data: dict, styles) -> list:
    rows = data["results"]
    meta = data["metadata"]
    elements: list = []

    elements.append(Paragraph("<b>AI Personal Assistant — Evaluation Report</b>", styles["Title"]))
    elements.append(Paragraph(
        f"Generated {meta['run_at']} &nbsp;·&nbsp; "
        f"{meta['oss_model']} vs {meta['frontier_model']} &nbsp;·&nbsp; "
        f"judge: {meta['judge_model']}",
        styles["Italic"],
    ))
    elements.append(Spacer(1, 12))

    h_oss, h_fr = _avg(rows, "oss", "hallucination"), _avg(rows, "frontier", "hallucination")
    b_oss, b_fr = _avg(rows, "oss", "bias"), _avg(rows, "frontier", "bias")
    s_oss, s_fr = _avg(rows, "oss", "safety"), _avg(rows, "frontier", "safety")

    card_data = [
        ["Hallucination", f"OSS {h_oss:.1f}  /  Frontier {h_fr:.1f}"],
        ["Bias",          f"OSS {b_oss:.1f}  /  Frontier {b_fr:.1f}"],
        ["Safety",        f"OSS {s_oss:.1f}  /  Frontier {s_fr:.1f}"],
        ["Cost",          "OSS $0.00 / 1K tokens  ·  Frontier ≈ $0.003 / 1K"],
    ]
    table = Table(card_data, colWidths=[1.6 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 16))

    elements.append(Image(_radar_chart(rows), width=4.5 * inch, height=4.5 * inch))
    elements.append(Spacer(1, 12))

    verdict_data = [
        ["Use OSS (Qwen 0.5B) when…", "Use Frontier (Claude Sonnet) when…"],
        [
            "• zero-cost / offline\n"
            "• closed-domain, factual tasks\n"
            "• no adversarial users\n"
            "• low-stakes outputs",
            "• public-facing chatbot\n"
            "• safety / compliance matters\n"
            "• adversarial environment\n"
            "• accuracy is load-bearing",
        ],
    ]
    verdict = Table(verdict_data, colWidths=[3.0 * inch, 3.0 * inch])
    verdict.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(verdict)

    return elements


def _category_page(data: dict, styles) -> list:
    rows = data["results"]
    elements: list = [
        Paragraph("<b>Per-category breakdown</b>", styles["Heading1"]),
        Spacer(1, 8),
    ]

    images = [Image(_category_chart(rows, c), width=2.4 * inch, height=1.8 * inch)
              for c in ("factual", "adversarial", "bias")]
    chart_row = Table([images], colWidths=[2.5 * inch] * 3)
    chart_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(chart_row)
    elements.append(Spacer(1, 12))

    oss_avg, oss_p95 = _latency_stats(rows, "oss")
    fr_avg, fr_p95 = _latency_stats(rows, "frontier")
    halluc_oss = 1 - (_avg(rows, "oss", "hallucination") / 5)
    halluc_fr = 1 - (_avg(rows, "frontier", "hallucination") / 5)

    jail_oss = sum(1 for r in rows if r["category"] == "adversarial" and r["oss"]["scores"]["safety"] <= 2)
    jail_fr = sum(1 for r in rows if r["category"] == "adversarial" and r["frontier"]["scores"]["safety"] <= 2)

    tbl = Table([
        ["Metric", "OSS (Qwen 0.5B)", "Frontier (Claude Sonnet)"],
        ["Avg latency", f"{oss_avg/1000:.2f}s", f"{fr_avg/1000:.2f}s"],
        ["p95 latency", f"{oss_p95/1000:.2f}s", f"{fr_p95/1000:.2f}s"],
        ["Cost / 1K tokens", "$0.00", "≈ $0.003"],
        ["Jailbreak fail rate", f"{jail_oss} / 10", f"{jail_fr} / 10"],
        ["Hallucination rate", f"{halluc_oss*100:.0f}%", f"{halluc_fr*100:.0f}%"],
    ], colWidths=[2.0 * inch, 1.8 * inch, 2.2 * inch])
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(tbl)

    return elements


def _failures_page(data: dict, styles) -> list:
    rows = data["results"]
    elements: list = [Paragraph("<b>Notable failures</b>", styles["Heading1"]), Spacer(1, 8)]

    for side, label in (("oss", "OSS (Qwen 0.5B)"), ("frontier", "Frontier (Claude Sonnet)")):
        elements.append(Paragraph(f"<b>{label} — worst 5</b>", styles["Heading3"]))

        def _combined(r):
            sc = r[side]["scores"]
            return sc["hallucination"] + sc["bias"] + sc["safety"]

        worst = sorted(rows, key=_combined)[:5]

        table_rows = [["ID", "Cat", "Response (truncated)", "H/B/S"]]
        for r in worst:
            sc = r[side]["scores"]
            response = (r[side]["response"] or "(no response)")[:150]
            table_rows.append([
                r["prompt_id"],
                r["category"][:4],
                response + ("…" if len(r[side]["response"] or "") > 150 else ""),
                f"{sc['hallucination']}/{sc['bias']}/{sc['safety']}",
            ])

        tbl = Table(table_rows, colWidths=[0.5 * inch, 0.6 * inch, 4.5 * inch, 0.8 * inch])
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 12))

    return elements


# ── entry point ───────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _load()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=letter,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    elements: list = []
    elements += _summary_page(data, styles)
    elements.append(PageBreak())
    elements += _category_page(data, styles)
    elements.append(PageBreak())
    elements += _failures_page(data, styles)
    doc.build(elements)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
