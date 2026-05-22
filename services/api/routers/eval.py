"""Read-only eval results browser + background runner.

Exposes:
    GET  /eval/results            full results.json payload
    GET  /eval/results/{prompt_id} single prompt row
    GET  /eval/report.pdf          download the rendered PDF
    POST /eval/run                 kick off a background eval (single-run-at-a-time)
    GET  /eval/runs/latest         poll current run status + partial rows

State lives in this module (single process, no concurrent runs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from eval.run_eval import run_all

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
# Override these in deployments where /app/eval is part of the image and
# would be wiped on container recreation. The prod overlay mounts a named
# volume at /app/var/eval and points these env vars there.
RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", str(_REPO_ROOT / "eval" / "results.json")))
REPORT_PATH = Path(os.getenv("EVAL_REPORT_PATH", str(_REPO_ROOT / "docs" / "eval_report.pdf")))


# ── In-memory run state ──────────────────────────────────────────────────


@dataclass
class RunStatus:
    run_id: str
    status: str               # queued | running | done | failed
    sections: list[str]
    limit: Optional[int]
    started_at: str
    completed_at: Optional[str] = None
    total_prompts: int = 0
    completed_prompts: int = 0
    current_prompt_id: Optional[str] = None
    error: Optional[str] = None
    partial_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_LOCK = threading.Lock()
_LATEST: Optional[RunStatus] = None
_BG_TASK: Optional[asyncio.Task] = None


def _set_latest(rs: RunStatus) -> None:
    global _LATEST
    with _LOCK:
        _LATEST = rs


def _get_latest() -> Optional[RunStatus]:
    with _LOCK:
        return _LATEST


def _is_running() -> bool:
    rs = _get_latest()
    return rs is not None and rs.status == "running"


def _track_background_task(task: asyncio.Task) -> None:
    """Hold a strong reference so the eval task is not GC'd mid-run."""
    global _BG_TASK
    _BG_TASK = task

    def _done(t: asyncio.Task) -> None:
        global _BG_TASK
        _BG_TASK = None
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.exception("background eval task failed", exc_info=exc)

    task.add_done_callback(_done)


def _load_results_file() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("results.json missing")
    return json.loads(RESULTS_PATH.read_text())


def _save_results_file(payload: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str))


# ── Pydantic request/response models ─────────────────────────────────────


class RunRequest(BaseModel):
    sections: list[str] = Field(default_factory=lambda: ["static", "agent"])
    limit: Optional[int] = None


class RunAccepted(BaseModel):
    run_id: str
    status: str


# ── Router ───────────────────────────────────────────────────────────────


router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/results")
async def get_results() -> dict:
    try:
        return await asyncio.to_thread(_load_results_file)
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "no results.json yet — run an eval via POST /eval/run or "
            "`python eval/run_eval.py`",
        ) from None
    except json.JSONDecodeError as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"results.json is corrupt: {e}",
        ) from e


@router.get("/results/{prompt_id}")
async def get_result_by_id(prompt_id: str) -> dict:
    payload = await get_results()
    for row in payload.get("static_results", []) + payload.get("agent_results", []):
        if row.get("prompt_id") == prompt_id:
            return row
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        f"prompt_id {prompt_id!r} not found in results.json",
    )


@router.get("/report.pdf")
async def get_report_pdf() -> FileResponse:
    if not REPORT_PATH.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "no eval_report.pdf yet — run an eval first",
        )
    return FileResponse(
        path=str(REPORT_PATH),
        media_type="application/pdf",
        filename="eval_report.pdf",
    )


@router.post("/run", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
async def start_run(body: RunRequest) -> RunAccepted:
    sections = [s.strip() for s in body.sections if s.strip()]
    invalid = [s for s in sections if s not in ("static", "agent")]
    if invalid:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"invalid sections {invalid!r}; must be subset of {{'static','agent'}}",
        )
    if not sections:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "sections must be a non-empty subset of {'static','agent'}",
        )
    if _is_running():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "another eval run is already in progress",
        )

    expected_total = 0
    if "static" in sections:
        from eval.prompts import ALL_PROMPTS

        expected_total += len(ALL_PROMPTS[: body.limit] if body.limit else ALL_PROMPTS)
    if "agent" in sections:
        from eval.agent_prompts import AGENT_PROMPTS

        expected_total += len(AGENT_PROMPTS[: body.limit] if body.limit else AGENT_PROMPTS)

    run_id = str(uuid.uuid4())
    rs = RunStatus(
        run_id=run_id,
        status="running",
        sections=sections,
        limit=body.limit,
        started_at=datetime.now(timezone.utc).isoformat(),
        total_prompts=expected_total,
    )
    _set_latest(rs)

    _track_background_task(asyncio.create_task(_run_in_background(rs)))

    return RunAccepted(run_id=run_id, status=rs.status)


@router.get("/runs/latest")
async def get_latest_run() -> dict:
    rs = _get_latest()
    if rs is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "no run has been started in this process yet",
        )
    return rs.to_dict()


# ── Background task ──────────────────────────────────────────────────────


async def _run_in_background(rs: RunStatus) -> None:
    def _on_progress(prompt_id: str, row: dict) -> None:
        with _LOCK:
            rs.completed_prompts += 1
            rs.current_prompt_id = prompt_id
            rs.partial_results.append(row)

    try:
        payload = await run_all(
            sections=rs.sections,
            limit=rs.limit,
            on_progress=_on_progress,
        )
        await asyncio.to_thread(_save_results_file, payload)
        try:
            # Regenerate the PDF too — best effort.
            from eval import report as report_module

            await asyncio.to_thread(report_module.main)
        except Exception as pdf_err:  # noqa: BLE001
            logger.exception("PDF regeneration failed")
            with _LOCK:
                rs.error = f"results saved, but PDF regen failed: {pdf_err}"
        with _LOCK:
            rs.status = "done"
            rs.completed_at = datetime.now(timezone.utc).isoformat()
    except Exception as e:  # noqa: BLE001
        logger.exception("eval run %s failed", rs.run_id)
        with _LOCK:
            rs.status = "failed"
            rs.error = str(e)
            rs.completed_at = datetime.now(timezone.utc).isoformat()
