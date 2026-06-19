"""Job orchestration — enqueue and execute OCR jobs."""
from __future__ import annotations

import json
import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import database as db
from .config import OCRConfig, project_path
from .pdf import convert_pdf_to_images, is_pdf
from .storage import build_file_url, job_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process thread-pool queue (MVP).
# Swap for RQ / Redis in production by changing enqueue_ocr_job().
# ---------------------------------------------------------------------------
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr_worker")
    return _executor


def shutdown_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_job_id() -> str:
    """Produce a timestamp-based job ID like ``20260613_153000_abcd1234``."""
    now = datetime.now(timezone.utc)
    short = uuid.uuid4().hex[:8]
    return now.strftime("%Y%m%d_%H%M%S") + f"_{short}"


def enqueue_ocr_job(job_id: str) -> None:
    """Push *job_id* into the worker queue."""
    _get_executor().submit(_safe_run, job_id)


def _safe_run(job_id: str) -> None:
    try:
        run_ocr_job(job_id)
    except Exception:
        logger.exception("Unhandled error in OCR job %s", job_id)
        try:
            db.update_job(job_id, status="failed", error=traceback.format_exc()[-500:])
        except Exception:
            logger.exception("Failed to persist error for job %s", job_id)


# ---------------------------------------------------------------------------
# Core job execution (called by worker)
# ---------------------------------------------------------------------------

def run_ocr_job(job_id: str) -> None:
    """Execute a queued OCR job end-to-end."""
    job = db.get_job(job_id)
    if job is None:
        logger.error("Job %s not found in database", job_id)
        return

    db.update_job(job_id, status="processing", progress=5, stage="input_saved")

    root = job_root(job_id)
    input_dir = root / "input"
    pages_dir = root / "pages"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate the uploaded file
    input_files = list(input_dir.iterdir())
    if not input_files:
        db.update_job(job_id, status="failed", error="No input file found")
        return
    input_path = input_files[0]

    # ── PDF conversion ─────────────────────────────────────────────────
    config = OCRConfig.from_env()
    max_pdf_pages = int(config.__class__.__dict__.get("MAX_PDF_PAGES", 50))
    # Read from env directly for upload-specific limits
    import os
    max_pdf_pages = int(os.getenv("MAX_PDF_PAGES", "50"))
    pdf_dpi = int(os.getenv("PDF_DPI", "220"))

    if is_pdf(input_path):
        db.update_job(job_id, progress=8, stage="converting_pdf")
        try:
            page_images = convert_pdf_to_images(
                input_path, pages_dir, dpi=pdf_dpi, max_pages=max_pdf_pages
            )
        except ValueError as exc:
            db.update_job(job_id, status="failed", error=str(exc))
            return
        db.update_job(
            job_id,
            progress=10,
            stage="pdf_converted",
            page_count=len(page_images),
        )
    else:
        page_images = [input_path]
        db.update_job(job_id, progress=10, stage="pdf_converted", page_count=1)

    # ── OCR processing ─────────────────────────────────────────────────
    from .service import DocumentOCRService

    mode = job.get("mode", "BALANCED")
    paddle_device = job.get("paddle_device") or config.paddle_device
    qwen_enabled = bool(job.get("qwen_enabled", False))

    service = DocumentOCRService(
        config.with_overrides(
            mode=mode,
            paddle_device=paddle_device,
            qwen_enabled=qwen_enabled,
        )
    )

    db.update_job(job_id, progress=20, stage="ocr_started")

    total_pages = len(page_images)
    all_page_results: list[dict[str, Any]] = []

    for page_idx, img_path in enumerate(page_images):
        page_num = page_idx + 1
        progress = 20 + int(70 * page_num / total_pages)
        db.update_job(
            job_id,
            progress=min(progress, 90),
            stage=f"ocr_page_{page_num}_of_{total_pages}",
            current_page=page_num,
        )

        result = service.run(
            img_path,
            out=output_dir,
            job_id=job_id,
            overrides={
                "mode": mode,
                "paddle_device": paddle_device,
                "qwen_enabled": qwen_enabled,
                "clean_output": page_idx == 0,  # Only clean on first page
            },
        )
        all_page_results.append(result)

    # ── Build result.json ──────────────────────────────────────────────
    db.update_job(job_id, progress=95, stage="result_packaging")

    combined = _build_result_json(job_id, all_page_results, output_dir)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    db.update_job(job_id, status="done", progress=100, stage="completed")
    logger.info("Job %s completed: %d page(s)", job_id, total_pages)


# ---------------------------------------------------------------------------
# Result building — maps local paths to API URLs
# ---------------------------------------------------------------------------

def _build_result_json(
    job_id: str,
    page_results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge per-page results into a single response with API URLs."""
    pages = []
    for page_idx, result in enumerate(page_results):
        for page in result.get("pages", []):
            mapped_page = _map_page_urls(job_id, page, output_dir)
            mapped_page["page_index"] = len(pages)
            pages.append(mapped_page)

    return {
        "job_id": job_id,
        "status": "done",
        "page_count": len(pages),
        "pages": pages,
    }


def _map_page_urls(
    job_id: str, page: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Replace local file paths with public API URLs."""
    mapped = dict(page)

    # Map page-level file references
    files = mapped.get("files", {})
    mapped_files = {}
    for key, local_path in files.items():
        if local_path and isinstance(local_path, str):
            try:
                rel = str(Path(local_path).relative_to(output_dir))
                mapped_files[key] = build_file_url(job_id, rel)
            except ValueError:
                mapped_files[key] = local_path
        else:
            mapped_files[key] = local_path
    mapped["files"] = mapped_files

    # Map line-level crop paths
    for line in mapped.get("lines", []):
        for path_key in ("crop_path", "line_crop_path"):
            val = line.get(path_key)
            if val and isinstance(val, str):
                try:
                    rel = str(Path(val).relative_to(output_dir))
                    line[path_key] = build_file_url(job_id, rel)
                except ValueError:
                    pass
        for part in line.get("parts", []):
            val = part.get("crop_path")
            if val and isinstance(val, str):
                try:
                    rel = str(Path(val).relative_to(output_dir))
                    part["crop_path"] = build_file_url(job_id, rel)
                except ValueError:
                    pass

    return mapped


def get_job_result(job_id: str) -> dict[str, Any] | None:
    """Read the cached result.json for a completed job."""
    result_path = job_root(job_id) / "output" / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))
