from __future__ import annotations

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ModuleNotFoundError as exc:
    raise RuntimeError("Install API dependencies first: pip install -r requirements-backend.txt") from exc

from .config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, OCRConfig, project_path
from .database import init_db
from .jobs import enqueue_ocr_job, generate_job_id, get_job_result, shutdown_executor
from .schemas import (
    JobCreateResponse,
    JobDeleteResponse,
    JobStatusResponse,
    OCRResultResponse,
)
from .security import api_key_middleware
from .service import DocumentOCRService
from .storage import (
    build_file_url,
    create_job_dirs,
    delete_job_files,
    safe_output_file_path,
    save_upload_file,
    zip_output_dir,
)
from . import database as db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    init_db()
    logger.info("Database initialised")
    yield
    shutdown_executor()
    logger.info("Worker executor shut down")


app = FastAPI(title="Vietnamese Document OCR Backend", lifespan=lifespan)

# CORS — allow frontend origins
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API-key middleware
app.middleware("http")(api_key_middleware)

service = DocumentOCRService(OCRConfig.from_env())


# ═══════════════════════════════════════════════════════════════════════════
# Existing endpoints (kept unchanged)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict:
    runtime = service.inspect_runtime()
    return {"status": "ok" if runtime["ready"] else "not_ready", **runtime}


@app.get("/config")
def config() -> dict:
    return service.inspect_runtime()["config"]


@app.post("/ocr")
def run_ocr(
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = "BALANCED",
    paddle_device: Annotated[str | None, Form()] = None,
    qwen_enabled: Annotated[bool, Form()] = False,
) -> dict:
    job_id = uuid.uuid4().hex
    job_root = project_path("backend_runs") / job_id
    input_dir = job_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "page.png").suffix or ".png"
    input_path = input_dir / f"page{suffix}"
    with input_path.open("wb") as dst:
        shutil.copyfileobj(file.file, dst)

    try:
        return service.run(
            input_path,
            out=job_root / "output",
            job_id=job_id,
            overrides={
                "mode": mode,
                "paddle_device": paddle_device,
                "qwen_enabled": qwen_enabled,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ═══════════════════════════════════════════════════════════════════════════
# New async job endpoints
# ═══════════════════════════════════════════════════════════════════════════

def _validate_upload(file: UploadFile) -> tuple[str, str]:
    """Validate uploaded file. Returns (extension, file_type)."""
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Check size (Content-Length may not be set; we validate after save)
    file_type = "pdf" if ext == ".pdf" else "image"
    return ext, file_type


@app.post("/api/jobs", response_model=JobCreateResponse)
def submit_job(
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = "BALANCED",
    paddle_device: Annotated[str | None, Form()] = None,
    qwen_enabled: Annotated[bool, Form()] = False,
) -> JobCreateResponse:
    """Submit a new OCR job (async)."""
    ext, file_type = _validate_upload(file)

    job_id = generate_job_id()
    dirs = create_job_dirs(job_id)

    # Save original file
    original_name = file.filename or f"upload{ext}"
    dst_path = dirs["input"] / original_name
    save_upload_file(file, dst_path)

    # Validate file size after save
    file_mb = dst_path.stat().st_size / (1024 * 1024)
    if file_mb > MAX_UPLOAD_MB:
        delete_job_files(job_id)
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_mb:.1f} MB). Maximum: {MAX_UPLOAD_MB} MB",
        )

    # Create DB record
    job = db.create_job(
        job_id=job_id,
        original_filename=original_name,
        file_type=file_type,
        mode=mode,
        paddle_device=paddle_device or service.config.paddle_device,
        qwen_enabled=qwen_enabled,
    )

    # Enqueue
    enqueue_ocr_job(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        message="Job queued successfully",
        created_at=job["created_at"],
    )


@app.get("/api/jobs")
def list_jobs(limit: int = 50) -> list[dict]:
    """List recent jobs."""
    return db.list_jobs(limit=limit)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a job."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result_url = None
    download_zip_url = None
    if job["status"] == "done":
        result_url = f"/api/jobs/{job_id}/result"
        download_zip_url = f"/api/jobs/{job_id}/download/zip"

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        stage=job.get("stage"),
        page_count=job.get("page_count"),
        current_page=job.get("current_page"),
        error=job.get("error"),
        result_url=result_url,
        download_zip_url=download_zip_url,
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@app.get("/api/jobs/{job_id}/result", response_model=OCRResultResponse)
def get_result(job_id: str) -> OCRResultResponse:
    """Get the full OCR result for a completed job."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done (status={job['status']})")

    result = get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result file not found")

    return OCRResultResponse(
        job_id=result["job_id"],
        status=result["status"],
        page_count=result["page_count"],
        pages=result["pages"],
    )


@app.get("/api/jobs/{job_id}/files/{file_path:path}")
def serve_output_file(job_id: str, file_path: str) -> FileResponse:
    """Serve a file from the job output directory."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        resolved = safe_output_file_path(job_id, file_path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(resolved)


@app.get("/api/jobs/{job_id}/download/zip")
def download_zip(job_id: str) -> FileResponse:
    """Download a ZIP of the entire job output."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job is not done yet")

    zip_path = zip_output_dir(job_id)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"ocr_{job_id}.zip",
    )


@app.delete("/api/jobs/{job_id}", response_model=JobDeleteResponse)
def delete_job(job_id: str) -> JobDeleteResponse:
    """Delete a job and its files."""
    deleted = db.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    delete_job_files(job_id)
    return JobDeleteResponse(job_id=job_id, deleted=True)
