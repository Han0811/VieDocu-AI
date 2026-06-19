"""File-system helpers for job directories, uploads, and downloads."""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from fastapi import UploadFile

from .config import project_path

RUNS_DIR_ENV = "OCR_RUNS_DIR"
DEFAULT_RUNS = "backend_runs"


def _runs_root() -> Path:
    return project_path(os.getenv(RUNS_DIR_ENV, DEFAULT_RUNS))


def job_root(job_id: str) -> Path:
    return _runs_root() / job_id


def create_job_dirs(job_id: str) -> dict[str, Path]:
    """Create and return the standard directory layout for a job."""
    root = job_root(job_id)
    dirs = {
        "root": root,
        "input": root / "input",
        "pages": root / "pages",
        "output": root / "output",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_upload_file(upload_file: UploadFile, dst: Path) -> None:
    """Stream an uploaded file to *dst*."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)


def safe_output_file_path(job_id: str, relative_path: str) -> Path:
    """Resolve *relative_path* inside the job output dir safely.

    Raises ``ValueError`` on path-traversal attempts.
    """
    output_dir = job_root(job_id) / "output"
    resolved = (output_dir / relative_path).resolve()
    if not str(resolved).startswith(str(output_dir.resolve())):
        raise ValueError("Path traversal detected")
    return resolved


def build_file_url(job_id: str, output_relative_path: str) -> str:
    """Map a local output-relative path to the public API URL."""
    clean = output_relative_path.replace("\\", "/").lstrip("/")
    return f"/api/jobs/{job_id}/files/{clean}"


def zip_output_dir(job_id: str) -> Path:
    """Create (or reuse) a ZIP of the job output directory.

    Returns the path to the created zip file.
    """
    output_dir = job_root(job_id) / "output"
    zip_path = job_root(job_id) / "output.zip"
    if zip_path.exists():
        return zip_path
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(output_dir)
                zf.write(file, arcname)
    return zip_path


def delete_job_files(job_id: str) -> None:
    """Remove the entire job directory from disk."""
    root = job_root(job_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
