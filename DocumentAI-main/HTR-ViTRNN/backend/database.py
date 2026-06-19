"""Job metadata storage backed by SQLAlchemy (SQLite for MVP, Postgres-ready)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import project_path

_DEFAULT_DB = f"sqlite:///{project_path('backend_runs/jobs.sqlite3')}"
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DB)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class OCRJob(Base):  # type: ignore[valid-type]
    __tablename__ = "ocr_jobs"

    job_id = Column(String, primary_key=True)
    original_filename = Column(Text, nullable=False)
    file_type = Column(String(16), nullable=False)
    status = Column(String(24), nullable=False, default="queued")
    progress = Column(Integer, nullable=False, default=0)
    stage = Column(String(64), nullable=True)
    page_count = Column(Integer, nullable=True)
    current_page = Column(Integer, nullable=True)
    output_dir = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    mode = Column(String(24), nullable=True)
    paddle_device = Column(String(24), nullable=True)
    qwen_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(row: OCRJob) -> dict[str, Any]:
    return {
        "job_id": row.job_id,
        "original_filename": row.original_filename,
        "file_type": row.file_type,
        "status": row.status,
        "progress": row.progress,
        "stage": row.stage,
        "page_count": row.page_count,
        "current_page": row.current_page,
        "output_dir": row.output_dir,
        "error": row.error,
        "mode": row.mode,
        "paddle_device": row.paddle_device,
        "qwen_enabled": row.qwen_enabled,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def create_job(
    *,
    job_id: str,
    original_filename: str,
    file_type: str,
    mode: str = "BALANCED",
    paddle_device: str = "gpu:0",
    qwen_enabled: bool = False,
) -> dict[str, Any]:
    now = _utcnow()
    row = OCRJob(
        job_id=job_id,
        original_filename=original_filename,
        file_type=file_type,
        status="queued",
        progress=0,
        stage="waiting",
        mode=mode,
        paddle_device=paddle_device,
        qwen_enabled=qwen_enabled,
        created_at=now,
        updated_at=now,
    )
    with SessionLocal() as session:  # type: Session
        session.add(row)
        session.commit()
        session.refresh(row)
        return _row_to_dict(row)


def update_job(job_id: str, **kwargs: Any) -> dict[str, Any] | None:
    with SessionLocal() as session:  # type: Session
        row = session.query(OCRJob).filter_by(job_id=job_id).first()
        if row is None:
            return None
        for key, value in kwargs.items():
            if hasattr(row, key):
                setattr(row, key, value)
        row.updated_at = _utcnow()
        if kwargs.get("status") in ("done", "failed"):
            row.completed_at = _utcnow()
        session.commit()
        session.refresh(row)
        return _row_to_dict(row)


def get_job(job_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:  # type: Session
        row = session.query(OCRJob).filter_by(job_id=job_id).first()
        return _row_to_dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with SessionLocal() as session:  # type: Session
        rows = (
            session.query(OCRJob)
            .order_by(OCRJob.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(r) for r in rows]


def delete_job(job_id: str) -> bool:
    with SessionLocal() as session:  # type: Session
        row = session.query(OCRJob).filter_by(job_id=job_id).first()
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True
