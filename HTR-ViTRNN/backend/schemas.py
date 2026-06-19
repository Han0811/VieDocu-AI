"""Pydantic response models for the async job API."""
from __future__ import annotations

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str | None = None
    page_count: int | None = None
    current_page: int | None = None
    error: str | None = None
    result_url: str | None = None
    download_zip_url: str | None = None
    created_at: str
    updated_at: str


class OCRResultResponse(BaseModel):
    job_id: str
    status: str
    page_count: int
    pages: list[dict]


class JobDeleteResponse(BaseModel):
    job_id: str
    deleted: bool
