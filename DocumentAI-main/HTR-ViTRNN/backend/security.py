"""Optional API-key middleware for protected endpoints."""
from __future__ import annotations

import os

from fastapi import HTTPException, Request

ENABLE_API_KEY = os.getenv("ENABLE_API_KEY", "false").lower() in ("1", "true", "yes")
API_KEY = os.getenv("API_KEY", "")

# Paths that never require an API key.
PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


async def api_key_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Starlette middleware that checks ``X-API-Key`` when enabled."""
    if not ENABLE_API_KEY:
        return await call_next(request)

    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
        return await call_next(request)

    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return await call_next(request)
