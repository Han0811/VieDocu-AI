"""Standalone OCR worker process.

Run with::

    python -m backend.worker

For MVP this simply keeps the thread-pool executor alive.
For production, swap to an RQ worker consuming from Redis.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from .database import init_db

    logger.info("Initialising database …")
    init_db()

    try:
        import redis as _redis_mod
        from rq import Worker, Queue, Connection

        redis_url = __import__("os").getenv("REDIS_URL", "redis://localhost:6379/0")
        conn = _redis_mod.from_url(redis_url)
        logger.info("Starting RQ worker (Redis at %s) …", redis_url)
        with Connection(conn):
            worker = Worker([Queue("ocr", connection=conn)])
            worker.work(with_scheduler=False)
    except (ImportError, Exception) as exc:
        if isinstance(exc, ImportError):
            logger.info("Redis/RQ not available — running in-process worker loop")
        else:
            logger.warning("Cannot connect to Redis (%s) — falling back to in-process loop", exc)

        _run_poll_loop()


def _run_poll_loop() -> None:
    """Simple polling loop for MVP (no Redis)."""
    from .database import SessionLocal, OCRJob
    from .jobs import run_ocr_job

    stop = False

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info("Worker poll loop started (Ctrl+C to stop)")

    while not stop:
        try:
            with SessionLocal() as session:
                row = (
                    session.query(OCRJob)
                    .filter_by(status="queued")
                    .order_by(OCRJob.created_at.asc())
                    .first()
                )
                job_id = row.job_id if row else None

            if job_id:
                logger.info("Picked up job %s", job_id)
                try:
                    run_ocr_job(job_id)
                except Exception:
                    logger.exception("Job %s failed", job_id)
            else:
                time.sleep(1)
        except Exception:
            logger.exception("Worker loop error")
            time.sleep(3)

    logger.info("Worker stopped")


if __name__ == "__main__":
    main()
