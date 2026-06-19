from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from document_ocr import process_pages

from .config import OCRConfig, project_path


class DocumentOCRService:
    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig.from_env()
        self.config.validate_static_paths()

    def run(
        self,
        source: str | Path,
        out: str | Path | None = None,
        *,
        job_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_path = project_path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        job_id = job_id or uuid.uuid4().hex
        output_dir = Path(out) if out is not None else Path(self.config.out) / job_id
        output_dir = project_path(output_dir)

        overrides = overrides or {}
        run_config = self.config.with_overrides(source=str(source_path), out=str(output_dir), **overrides)
        args = run_config.to_namespace()
        process_pages(args)

        return self._collect_result(job_id=job_id, output_dir=output_dir, config=run_config)

    def _collect_result(self, *, job_id: str, output_dir: Path, config: OCRConfig) -> dict[str, Any]:
        lines_dir = output_dir / "lines"
        texts_dir = output_dir / "texts"
        metadata_dir = output_dir / "metadata"
        debug_dir = output_dir / "debug_boxes"

        pages = []
        for lines_json in sorted(lines_dir.glob("*.json")):
            page_name = lines_json.stem
            lines_payload = json.loads(lines_json.read_text(encoding="utf-8"))
            text_path = texts_dir / f"{page_name}.txt"
            metadata_path = metadata_dir / f"{page_name}.json"
            pages.append(
                {
                    "page_name": page_name,
                    "text": text_path.read_text(encoding="utf-8") if text_path.exists() else "",
                    "lines": lines_payload.get("lines", []),
                    "files": {
                        "text": str(text_path),
                        "lines_json": str(lines_json),
                        "lines_tsv": str(lines_dir / f"{page_name}.tsv"),
                        "regions_json": str(output_dir / "regions" / f"{page_name}.json"),
                        "regions_tsv": str(output_dir / "regions" / f"{page_name}.tsv"),
                        "metadata": str(metadata_path),
                        "debug_style_regions": str(debug_dir / f"{page_name}_style_regions.jpg"),
                        "debug_final_lines": str(debug_dir / f"{page_name}_final_lines.jpg"),
                    },
                }
            )

        return {
            "job_id": job_id,
            "output_dir": str(output_dir),
            "page_count": len(pages),
            "pages": pages,
            "config": {
                "mode": config.mode,
                "paddle_device": config.paddle_device,
                "qwen_enabled": config.qwen_enabled,
                "checkpoint": str(project_path(config.checkpoint)),
                "charset_file": str(project_path(config.charset_file)),
            },
        }

    def inspect_runtime(self) -> dict[str, Any]:
        checkpoint = project_path(self.config.checkpoint)
        charset = project_path(self.config.charset_file)
        return {
            "ready": checkpoint.exists() and charset.exists(),
            "project_root": str(project_path(".")),
            "checkpoint": {"path": str(checkpoint), "exists": checkpoint.exists()},
            "charset_file": {"path": str(charset), "exists": charset.exists()},
            "config": asdict(self.config),
        }

    # ── High-level entry used by async jobs ────────────────────────────
    def run_file(
        self,
        input_path: Path,
        output_dir: Path,
        mode: str = "BALANCED",
        paddle_device: str | None = None,
        qwen_enabled: bool = False,
    ) -> dict[str, Any]:
        """Process an image or PDF and return a frontend-compatible result.

        If *input_path* is a PDF, convert each page to an image first.
        Then delegate to the existing ``run()`` method.
        """
        from .pdf import convert_pdf_to_images, is_pdf
        from .config import MAX_PDF_PAGES, PDF_DPI

        overrides: dict[str, Any] = {
            "mode": mode,
            "qwen_enabled": qwen_enabled,
        }
        if paddle_device:
            overrides["paddle_device"] = paddle_device

        if is_pdf(input_path):
            pages_dir = input_path.parent.parent / "pages"
            page_images = convert_pdf_to_images(
                input_path, pages_dir, dpi=PDF_DPI, max_pages=MAX_PDF_PAGES
            )
        else:
            page_images = [input_path]

        # Process all pages into the same output dir
        result = self.run(
            page_images[0] if len(page_images) == 1 else page_images[0],
            out=output_dir,
            overrides=overrides,
        )

        # For multi-page PDFs, process remaining pages
        if len(page_images) > 1:
            for img in page_images[1:]:
                extra = self.run(
                    img,
                    out=output_dir,
                    overrides={**overrides, "clean_output": False},
                )
                result["pages"].extend(extra.get("pages", []))
            result["page_count"] = len(result["pages"])

        return result

