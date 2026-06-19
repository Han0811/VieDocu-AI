"""PDF-to-image conversion using pypdfium2."""
from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def is_pdf(path: Path) -> bool:
    """Return *True* if *path* has a PDF extension."""
    return path.suffix.lower() == ".pdf"


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 220,
    max_pages: int | None = None,
) -> list[Path]:
    """Render each page of *pdf_path* to a PNG in *output_dir*.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF file.
    output_dir:
        Directory where page images are saved (``page_001.png``, …).
    dpi:
        Rendering resolution (default 220).
    max_pages:
        Optional upper limit.  Raises ``ValueError`` if the PDF
        exceeds this count.

    Returns
    -------
    list[Path]
        Sorted list of output image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    page_count = len(pdf)

    if max_pages is not None and page_count > max_pages:
        pdf.close()
        raise ValueError(
            f"PDF has {page_count} pages, exceeding the limit of {max_pages}"
        )

    scale = dpi / 72.0
    image_paths: list[Path] = []

    for idx in range(page_count):
        page = pdf[idx]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Ensure RGB (drop alpha if present)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        out_path = output_dir / f"page_{idx + 1:03d}.png"
        pil_image.save(str(out_path), format="PNG")
        image_paths.append(out_path)

    pdf.close()
    return image_paths
