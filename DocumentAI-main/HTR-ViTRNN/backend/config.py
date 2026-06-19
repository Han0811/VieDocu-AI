from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Upload / PDF / infra limits ────────────────────────────────────────
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "50"))
PDF_DPI: int = int(os.getenv("PDF_DPI", "220"))
ALLOWED_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".pdf"
}
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'backend_runs' / 'jobs.sqlite3'}")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(slots=True)
class OCRConfig:
    source: str | None = None
    out: str = "backend_runs"
    mode: str = "BALANCED"
    clean_output: bool = True

    checkpoint: str = "./output/viet_lr1e4_64_layers_1024_dim/best_CER.pth"
    charset_file: str = "./output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json"
    nb_cls: int = 512
    img_size: list[int] = field(default_factory=lambda: [512, 64])
    num_layers_RNN: int = 1
    hidden_dim_RNN: int = 1024
    device: str | None = None

    pad_x_ratio: float = 0.0
    pad_y_ratio: float = 0.08
    htr_batch_size: int = 64

    paddle_device: str = "gpu:0"
    text_det_limit_side_len: int = 1536
    text_det_limit_type: str = "max"
    det_model_name: str = "PP-OCRv5_server_det"
    det_model_dir: str | None = None
    det_thresh: float = 0.2
    det_box_thresh: float = 0.4
    det_unclip_ratio: float = 2.2

    split_line_by_style: bool = True
    style_split_use_paddle_subboxes: bool = False
    style_split_min_score: float = 0.20
    style_split_min_width_ratio: float = 0.015
    style_split_min_height_ratio: float = 0.35
    style_split_min_area_ratio: float = 0.002
    style_split_merge_gap_ratio: float = 0.025
    style_split_pad_x_ratio: float = 0.01
    style_split_pad_y_ratio: float = 0.08

    mixed_split_min_component_area: int = 4
    mixed_split_mark_height_ratio: float = 0.18
    mixed_split_mark_width_ratio: float = 0.22
    mixed_split_mark_attach_ratio: float = 0.35
    mixed_split_handwriting_height_ratio: float = 0.36
    mixed_split_handwriting_width_ratio: float = 0.55
    mixed_split_handwriting_area_factor: float = 1.70
    mixed_split_printed_max_height_ratio: float = 0.42
    mixed_split_printed_max_width_ratio: float = 0.75
    mixed_split_merge_gap_ratio: float = 0.035
    mixed_split_max_runs: int = 3
    mixed_split_min_run_width_ratio: float = 0.06
    mixed_split_min_run_width_px: int = 18
    mixed_split_pad_x_ratio: float = 0.03
    mixed_split_pad_y_ratio: float = 0.12

    form_split_min_gap_ratio: float = 0.045
    form_split_min_gap_px: int = 6
    form_split_separator_window_ratio: float = 0.75
    form_split_min_mark_count: int = 2
    form_split_mark_min_y_ratio: float = 0.25
    form_split_min_prefix_ratio: float = 0.10
    form_split_min_tail_ratio: float = 0.12
    form_split_tail_height_factor: float = 1.25
    form_split_tail_y_delta_ratio: float = 0.18

    qwen_enabled: bool = False
    qwen_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_max_calls_per_page: int | None = None
    qwen_min_confidence_to_accept: float = 0.80
    qwen_verify_all_lines: bool = False
    paddle_lang: str = "en"

    @classmethod
    def from_env(cls) -> "OCRConfig":
        config = cls()
        if os.getenv("OCR_CHECKPOINT"):
            config.checkpoint = os.environ["OCR_CHECKPOINT"]
        if os.getenv("OCR_CHARSET_FILE"):
            config.charset_file = os.environ["OCR_CHARSET_FILE"]
        if os.getenv("OCR_OUTPUT_DIR"):
            config.out = os.environ["OCR_OUTPUT_DIR"]
        if os.getenv("OCR_PADDLE_DEVICE"):
            config.paddle_device = os.environ["OCR_PADDLE_DEVICE"]
        if os.getenv("OCR_TORCH_DEVICE"):
            config.device = os.environ["OCR_TORCH_DEVICE"]
        if os.getenv("OCR_QWEN_ENABLED"):
            config.qwen_enabled = os.environ["OCR_QWEN_ENABLED"].strip() == "1"
        return config

    def with_overrides(self, **overrides: Any) -> "OCRConfig":
        values = asdict(self)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return OCRConfig(**values)

    def to_namespace(self) -> Namespace:
        values = asdict(self)
        values["checkpoint"] = str(project_path(values["checkpoint"]))
        values["charset_file"] = str(project_path(values["charset_file"]))
        values["out"] = str(project_path(values["out"]))
        if values["det_model_dir"]:
            values["det_model_dir"] = str(project_path(values["det_model_dir"]))
        if values["qwen_max_calls_per_page"] is None:
            values["qwen_max_calls_per_page"] = 16 if values["mode"] == "ACCURATE" else 12
        if values["mode"] == "FAST":
            values["qwen_enabled"] = False
        return Namespace(**values)

    def validate_static_paths(self) -> None:
        checkpoint = project_path(self.checkpoint)
        charset = project_path(self.charset_file)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if not charset.exists():
            raise FileNotFoundError(f"Charset file not found: {charset}")

