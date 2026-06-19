from __future__ import annotations

import argparse
import json

from .config import OCRConfig
from .service import DocumentOCRService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend OCR smoke/production runner")
    parser.add_argument("--source", required=True, help="Input image or folder")
    parser.add_argument("--out", default=None, help="Output directory; defaults to backend_runs/<job_id>")
    parser.add_argument("--mode", choices=["FAST", "BALANCED", "ACCURATE"], default=None)
    parser.add_argument("--paddle-device", default=None)
    parser.add_argument("--device", default=None, help="PyTorch HTR device")
    parser.add_argument("--qwen-enabled", action="store_true")
    parser.add_argument("--qwen-max-calls-per-page", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--charset-file", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = OCRConfig.from_env().with_overrides(
        mode=args.mode,
        paddle_device=args.paddle_device,
        device=args.device,
        qwen_enabled=args.qwen_enabled,
        qwen_max_calls_per_page=args.qwen_max_calls_per_page,
        checkpoint=args.checkpoint,
        charset_file=args.charset_file,
    )
    service = DocumentOCRService(config)
    result = service.run(args.source, out=args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

