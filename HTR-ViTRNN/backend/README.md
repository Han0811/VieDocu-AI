# Backend OCR

Backend entrypoints for the document OCR pipeline.

## CLI smoke test

```bash
/home/mpeclab/torch-env/bin/python -m backend.cli \
  --source ../img \
  --out backend_runs/smoke \
  --paddle-device cpu
```

## API

Install API-only dependencies:

```bash
/home/mpeclab/torch-env/bin/pip install -r requirements-backend.txt
```

Run server:

```bash
./run/backend_api.sh
```

Endpoints:

- `GET /health`: runtime readiness, checkpoint and charset paths.
- `GET /config`: active OCR defaults.
- `POST /ocr`: multipart upload field `file`, optional form fields `mode`, `paddle_device`, `qwen_enabled`.

Main outputs per job:

- `output/texts/*.txt`: final line text.
- `output/lines/*.tsv`: `line_id`, `text`.
- `output/lines/*.json`: clean line records with ordered parts.
- `output/regions/*.tsv`: printed/handwriting regions and crop paths.
- `output/debug_boxes/*_style_regions.jpg`: visual style-region debug.

