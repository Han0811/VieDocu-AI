# Document OCR Production Build Guide

Ngày cập nhật: 2026-06-13
Mục tiêu: biến backend Document OCR hiện tại thành sản phẩm hoàn chỉnh có giao diện web, hỗ trợ upload ảnh/PDF, xử lý OCR bằng GPU server, deploy qua domain, Docker, CI/CD.

---

# 0. Bối cảnh sản phẩm hiện tại

Project hiện đã có backend OCR inference chạy được.

Các thành phần đã tồn tại:

```text
backend/
  config.py
  service.py
  api.py
  cli.py
  README.md

document_ocr.py
qwen.py

run/
  document_ocr.sh
  backend_api.sh

model/
  HTR_ViTRNN.py
  resnet18.py

utils/
  utils.py

output/
  viet_lr1e4_64_layers_1024_dim/
    best_CER.pth
    viet_hf_charset.json

requirements-backend.txt
```

Backend hiện có:

```text
GET  /health
GET  /config
POST /ocr
```

`POST /ocr` hiện đang xử lý đồng bộ:

```text
upload image
→ process_pages()
→ trả JSON kết quả OCR
```

Output OCR hiện đã có:

```text
texts/<page>.txt
texts_raw/<page>.txt
lines/<page>.tsv
lines/<page>.json
regions/<page>.tsv
regions/<page>.json
metadata/<page>.json
debug_boxes/<page>_style_regions.jpg
debug_boxes/<page>_final_lines.jpg
line_crops/<page>/*.png
```

Frontend cần ưu tiên dùng:

```text
pages[].text
pages[].lines
pages[].files.debug_style_regions
```

---

# 1. Mục tiêu sản phẩm production

Sản phẩm cuối cần có:

```text
1. Web UI để upload ảnh hoặc PDF.
2. Backend nhận file ảnh/PDF.
3. Nếu là PDF, convert từng page thành image.
4. Tạo OCR job bất đồng bộ.
5. Worker xử lý OCR từng page.
6. Frontend poll trạng thái job.
7. Khi xong, frontend hiển thị:
   - text cuối
   - danh sách dòng OCR
   - ảnh debug box
   - regions printed/handwriting
   - crop nếu cần
8. Cho phép download:
   - TXT
   - JSON
   - ZIP output
   - DOCX nếu bổ sung sau
9. Deploy production bằng Docker Compose.
10. Máy RTX 5070 Ti làm GPU server.
11. Public qua domain bằng Cloudflare Tunnel.
12. CI/CD bằng GitHub Actions self-hosted runner.
```

---

# 2. Kiến trúc production đề xuất

## 2.1 Kiến trúc tổng thể

```text
User Browser
    ↓
https://ocr.your-domain.com
    ↓
Cloudflare DNS + Tunnel
    ↓
cloudflared container
    ↓
frontend container
    ↓
backend api container
    ↓
redis queue
    ↓
ocr worker container using RTX 5070 Ti
    ↓
backend_runs/<job_id>/output/
    ↓
PostgreSQL metadata
```

## 2.2 Container production

Cần có các service:

```text
frontend       Next.js/React UI
api            FastAPI API
worker         OCR worker dùng GPU
redis          job queue
postgres       job metadata
cloudflared    expose domain ra Internet
```

Optional sau này:

```text
minio          object storage nếu muốn tách file khỏi local disk
nginx          chỉ cần nếu không dùng Next.js riêng hoặc không dùng Cloudflare Tunnel
prometheus     monitoring
grafana        dashboard
uptime-kuma    uptime monitor
```

Giai đoạn đầu nên tránh phức tạp:

```text
Không dùng Kubernetes.
Không dùng microservice quá sớm.
Không dùng GPU VPS.
Không dùng Qwen mặc định.
```

---

# 3. Repo structure production mong muốn

Agent cần refactor repo thành cấu trúc này:

```text
document-ocr-product/
│
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── service.py
│   ├── cli.py
│   ├── schemas.py
│   ├── jobs.py
│   ├── storage.py
│   ├── pdf.py
│   ├── security.py
│   ├── database.py
│   └── README.md
│
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── jobs/
│   │   │   └── [jobId]/
│   │   │       └── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── UploadPanel.tsx
│   │   ├── JobStatus.tsx
│   │   ├── OCRResultViewer.tsx
│   │   ├── LineTable.tsx
│   │   ├── DebugImageViewer.tsx
│   │   ├── PageTabs.tsx
│   │   └── DownloadButtons.tsx
│   └── lib/
│       └── api.ts
│
├── model/
│   ├── HTR_ViTRNN.py
│   └── resnet18.py
│
├── utils/
│   └── utils.py
│
├── output/
│   └── viet_lr1e4_64_layers_1024_dim/
│       ├── best_CER.pth
│       └── viet_hf_charset.json
│
├── run/
│   ├── backend_api.sh
│   └── document_ocr.sh
│
├── scripts/
│   ├── deploy.sh
│   ├── backup.sh
│   ├── restore.sh
│   ├── smoke_api.sh
│   └── clean_runs.sh
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker.gpu
│   └── Dockerfile.frontend
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── backend_runs/
│   └── .gitkeep
│
├── requirements-backend.txt
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
├── .gitignore
└── README.md
```

---

# 4. Quy tắc quan trọng khi coding

Agent phải tuân thủ:

```text
1. Không phá code OCR đang chạy được.
2. Không đổi contract output cũ nếu không cần.
3. Backend service vẫn gọi được DocumentOCRService.
4. CLI smoke test vẫn phải chạy.
5. API mới phải support ảnh và PDF.
6. Job async phải là default cho frontend.
7. Endpoint sync /ocr có thể giữ để debug.
8. Qwen mặc định false.
9. Paddle device mặc định đọc từ env.
10. Model checkpoint và charset không hardcode tuyệt đối nếu đã có config.
11. Không commit .env thật.
12. Không commit backend_runs thật.
13. Không commit file PDF/image test lớn.
14. Output phải có đường dẫn file có thể download qua API.
```

---

# 5. Backend production design

## 5.1 Backend modes

Backend cần hỗ trợ 2 kiểu xử lý:

### Sync mode

Dùng cho smoke test/dev:

```text
POST /ocr
```

Input:

```text
file
mode
paddle_device
qwen_enabled
```

Output:

```text
job_id
output_dir
page_count
pages[]
config
```

### Async mode

Dùng cho frontend/production:

```text
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/result
GET  /api/jobs/{job_id}/files/{path}
GET  /api/jobs/{job_id}/download/zip
DELETE /api/jobs/{job_id}
```

Luồng async:

```text
frontend upload file
→ API save input file
→ API insert job metadata
→ API push job_id vào Redis
→ API trả job_id ngay
→ worker lấy job_id
→ worker xử lý ảnh/PDF
→ worker update progress
→ frontend poll status
→ frontend render result
```

---

# 6. API contract mới

## 6.1 Health

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "ready": true,
  "runtime": {
    "checkpoint_exists": true,
    "charset_exists": true,
    "paddle_device": "gpu:0",
    "qwen_enabled_default": false
  }
}
```

---

## 6.2 Config

```http
GET /config
```

Response giữ giống hiện tại nhưng không trả secret.

---

## 6.3 Submit OCR job

```http
POST /api/jobs
Content-Type: multipart/form-data
```

Form fields:

```text
file: image/pdf
mode: FAST | BALANCED | ACCURATE
paddle_device: cpu | gpu:0
qwen_enabled: true | false
```

Response:

```json
{
  "job_id": "20260613_153000_abcd1234",
  "status": "queued",
  "message": "Job queued successfully",
  "created_at": "2026-06-13T15:30:00+07:00"
}
```

Rules:

```text
- Accept image: jpg, jpeg, png, webp, tif, tiff.
- Accept PDF: pdf.
- Reject unknown file type.
- Reject file larger than MAX_UPLOAD_MB.
- Store original file at backend_runs/<job_id>/input/<original_filename>.
- Create backend_runs/<job_id>/output/.
```

---

## 6.4 Get job status

```http
GET /api/jobs/{job_id}
```

Response while queued:

```json
{
  "job_id": "20260613_153000_abcd1234",
  "status": "queued",
  "progress": 0,
  "stage": "waiting",
  "created_at": "2026-06-13T15:30:00+07:00",
  "updated_at": "2026-06-13T15:30:00+07:00"
}
```

Response while processing:

```json
{
  "job_id": "20260613_153000_abcd1234",
  "status": "processing",
  "progress": 45,
  "stage": "ocr_page_2_of_5",
  "page_count": 5,
  "current_page": 2,
  "created_at": "2026-06-13T15:30:00+07:00",
  "updated_at": "2026-06-13T15:30:40+07:00"
}
```

Response when done:

```json
{
  "job_id": "20260613_153000_abcd1234",
  "status": "done",
  "progress": 100,
  "stage": "completed",
  "page_count": 5,
  "result_url": "/api/jobs/20260613_153000_abcd1234/result",
  "download_zip_url": "/api/jobs/20260613_153000_abcd1234/download/zip",
  "created_at": "2026-06-13T15:30:00+07:00",
  "updated_at": "2026-06-13T15:32:20+07:00"
}
```

Response when failed:

```json
{
  "job_id": "20260613_153000_abcd1234",
  "status": "failed",
  "progress": 0,
  "stage": "failed",
  "error": "Cannot process PDF page 3",
  "created_at": "2026-06-13T15:30:00+07:00",
  "updated_at": "2026-06-13T15:31:10+07:00"
}
```

---

## 6.5 Get job result

```http
GET /api/jobs/{job_id}/result
```

Response:

```json
{
  "job_id": "20260613_153000_abcd1234",
  "status": "done",
  "page_count": 2,
  "pages": [
    {
      "page_index": 0,
      "page_name": "page_001",
      "text": "....",
      "lines": [
        {
          "line_id": 21,
          "text": "Số tiền đã chuyển: 500.000000(ND",
          "raw_text": "Số tiền đã chuyển: 500.000000(ND",
          "line_crop_path": "/api/jobs/<job_id>/files/line_crops/page_001/line_021_full.png",
          "box_xyxy": [100, 200, 1200, 250],
          "region_types": ["printed", "handwriting"],
          "parts": [
            {
              "part_id": 0,
              "region_id": 21,
              "region_type": "printed",
              "text": "Số tiền đã chuyển:",
              "raw_text": "Số tiền đã chuyển:",
              "crop_path": "/api/jobs/<job_id>/files/line_crops/page_001/line_021_part_00.png",
              "box_xyxy": [100, 200, 500, 250],
              "local_box_xyxy": [0, 0, 400, 50],
              "dropped": false
            }
          ]
        }
      ],
      "files": {
        "text": "/api/jobs/<job_id>/files/texts/page_001.txt",
        "lines_json": "/api/jobs/<job_id>/files/lines/page_001.json",
        "regions_json": "/api/jobs/<job_id>/files/regions/page_001.json",
        "metadata": "/api/jobs/<job_id>/files/metadata/page_001.json",
        "debug_style_regions": "/api/jobs/<job_id>/files/debug_boxes/page_001_style_regions.jpg",
        "debug_final_lines": "/api/jobs/<job_id>/files/debug_boxes/page_001_final_lines.jpg"
      }
    }
  ]
}
```

Important:

```text
Không trả path local kiểu /home/... cho frontend.
Phải map local file path thành API URL.
```

---

## 6.6 Serve output files

```http
GET /api/jobs/{job_id}/files/{file_path:path}
```

Ví dụ:

```text
GET /api/jobs/abc/files/debug_boxes/page_001_style_regions.jpg
GET /api/jobs/abc/files/texts/page_001.txt
GET /api/jobs/abc/files/lines/page_001.json
```

Security rule:

```text
- Chỉ cho đọc file bên trong backend_runs/<job_id>/output/.
- Không cho path traversal: ../, absolute path, symlink escape.
- Nếu file không tồn tại → 404.
```

---

## 6.7 Download ZIP

```http
GET /api/jobs/{job_id}/download/zip
```

Return:

```text
application/zip
```

ZIP content:

```text
texts/
lines/
regions/
metadata/
debug_boxes/
line_crops/
result.json
```

---

## 6.8 Delete job

```http
DELETE /api/jobs/{job_id}
```

Dùng để frontend/admin xóa job.

Response:

```json
{
  "job_id": "abc",
  "deleted": true
}
```

---

# 7. PDF support

## 7.1 Mục tiêu

Backend phải nhận được:

```text
.jpg
.jpeg
.png
.webp
.tif
.tiff
.pdf
```

Nếu input là ảnh:

```text
input image
→ process_pages([image_path])
```

Nếu input là PDF:

```text
input pdf
→ convert mỗi page thành image
→ process_pages([page_001.png, page_002.png, ...])
```

## 7.2 Thư mục PDF pages

Với job ID:

```text
backend_runs/<job_id>/
  input/
    original.pdf
  pages/
    page_001.png
    page_002.png
    page_003.png
  output/
```

## 7.3 Module cần tạo

Tạo file:

```text
backend/pdf.py
```

Nhiệm vụ:

```python
def is_pdf(path: Path) -> bool:
    ...

def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 220,
    max_pages: int | None = None
) -> list[Path]:
    ...
```

Yêu cầu:

```text
- Dùng pypdfium2 hoặc pdf2image.
- Khuyến nghị pypdfium2 vì dễ đóng gói hơn.
- Output image nên là PNG hoặc JPEG chất lượng cao.
- DPI mặc định 220 hoặc 300.
- Với hợp đồng scan, DPI 220 thường đủ để detect/OCR.
- Có giới hạn MAX_PDF_PAGES.
- Nếu PDF quá nhiều trang, reject hoặc xử lý theo config.
```

## 7.4 Config PDF

Thêm vào `backend/config.py`:

```python
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
PDF_DPI = int(os.getenv("PDF_DPI", "220"))
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".pdf"
}
```

## 7.5 Dependency

Thêm vào `requirements-backend.txt`:

```text
fastapi
uvicorn[standard]
python-multipart
redis
rq
sqlalchemy
psycopg2-binary
pydantic
pydantic-settings
pypdfium2
Pillow
aiofiles
```

Nếu dùng `pdf2image`, container phải cài `poppler-utils`.

---

# 8. Async job queue design

## 8.1 Tại sao phải async?

Không để frontend chờ request OCR quá lâu.

Không làm:

```text
POST /ocr → xử lý PDF 20 trang → chờ vài phút → trả result
```

Phải làm:

```text
POST /api/jobs → trả job_id ngay
worker xử lý nền
frontend poll /api/jobs/{job_id}
```

## 8.2 Redis queue

Dùng Redis + RQ cho đơn giản.

Tạo file:

```text
backend/jobs.py
```

Chức năng:

```python
def enqueue_ocr_job(job_id: str) -> None:
    ...

def run_ocr_job(job_id: str) -> None:
    ...
```

## 8.3 Job status storage

Dùng PostgreSQL hoặc JSON file. Production nên dùng PostgreSQL.

Bảng `ocr_jobs`:

```sql
CREATE TABLE IF NOT EXISTS ocr_jobs (
    job_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    stage TEXT,
    page_count INTEGER,
    current_page INTEGER,
    output_dir TEXT,
    error TEXT,
    mode TEXT,
    paddle_device TEXT,
    qwen_enabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);
```

Trạng thái hợp lệ:

```text
queued
processing
done
failed
deleted
```

## 8.4 Worker flow

Worker xử lý:

```text
1. Load job metadata từ DB.
2. Update status = processing.
3. Kiểm tra file input.
4. Nếu PDF:
   - convert PDF to images.
   - update page_count.
5. Nếu image:
   - page_count = 1.
6. Gọi DocumentOCRService.
7. Ghi output.
8. Build result.json tổng.
9. Update status = done.
10. Nếu lỗi:
   - status = failed.
   - lưu error.
```

## 8.5 Progress update

Progress đơn giản:

```text
0    queued
5    input_saved
10   pdf_converted
20   ocr_started
20-90 per page OCR
95   result_packaging
100  done
```

Nếu chưa sửa được `process_pages()` để callback progress, cứ update theo stage lớn trước.

---

# 9. Backend implementation checklist

Agent cần làm theo thứ tự:

## 9.1 Tạo schemas

Tạo:

```text
backend/schemas.py
```

Gồm:

```python
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
```

## 9.2 Tạo storage helper

Tạo:

```text
backend/storage.py
```

Chức năng:

```python
def create_job_dirs(job_id: str) -> dict[str, Path]:
    ...

def save_upload_file(upload_file: UploadFile, dst: Path) -> None:
    ...

def safe_output_file_path(job_id: str, relative_path: str) -> Path:
    ...

def build_file_url(job_id: str, output_relative_path: str) -> str:
    ...

def zip_output_dir(job_id: str) -> Path:
    ...
```

## 9.3 Tạo database helper

Tạo:

```text
backend/database.py
```

Có thể dùng SQLAlchemy.

Chức năng:

```python
def init_db() -> None:
    ...

def create_job(...) -> None:
    ...

def update_job(job_id: str, **kwargs) -> None:
    ...

def get_job(job_id: str) -> dict | None:
    ...

def list_jobs(limit: int = 50) -> list[dict]:
    ...
```

Nếu muốn nhanh hơn trong MVP, có thể dùng SQLite trước:

```text
DATABASE_URL=sqlite:///./backend_runs/jobs.sqlite3
```

Production sau đổi sang Postgres:

```text
DATABASE_URL=postgresql://ocr_user:ocr_password@postgres:5432/ocr_db
```

## 9.4 Sửa backend/api.py

Giữ endpoint cũ:

```text
GET /health
GET /config
POST /ocr
```

Thêm endpoint mới:

```text
POST   /api/jobs
GET    /api/jobs
GET    /api/jobs/{job_id}
GET    /api/jobs/{job_id}/result
GET    /api/jobs/{job_id}/files/{file_path:path}
GET    /api/jobs/{job_id}/download/zip
DELETE /api/jobs/{job_id}
```

## 9.5 Sửa backend/service.py

Cần thêm method:

```python
class DocumentOCRService:
    def run_file(
        self,
        input_path: Path,
        output_dir: Path,
        mode: str,
        paddle_device: str,
        qwen_enabled: bool,
    ) -> dict:
        ...
```

Logic:

```text
if input is pdf:
    convert to page images
else:
    use image as one page

call existing process_pages()
collect outputs
return frontend-compatible result
```

## 9.6 Không phá CLI

Lệnh này vẫn phải chạy:

```bash
python -m backend.cli \
  --source ../img/a42a7b9b1f5d9e03c74c.jpg \
  --out backend_runs/smoke_a42 \
  --paddle-device cpu
```

---

# 10. Frontend design

## 10.1 Framework

Dùng:

```text
Next.js + TypeScript + TailwindCSS
```

Nếu muốn đơn giản hơn có thể dùng Vite React, nhưng Next.js hợp deploy production hơn.

## 10.2 Frontend pages

Cần có:

```text
/
  Upload page

/jobs/[jobId]
  Job result page
```

Optional:

```text
/history
  danh sách job gần đây

/settings
  chọn mode mặc định, qwen, paddle device
```

## 10.3 UI layout chính

Trang `/`:

```text
Header:
  Document OCR

Main:
  Upload card
    - drag & drop file
    - choose file
    - file type hint: JPG, PNG, WEBP, TIFF, PDF
    - mode select: FAST / BALANCED / ACCURATE
    - qwen toggle: off by default
    - submit button

Right/Bottom:
  status panel
    - current upload
    - job id
    - progress
```

Trang `/jobs/[jobId]`:

```text
Top:
  job status
  progress bar
  download buttons

Body:
  page tabs:
    Page 1 | Page 2 | ...

For selected page:
  Left:
    debug image viewer
      - style regions
      - final lines
  Right:
    OCR text viewer
    line table
    regions table optional
```

## 10.4 Components cần tạo

```text
components/UploadPanel.tsx
components/JobStatus.tsx
components/OCRResultViewer.tsx
components/PageTabs.tsx
components/DebugImageViewer.tsx
components/LineTable.tsx
components/DownloadButtons.tsx
components/ModeSelector.tsx
components/QwenToggle.tsx
```

## 10.5 Frontend API client

Tạo:

```text
frontend/lib/api.ts
```

Functions:

```typescript
export async function submitJob(file: File, options: SubmitOptions): Promise<JobCreateResponse>

export async function getJobStatus(jobId: string): Promise<JobStatusResponse>

export async function getJobResult(jobId: string): Promise<OCRResultResponse>

export function getJobZipUrl(jobId: string): string
```

Environment:

```text
NEXT_PUBLIC_API_BASE_URL=/api-backend
```

Nếu frontend và backend cùng domain, có thể dùng reverse proxy hoặc rewrite.

Với Cloudflare Tunnel có thể expose frontend container và frontend gọi backend bằng internal URL thông qua Next.js server, hoặc đơn giản cho frontend gọi:

```text
https://ocr-api.your-domain.com
```

Khuyến nghị MVP:

```text
ocr.your-domain.com       frontend
ocr-api.your-domain.com   backend
```

Hoặc đơn giản hơn:

```text
ocr.your-domain.com       frontend
ocr.your-domain.com/api   backend reverse path
```

Giai đoạn đầu có thể dùng 2 hostname cho dễ debug.

---

# 11. Frontend behavior

## 11.1 Upload flow

```text
1. User chọn file.
2. Frontend validate extension.
3. Frontend validate size.
4. User chọn mode.
5. User bật/tắt Qwen.
6. Click OCR.
7. Frontend POST /api/jobs.
8. Nhận job_id.
9. Redirect /jobs/<job_id>.
10. Poll status mỗi 1-2 giây.
11. Khi status=done, fetch result.
12. Render result.
```

## 11.2 Polling

Pseudo-code:

```typescript
useEffect(() => {
  const timer = setInterval(async () => {
    const status = await getJobStatus(jobId)
    setStatus(status)

    if (status.status === "done") {
      clearInterval(timer)
      const result = await getJobResult(jobId)
      setResult(result)
    }

    if (status.status === "failed") {
      clearInterval(timer)
      setError(status.error)
    }
  }, 1500)

  return () => clearInterval(timer)
}, [jobId])
```

## 11.3 Result viewer

Hiển thị:

```text
- Full text
- Copy text button
- Download TXT
- Download JSON
- Download ZIP
- Lines table
- Debug image
```

Line table columns:

```text
line_id
text
region_types
box
actions
```

Actions:

```text
- copy line
- view crop
- view parts
```

---

# 12. Docker production

## 12.1 .env.example

Tạo `.env.example`:

```env
APP_ENV=production
TZ=Asia/Ho_Chi_Minh

# API
API_HOST=0.0.0.0
API_PORT=8000
PUBLIC_API_BASE_URL=https://ocr-api.your-domain.com

# Frontend
NEXT_PUBLIC_API_BASE_URL=https://ocr-api.your-domain.com

# OCR
OCR_RUNS_DIR=/app/backend_runs
OCR_MODEL_DIR=/app/output/viet_lr1e4_64_layers_1024_dim
OCR_CHECKPOINT=/app/output/viet_lr1e4_64_layers_1024_dim/best_CER.pth
OCR_CHARSET=/app/output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json
OCR_PADDLE_DEVICE=gpu:0
OCR_QWEN_ENABLED=false

# Upload
MAX_UPLOAD_MB=100
MAX_PDF_PAGES=50
PDF_DPI=220

# DB
POSTGRES_USER=ocr_user
POSTGRES_PASSWORD=change_me
POSTGRES_DB=ocr_db
DATABASE_URL=postgresql://ocr_user:change_me@postgres:5432/ocr_db

# Redis
REDIS_URL=redis://redis:6379/0

# Cloudflare
CLOUDFLARE_TUNNEL_TOKEN=change_me

# Security
API_KEY=change_me
ENABLE_API_KEY=false
```

## 12.2 Dockerfile API

Tạo:

```text
docker/Dockerfile.api
```

```dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    poppler-utils \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY backend ./backend
COPY document_ocr.py ./document_ocr.py
COPY qwen.py ./qwen.py
COPY model ./model
COPY utils ./utils
COPY output ./output
COPY run ./run

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 12.3 Dockerfile Worker GPU

Tạo:

```text
docker/Dockerfile.worker.gpu
```

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libgl1 \
    poppler-utils \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip3 install --no-cache-dir -r requirements-backend.txt

COPY backend ./backend
COPY document_ocr.py ./document_ocr.py
COPY qwen.py ./qwen.py
COPY model ./model
COPY utils ./utils
COPY output ./output
COPY run ./run

CMD ["python3", "-m", "backend.worker"]
```

Agent cần tạo thêm:

```text
backend/worker.py
```

Worker dùng RQ hoặc loop Redis.

## 12.4 Dockerfile Frontend

Tạo:

```text
docker/Dockerfile.frontend
```

```dockerfile
FROM node:22-alpine AS deps

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

FROM node:22-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend ./
RUN npm run build

FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

COPY --from=builder /app ./

EXPOSE 3000

CMD ["npm", "start"]
```

## 12.5 docker-compose.yml

Tạo:

```yaml
services:
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    container_name: dococr_frontend
    restart: unless-stopped
    env_file:
      - .env
    depends_on:
      - api
    ports:
      - "3000:3000"

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    container_name: dococr_api
    restart: unless-stopped
    env_file:
      - .env
    depends_on:
      - redis
      - postgres
    volumes:
      - ./backend_runs:/app/backend_runs
      - ./output:/app/output
    ports:
      - "8000:8000"

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker.gpu
    container_name: dococr_worker
    restart: unless-stopped
    env_file:
      - .env
    depends_on:
      - redis
      - postgres
    volumes:
      - ./backend_runs:/app/backend_runs
      - ./output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    container_name: dococr_redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16-alpine
    container_name: dococr_postgres
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - postgres_data:/var/lib/postgresql/data

  cloudflared:
    image: cloudflare/cloudflared:latest
    container_name: dococr_cloudflared
    restart: unless-stopped
    command: tunnel --no-autoupdate run --token ${CLOUDFLARE_TUNNEL_TOKEN}
    depends_on:
      - frontend
      - api

volumes:
  redis_data:
  postgres_data:
```

Note:

```text
- Với Docker Compose hiện đại, GPU reservation có thể dùng được.
- Nếu GPU không chạy, test bằng docker run --gpus all trước.
- Nếu Compose không cấp GPU đúng, thêm runtime/nvidia config theo môi trường server.
```

---

# 13. Cloudflare Tunnel + Domain

## 13.1 Mô hình deploy

Máy RTX 5070 Ti đặt ở nhà/lab:

```text
Không cần mở port router.
Không cần IP tĩnh.
Không cần VPS GPU.
```

Cloudflare Tunnel expose:

```text
ocr.your-domain.com      → http://frontend:3000
ocr-api.your-domain.com  → http://api:8000
```

## 13.2 Cloudflare config

Trong Cloudflare Dashboard:

```text
Zero Trust
→ Networks
→ Tunnels
→ Create Tunnel
→ Cloudflared
```

Tạo tunnel:

```text
name: document-ocr-server
```

Lấy token đưa vào `.env`:

```env
CLOUDFLARE_TUNNEL_TOKEN=...
```

Public hostnames:

```text
Hostname: ocr.your-domain.com
Service:  http://frontend:3000

Hostname: ocr-api.your-domain.com
Service:  http://api:8000
```

Nếu chỉ muốn một domain:

```text
ocr.your-domain.com      → frontend
ocr.your-domain.com/api  → backend
```

Nhưng cách 2 hostname dễ hơn cho MVP.

---

# 14. CI/CD bằng GitHub self-hosted runner

Vì máy GPU nằm sau NAT, GitHub Actions SSH từ ngoài vào sẽ khó.

Dùng self-hosted runner cài trực tiếp trên máy RTX 5070 Ti.

## 14.1 Cài runner

Trong GitHub repo:

```text
Settings
→ Actions
→ Runners
→ New self-hosted runner
→ Linux x64
```

Chạy lệnh GitHub đưa trên máy server.

Nên chạy runner như service.

## 14.2 Workflow deploy

Tạo:

```text
.github/workflows/deploy.yml
```

```yaml
name: Deploy Document OCR

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: self-hosted

    steps:
      - name: Checkout source
        uses: actions/checkout@v4

      - name: Check required files
        run: |
          test -f document_ocr.py
          test -f qwen.py
          test -f backend/api.py
          test -f backend/service.py
          test -f requirements-backend.txt
          test -f output/viet_lr1e4_64_layers_1024_dim/best_CER.pth
          test -f output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json

      - name: Keep production env
        run: |
          if [ -f /home/ocr/.env ]; then
            cp /home/ocr/.env .env
          elif [ ! -f .env ]; then
            cp .env.example .env
          fi

      - name: Python compile check
        run: |
          python3 -m py_compile \
            document_ocr.py \
            qwen.py \
            backend/config.py \
            backend/service.py \
            backend/api.py \
            backend/cli.py || true

      - name: Build and restart containers
        run: |
          docker compose up -d --build
          docker image prune -f

      - name: Health check API
        run: |
          sleep 10
          curl -f http://127.0.0.1:8000/health

      - name: Health check Frontend
        run: |
          curl -f http://127.0.0.1:3000 || true
```

Important:

```text
- .env thật nằm ở /home/ocr/.env.
- Không commit .env thật.
- Nếu checkpoint lớn không muốn commit, mount volume output/ trên server.
```

---

# 15. Server setup RTX 5070 Ti

## 15.1 OS

Khuyến nghị:

```text
Ubuntu 22.04 LTS hoặc Ubuntu 24.04 LTS
```

Không khuyến nghị Windows cho production lâu dài.

## 15.2 Cài Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl git ufw
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Logout/login lại.

## 15.3 Cài NVIDIA driver

Kiểm tra:

```bash
nvidia-smi
```

Nếu chưa có driver, cài driver phù hợp GPU.

## 15.4 Cài NVIDIA Container Toolkit

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Nếu thấy RTX 5070 Ti là đạt.

## 15.5 Clone repo

```bash
mkdir -p /home/ocr
cd /home/ocr
git clone <repo-url> document-ocr-product
cd document-ocr-product
cp .env.example .env
nano .env
```

## 15.6 Run production

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
docker compose logs -f worker
```

Test local:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:3000
```

---

# 16. Security checklist

## 16.1 API key

Thêm middleware trong:

```text
backend/security.py
```

Logic:

```text
Nếu ENABLE_API_KEY=false → bỏ qua.
Nếu ENABLE_API_KEY=true:
  yêu cầu header X-API-Key.
```

Các endpoint cần bảo vệ:

```text
POST /api/jobs
GET /api/jobs
GET /api/jobs/{job_id}
DELETE /api/jobs/{job_id}
```

Có thể không bảo vệ:

```text
GET /health
```

Hoặc chỉ expose health đơn giản.

## 16.2 File upload security

Validate:

```text
- extension
- MIME type nếu có
- size
- PDF page count
- path traversal
```

Reject:

```text
- .exe
- .sh
- .py
- .zip nếu chưa support
- PDF quá nhiều trang
- file quá lớn
```

## 16.3 Không public service nội bộ

Không expose ra Internet:

```text
redis
postgres
docker socket
minio console nếu có
```

Chỉ public qua Cloudflare Tunnel:

```text
frontend
api
```

## 16.4 CORS

Nếu dùng 2 domain:

```text
ocr.your-domain.com
ocr-api.your-domain.com
```

Trong FastAPI thêm CORS:

```python
allow_origins=[
    "https://ocr.your-domain.com"
]
```

Dev:

```python
"http://localhost:3000"
```

---

# 17. Backup

Tạo:

```text
scripts/backup.sh
```

```bash
#!/usr/bin/env bash
set -e

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT=/home/ocr/backups
BACKUP_DIR=$BACKUP_ROOT/$DATE

mkdir -p "$BACKUP_DIR"

echo "[1/3] Backup postgres..."
docker exec dococr_postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > "$BACKUP_DIR/postgres.sql" || true

echo "[2/3] Backup backend_runs metadata/output..."
tar -czf "$BACKUP_DIR/backend_runs.tar.gz" backend_runs || true

echo "[3/3] Backup env and model refs..."
cp .env "$BACKUP_DIR/env.copy" || true
tar -czf "$BACKUP_DIR/output_model.tar.gz" output/viet_lr1e4_64_layers_1024_dim || true

echo "Backup saved to $BACKUP_DIR"
```

Note:

```text
Không share env.copy.
Không upload env.copy lên GitHub.
```

---

# 18. Cleanup old jobs

Tạo:

```text
scripts/clean_runs.sh
```

```bash
#!/usr/bin/env bash
set -e

DAYS=${1:-14}

echo "Delete backend_runs older than $DAYS days"
find backend_runs -mindepth 1 -maxdepth 1 -type d -mtime +$DAYS -print
```

Ban đầu chỉ print. Sau khi chắc chắn mới thêm `-exec rm -rf`.

Production nên có config:

```env
RETENTION_DAYS=14
```

---

# 19. Monitoring tối thiểu

Cần có:

```text
GET /health
docker compose ps
docker compose logs
disk usage
GPU usage
queue length
job failed count
```

API `/health` nên trả:

```json
{
  "status": "ok",
  "ready": true,
  "gpu": {
    "available": true,
    "device": "gpu:0"
  },
  "queue": {
    "queued": 0
  },
  "storage": {
    "runs_dir_exists": true
  }
}
```

Có thể thêm sau:

```text
Uptime Kuma
Prometheus
Grafana
Loki
```

---

# 20. Test plan

## 20.1 Unit/smoke test backend

Commands phải pass:

```bash
python -m py_compile \
  document_ocr.py \
  qwen.py \
  backend/config.py \
  backend/service.py \
  backend/api.py \
  backend/cli.py
```

```bash
python -m backend.cli --help
```

```bash
python -c "from backend.service import DocumentOCRService; s=DocumentOCRService(); print(s.inspect_runtime()['ready'])"
```

## 20.2 API test local

Run:

```bash
docker compose up -d --build
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Sync OCR image:

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -F "file=@sample.jpg" \
  -F "mode=BALANCED" \
  -F "paddle_device=cpu" \
  -F "qwen_enabled=false"
```

Async OCR image:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "file=@sample.jpg" \
  -F "mode=BALANCED" \
  -F "paddle_device=cpu" \
  -F "qwen_enabled=false"
```

Check job:

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

Get result:

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>/result
```

PDF test:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "file=@sample.pdf" \
  -F "mode=BALANCED" \
  -F "paddle_device=cpu" \
  -F "qwen_enabled=false"
```

## 20.3 Frontend test

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Test:

```text
- upload jpg
- upload png
- upload pdf 1 page
- upload pdf nhiều page
- poll status
- render lines
- render debug image
- download zip
```

## 20.4 Docker GPU test

```bash
docker compose exec worker nvidia-smi
```

Nếu lỗi:

```text
- kiểm tra NVIDIA driver.
- kiểm tra nvidia-container-toolkit.
- kiểm tra Docker runtime.
```

---

# 21. Acceptance criteria

Sản phẩm được coi là hoàn thành MVP production khi:

```text
1. Frontend upload được ảnh.
2. Frontend upload được PDF.
3. Backend tạo job async.
4. Worker xử lý job.
5. Frontend xem được progress.
6. Frontend hiển thị text cuối.
7. Frontend hiển thị line table.
8. Frontend hiển thị debug image.
9. Frontend download được ZIP output.
10. Docker Compose chạy được toàn bộ stack.
11. Cloudflare domain truy cập được từ ngoài mạng.
12. GitHub push main tự deploy trên máy GPU.
13. /health trả ready=true.
14. Không cần data train để chạy inference.
15. Qwen mặc định tắt.
16. Redis/Postgres không public.
17. File output không lộ path local.
18. PDF nhiều trang sinh đúng pages[].
```

---

# 22. Implementation order cho coding agent

Làm theo đúng thứ tự này.

## Phase 1: Backend async foundation

```text
1. Thêm backend/schemas.py.
2. Thêm backend/storage.py.
3. Thêm backend/pdf.py.
4. Thêm backend/database.py.
5. Thêm backend/jobs.py.
6. Thêm backend/worker.py.
7. Sửa backend/api.py thêm /api/jobs.
8. Giữ /ocr cũ.
9. Test ảnh sync.
10. Test ảnh async.
```

## Phase 2: PDF support

```text
1. Cài pypdfium2.
2. Implement convert_pdf_to_images().
3. Tích hợp vào service.
4. Test PDF 1 page.
5. Test PDF nhiều page.
6. Validate MAX_PDF_PAGES.
```

## Phase 3: File serving + result packaging

```text
1. Build result.json tổng trong output.
2. Map local path thành API URL.
3. Implement /files/{path}.
4. Implement /download/zip.
5. Test debug image mở được từ browser.
```

## Phase 4: Frontend MVP

```text
1. Tạo Next.js app.
2. Tạo UploadPanel.
3. Tạo job submit API client.
4. Tạo /jobs/[jobId].
5. Poll status.
6. Render result text.
7. Render line table.
8. Render debug image.
9. Add download buttons.
```

## Phase 5: Docker

```text
1. Viết Dockerfile.api.
2. Viết Dockerfile.worker.gpu.
3. Viết Dockerfile.frontend.
4. Viết docker-compose.yml.
5. Test local CPU.
6. Test GPU worker.
```

## Phase 6: Cloudflare + domain

```text
1. Mua domain.
2. Add domain vào Cloudflare.
3. Tạo Cloudflare Tunnel.
4. Add token vào .env.
5. Public ocr.domain.com → frontend.
6. Public ocr-api.domain.com → api.
7. Test upload từ mạng ngoài.
```

## Phase 7: CI/CD

```text
1. Cài GitHub self-hosted runner trên máy GPU.
2. Tạo deploy.yml.
3. Push main.
4. Kiểm tra tự build/restart.
5. Health check pass.
```

## Phase 8: Hardening

```text
1. API key.
2. CORS.
3. Upload size limit.
4. PDF page limit.
5. Cleanup old jobs.
6. Backup script.
7. Monitoring cơ bản.
```

---

# 23. Không làm trong MVP

Không làm các thứ sau ở MVP:

```text
- User login nhiều role.
- Payment.
- Kubernetes.
- Multi-GPU scheduling.
- Auto scale.
- Human correction editor phức tạp.
- Qwen sửa OCR.
- Train model trong production.
- Public MinIO console.
- Public database.
```

Có thể làm sau:

```text
- DOCX export.
- Search OCR text.
- Human review UI.
- Batch upload nhiều file.
- Admin dashboard.
- Usage analytics.
- OCR confidence visualization.
```

---

# 24. Prompt cho agent coding

Dùng prompt này để giao việc cho coding agent:

```text
Bạn đang làm việc trong repo Document OCR đã có backend OCR inference chạy được.

Mục tiêu của bạn là biến repo thành sản phẩm production có frontend, hỗ trợ upload ảnh/PDF, xử lý OCR async bằng worker, Docker Compose, Cloudflare Tunnel và CI/CD self-hosted runner.

Không được phá pipeline OCR hiện tại. Giữ lại endpoint sync /ocr và CLI smoke test. Thêm async job API mới dưới /api/jobs. PDF phải được convert thành ảnh trước khi gọi DocumentOCRService. Output cuối phải giữ contract hiện có gồm text, lines, regions, metadata, debug_boxes và line_crops.

Hãy triển khai theo các phase:

Phase 1:
- backend/schemas.py
- backend/storage.py
- backend/pdf.py
- backend/database.py
- backend/jobs.py
- backend/worker.py
- sửa backend/api.py thêm async job endpoints

Phase 2:
- support PDF bằng pypdfium2
- validate extension, size, max pages

Phase 3:
- serve output files an toàn
- generate result.json
- download ZIP

Phase 4:
- tạo frontend Next.js TypeScript
- upload panel
- job status polling
- result viewer
- line table
- debug image viewer
- download buttons

Phase 5:
- Dockerfile.api
- Dockerfile.worker.gpu
- Dockerfile.frontend
- docker-compose.yml
- .env.example

Phase 6:
- GitHub Actions deploy.yml cho self-hosted runner

Yêu cầu bắt buộc:
- Qwen mặc định false.
- Không cần data train.
- Checkpoint và charset lấy từ output/viet_lr1e4_64_layers_1024_dim/.
- Không commit .env thật.
- Không expose Redis/Postgres.
- Không trả path local cho frontend.
- Phải có curl test trong README.
- Phải có acceptance checklist.
```

---

# 25. Kết luận

Sản phẩm nên đi theo kiến trúc:

```text
Next.js frontend
+ FastAPI backend
+ Redis queue
+ OCR worker GPU
+ PostgreSQL metadata
+ Docker Compose
+ Cloudflare Tunnel
+ GitHub self-hosted runner
```

Đây là hướng hợp lý nhất vì:

```text
- Tận dụng được máy RTX 5070 Ti làm server.
- Không cần thuê GPU VPS đắt.
- Có domain truy cập mọi nơi.
- Có frontend cho người dùng.
- Có async job để xử lý PDF nhiều trang.
- Có output contract rõ cho debug và phát triển tiếp.
```
