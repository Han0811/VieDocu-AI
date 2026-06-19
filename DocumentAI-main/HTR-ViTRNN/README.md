# HTR-ViTRNN Document OCR Production Stack

A complete production-ready pipeline for Vietnamese handwritten and printed document OCR, featuring real-time asynchronous background job processing, PDF document conversion support, dynamic web-based analysis dashboard, database tracking, and GPU worker orchestration.

---

## 🏗️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │     Next.js Frontend     │ (Port 3000)
                          └─────────────┬────────────┘
                                        │ (HTTP REST / JSON / Files)
                                        ▼
                          ┌──────────────────────────┐
                          │    FastAPI Gateway API   │ (Port 8000)
                          └─────────────┬────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │    SQLite/Postgres    │                     │      Redis Queue      │
     │   (Job Metadata DB)   │                     │   (Asynchronous RQ)   │
     └───────────────────────┘                     └───────────┬───────────┘
                                                               │
                                                               ▼
                                                   ┌───────────────────────┐
                                                   │    GPU OCR Worker     │ (RTX 5070 Ti)
                                                   └───────────────────────┘
```

- **Frontend**: Next.js (React 18 + TS + TailwindCSS). Includes a glassmorphic dashboard, upload panel, status polling, full text results view, page pagination, line crop previews, style region badges, and bulk ZIP exports.
- **Backend API**: FastAPI REST Gateway managing async job submissions, polling, output file serving, and ZIP packages.
- **Task Queue**: In-memory ThreadPoolExecutor for lightweight local development, automatically swapping to Redis & RQ (Redis Queue) in Docker production.
- **OCR Engine**: Original HTR-ViTRNN PyTorch model for handwriting, PaddleOCR for printed text, and Qwen-VL for OCR correction.

---

## ⚡ Local Quickstart (Without Docker)

You can run the backend and frontend locally using the host Python environment (`/home/mpeclab/torch-env`).

### 1. Start the Backend API
```bash
# Set environment variables
export DATABASE_URL="sqlite:///backend_runs/jobs.sqlite3"
export PORT=8000

# Run API server
/home/mpeclab/torch-env/bin/python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

### 2. Start the Backend Worker
In a new terminal:
```bash
/home/mpeclab/torch-env/bin/python -m backend.worker
```

### 3. Run the Frontend Dashboard
In a new terminal:
```bash
cd frontend
npm run dev
# Dashboard available at http://localhost:3000
```

---

## 🐳 Docker Deployment (Production GPU Stack)

Production deployments use Docker Compose and utilize host NVIDIA GPU acceleration.

### 1. Prerequisites
Ensure you have Docker, Docker Compose, and the **NVIDIA Container Toolkit** installed.

### 2. Run the Stack
```bash
# Build and launch all containers
docker compose up -d --build
```
This launches:
- `ocr_redis` on port 6379
- `ocr_api` on port 8000
- `ocr_worker` utilizing GPU:0 (`deploy.resources.reservations.devices`)
- `ocr_frontend` on port 3000

---

## ☁️ Cloudflare Tunnel & Domain Configuration

For public internet accessibility, configure a Cloudflare Tunnel. This bypasses the need for public IP forwarding and provides out-of-the-box SSL certificates.

### 1. Install cloudflared
On the host GPU server:
```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

### 2. Authenticate & Create Tunnel
```bash
cloudflared tunnel login
cloudflared tunnel create ocr-tunnel
```
This generates a Tunnel ID and a credentials JSON file in `~/.cloudflared/`.

### 3. Create Configuration File
Create `~/.cloudflared/config.yml`:
```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/mpeclab/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: ocr.vietnamese-documents.org
    service: http://localhost:3000
  - hostname: ocr-api.vietnamese-documents.org
    service: http://localhost:8000
  - service: http_status:404
```

### 4. Create DNS Records & Run
Route the domains through Cloudflare dashboard:
```bash
cloudflared tunnel route dns ocr-tunnel ocr.vietnamese-documents.org
cloudflared tunnel route dns ocr-tunnel ocr-api.vietnamese-documents.org

# Start the tunnel in the background
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

---

## 🛠️ Scripts & Maintenance

All utility scripts are located in `scripts/`:

- **Run Smoke Tests**:
  ```bash
  ./scripts/smoke_api.sh http://localhost:8000
  ```
- **Prune Old Runs**:
  Automatically deletes files and DB logs older than 7 days:
  ```bash
  ./scripts/clean_runs.sh 7
  ```
- **System Backups**:
  Backs up the SQLite database and job outputs:
  ```bash
  ./scripts/backup.sh
  ```
- **Manual Deploy**:
  ```bash
  ./scripts/deploy.sh
  ```

---

## 🔄 CI/CD Pipeline

The stack includes a Github Actions Workflow (`.github/workflows/deploy.yml`):
- **Triggers**: On pushes or PRs to `main`.
- **Lint Check**: Verifies python files build and Next.js typescript builds successfully.
- **CD Deploy**: Triggers on `main` push via a `self-hosted` runner on the RTX 5070 Ti GPU server, pulling latest changes, building docker images, restarting containers, and cleaning up space.
