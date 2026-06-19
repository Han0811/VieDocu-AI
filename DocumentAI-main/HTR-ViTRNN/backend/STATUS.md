# Trạng thái sản phẩm Document OCR

Ngày cập nhật: 2026-06-13

## 1. Mục tiêu hiện tại

Sản phẩm đang ở giai đoạn chuẩn bị backend trước khi triển khai frontend.

Mục tiêu backend:

- Nhận ảnh tài liệu đầu vào.
- Detect từng dòng bằng PaddleOCRv5.
- Với dòng có cả chữ in và chữ viết tay, crop thêm từng phần `printed` / `handwriting`.
- Đưa HTR nhận diện trên crop phù hợp.
- Dùng metadata nối lại đúng thứ tự để tạo text cuối theo dòng.
- Qwen chỉ dùng làm bộ lọc rác nếu bật, không dùng để sửa OCR.
- Output cuối phải sạch, dễ dùng cho frontend/API.

## 2. Kiến trúc hiện tại

Backend đã được tách thành package `backend/`.

Các file chính:

- `backend/config.py`: cấu hình production, đường dẫn checkpoint, charset, Paddle, HTR, Qwen.
- `backend/service.py`: service chạy OCR end-to-end, trả JSON cho API/frontend.
- `backend/api.py`: FastAPI app với các endpoint cơ bản.
- `backend/cli.py`: CLI smoke test dùng cùng codepath với API.
- `backend/README.md`: hướng dẫn chạy backend.
- `requirements-backend.txt`: dependency riêng cho API.
- `run/backend_api.sh`: script chạy FastAPI bằng uvicorn.
- `document_ocr.py`: OCR engine chính.
- `qwen.py`: Qwen garbage-filter module, chỉ dùng khi bật Qwen.

Backend service hiện gọi trực tiếp `process_pages()` trong `document_ocr.py`.

## 3. Runtime OCR Pipeline

Pipeline hiện tại:

1. Nhận ảnh hoặc folder ảnh.
2. PaddleOCRv5 detect line-level boxes.
3. Lưu full-line crop: `line_xxx_full.png`.
4. Phân tích từng line crop:
   - Nếu line in thuần: giữ full-line, không tách nhỏ.
   - Nếu line viết tay thuần: giữ full-line.
   - Nếu line mixed đủ chắc: tách thành part crop `line_xxx_part_00.png`, `line_xxx_part_01.png`.
5. HTR nhận diện crop.
6. Rule cleanup xử lý lỗi rõ ràng và drop rác.
7. Nếu bật Qwen:
   - chỉ chọn candidate nghi rác.
   - Qwen trả verdict keep/drop/uncertain.
   - không dùng Qwen để sửa nội dung OCR.
8. Metadata nối các part theo tọa độ x để tạo line text cuối.
9. Ghi output sạch cho frontend.

## 4. Output Contract Cho Frontend

Mỗi job sinh một thư mục output, ví dụ:

```text
backend_runs/<job_id>/output/
```

Hoặc nếu chạy CLI có `--out`, output nằm đúng thư mục được truyền.

Các output quan trọng:

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

### `texts/<page>.txt`

Text cuối cùng, mỗi dòng là một line sau khi merge part:

```text
Số tiền đã chuyển: 500.000000(ND (Bằng chữ: Năm trăm triệu đồng)
```

Dùng khi frontend chỉ cần hiển thị text đơn giản.

### `lines/<page>.tsv`

Output sạch nhất cho frontend dạng bảng:

```text
line_id	text
21	Số tiền đã chuyển: 500.000000(ND (Bằng chữ: Năm trăm triệu đồng)
```

### `lines/<page>.json`

Output chính cho frontend nếu cần highlight/click từng dòng.

Mỗi line có:

- `line_id`
- `text`
- `raw_text`
- `line_crop_path`
- `box_xyxy`
- `region_types`
- `parts`

`parts` chứa metadata từng crop con:

- `part_id`
- `region_id`
- `region_type`: `printed`, `handwriting`, `mixed`
- `text`
- `raw_text`
- `crop_path`
- `box_xyxy`
- `local_box_xyxy`
- `dropped`

### `regions/<page>.tsv`

Dùng để debug tách chữ in / chữ viết tay:

```text
line_id	region_id	region_type	text	crop_path	local_box_xyxy	box_xyxy
21	21	printed	Số tiền đã chuyển:	.../line_021_part_00.png	...
21	22	handwriting	500.000000...	.../line_021_part_01.png	...
```

### Debug Image

`debug_boxes/<page>_style_regions.jpg` vẽ các region theo style:

- `printed`
- `handwriting`
- `mixed`

File này dùng để kiểm tra nhanh tách part có đúng không.

## 5. API Hiện Tại

FastAPI app nằm ở:

```text
backend/api.py
```

Endpoint:

- `GET /health`
- `GET /config`
- `POST /ocr`

### `GET /health`

Trả trạng thái runtime:

- checkpoint có tồn tại không.
- charset file có tồn tại không.
- config hiện tại.

### `GET /config`

Trả config backend đang dùng.

### `POST /ocr`

Nhận multipart upload:

- `file`: ảnh đầu vào.
- `mode`: `FAST`, `BALANCED`, `ACCURATE`.
- `paddle_device`: ví dụ `cpu`, `gpu:0`.
- `qwen_enabled`: `true` / `false`.

Response trả:

- `job_id`
- `output_dir`
- `page_count`
- `pages[]`
- `pages[].text`
- `pages[].lines`
- `pages[].files`
- `config`

Frontend có thể dùng trực tiếp `pages[].lines` để render kết quả.

## 6. Cách Chạy Backend

### CLI Smoke Test

```bash
/home/mpeclab/torch-env/bin/python -m backend.cli \
  --source ../img/a42a7b9b1f5d9e03c74c.jpg \
  --out backend_runs/smoke_a42 \
  --paddle-device cpu
```

Lệnh này đã chạy thành công trên môi trường hiện tại.

### API Server

Cài dependency API:

```bash
/home/mpeclab/torch-env/bin/pip install -r requirements-backend.txt
```

Chạy server:

```bash
./run/backend_api.sh
```

Mặc định:

```text
HOST=0.0.0.0
PORT=8000
```

Có thể đổi:

```bash
HOST=127.0.0.1 PORT=8080 ./run/backend_api.sh
```

## 7. Dependency Và Model Runtime

Backend inference cần giữ:

```text
document_ocr.py
qwen.py
backend/
run/document_ocr.sh
run/backend_api.sh
model/HTR_ViTRNN.py
model/resnet18.py
utils/utils.py
output/viet_lr1e4_64_layers_1024_dim/best_CER.pth
output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json
requirements-backend.txt
```

Checkpoint hiện tại:

```text
output/viet_lr1e4_64_layers_1024_dim/best_CER.pth
```

Charset hiện tại:

```text
output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json
```

`document_ocr.py` hiện không còn cần đọc `data/viet_hf/train.ln`, `valid.ln`, `test.ln`.

## 8. File Có Thể Bỏ Nếu Chỉ Deploy Inference

Nếu không train nữa, backend inference không cần:

```text
data/
train.py
valid.py
test.py
utils/option.py
utils/sam.py
example/
img/
.vscode/
outputs_dococr_paddle/
backend_runs/
__pycache__/
```

Lưu ý:

- `output/` đang bị `.gitignore` ignore, nhưng vẫn cần checkpoint và charset khi deploy.
- Nếu deploy container, nên copy checkpoint/charset vào image hoặc mount volume model.

## 9. Những Gì Đã Test

Đã chạy:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/mpeclab/torch-env/bin/python -m py_compile \
  document_ocr.py qwen.py backend/config.py backend/service.py backend/api.py backend/cli.py
```

Đã chạy:

```bash
bash -n run/document_ocr.sh
```

Đã chạy:

```bash
/home/mpeclab/torch-env/bin/python -m backend.cli --help
```

Đã test service runtime:

```bash
/home/mpeclab/torch-env/bin/python -c \
  "from backend.service import DocumentOCRService; s=DocumentOCRService(); print(s.inspect_runtime()['ready'])"
```

Kết quả:

```text
True
```

Đã chạy smoke OCR thật:

```bash
/home/mpeclab/torch-env/bin/python -m backend.cli \
  --source ../img/a42a7b9b1f5d9e03c74c.jpg \
  --out backend_runs/smoke_a42 \
  --paddle-device cpu
```

Kết quả:

- xử lý 1 page thành công.
- sinh `texts`, `lines`, `regions`, `metadata`, `debug_boxes`.
- response JSON có đủ `pages[].text`, `pages[].lines`, `pages[].files`.

## 10. Trạng Thái GPU / CPU

Môi trường hiện tại khi test không nhận `gpu:0` cho Paddle:

```text
Cannot use GPU because there is no GPU detected
```

Vì vậy smoke test đã chạy bằng:

```bash
--paddle-device cpu
```

Trong production có GPU, có thể dùng:

```bash
--paddle-device gpu:0
```

hoặc env:

```bash
OCR_PADDLE_DEVICE=gpu:0
```

## 11. Rủi Ro / Việc Còn Cần Làm

### API dependency chưa cài trong env hiện tại

`backend/api.py` cần:

```text
fastapi
uvicorn[standard]
python-multipart
```

Hiện đã có `requirements-backend.txt`, nhưng chưa cài trong env.

### OCR quality vẫn phụ thuộc HTR checkpoint

Backend output contract đã ổn, nhưng text nhận diện vẫn phụ thuộc chất lượng model HTR.

Các lỗi OCR như sai dấu, sai số, đọc sai chữ viết tay vẫn cần:

- cải thiện checkpoint.
- hoặc thêm post-processing riêng theo nghiệp vụ.
- hoặc dùng Qwen chỉ để drop rác, không sửa text.

### Qwen chưa nên bật mặc định

Lý do:

- tốn VRAM/RAM.
- chậm hơn.
- hiện design chỉ dùng lọc rác.

Mặc định đang là:

```text
qwen_enabled = false
```

### Backend hiện chạy OCR đồng bộ

`POST /ocr` hiện chạy sync.

Nếu frontend upload file lớn hoặc nhiều page, nên thêm job queue:

- submit job.
- trả `job_id`.
- frontend poll status.
- backend lưu result theo job.

## 12. Kế Hoạch Tiếp Theo Trước Frontend

Ưu tiên tiếp theo:

1. Cài `requirements-backend.txt`.
2. Chạy `./run/backend_api.sh`.
3. Test `GET /health`.
4. Test `POST /ocr` bằng curl/Postman.
5. Chốt response schema cho frontend.
6. Thêm static file serving hoặc endpoint download file debug/crop.
7. Thêm job status nếu cần xử lý bất đồng bộ.
8. Dọn repo chỉ còn inference files.

## 13. Curl Test Gợi Ý

Sau khi chạy API:

```bash
curl http://127.0.0.1:8000/health
```

Upload OCR:

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -F "file=@../img/a42a7b9b1f5d9e03c74c.jpg" \
  -F "mode=BALANCED" \
  -F "paddle_device=cpu" \
  -F "qwen_enabled=false"
```

Frontend chỉ cần dùng:

```text
response.pages[0].text
response.pages[0].lines
response.pages[0].files.debug_style_regions
```

## 14. Kết Luận

Backend đã đủ nền để bắt đầu nối frontend:

- Có service layer.
- Có API skeleton.
- Có CLI smoke test.
- Có output contract rõ.
- Không còn phụ thuộc data train để chạy OCR.
- Checkpoint và charset đã tách rõ khỏi code.

Việc cần làm trước khi frontend gọi thật:

- cài API dependencies.
- chạy API server.
- test upload thật.
- quyết định có cần async job queue không.

`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               