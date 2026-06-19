import argparse
import json
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


PROMPT_VI = """
Bạn là OCR engine cho chữ viết tay tiếng Việt.

Hãy đọc chính xác nội dung trong ảnh.
Giữ đúng dấu tiếng Việt, dấu câu, chữ hoa/chữ thường nếu nhìn thấy.
Không thêm nội dung không có trong ảnh.
Không giải thích.
Không tóm tắt.
Chỉ trả về đúng nội dung văn bản trong ảnh.
""".strip()


VERIFY_PROMPT_VI = """
Bạn là bộ lọc rác OCR cho tài liệu tiếng Việt.

Bạn nhận:
- Ảnh crop của một dòng hoặc một vùng văn bản.
- Kết quả OCR hiện tại của vùng đó.
- Loại trường nếu có.
- Context gần nếu có.

Mục tiêu:
Chỉ quyết định vùng này có phải nội dung văn bản chính hay là rác cần bỏ.
Không sửa OCR. Không đọc lại thay OCR. Không thay đổi nội dung OCR hiện tại.

QUY TRÌNH BẮT BUỘC:
1. Nhìn ảnh crop để phân loại vùng.
2. Nếu crop là chữ thật thuộc nội dung chính của tài liệu, đặt action="keep" và final_text giữ đúng OCR hiện tại.
   - Nếu crop chỉ là nền trắng, vùng trống, đường kẻ, gạch chân, khung bảng, dấu chấm dẫn dòng, logo, icon, dấu đỏ, con dấu, chữ trang trí, nhiễu scan, ký hiệu rời, hoặc ký tự rác không tạo thành nội dung đọc được, đặt action="drop".
   - Nếu crop có chữ nhưng quá mờ, bị cắt mất chữ, hoặc dính nhiều dòng đến mức không thể xác định, đặt action="uncertain".
3. Không được dùng action="replace". Nếu là text thật nhưng OCR có vẻ sai, vẫn action="keep" hoặc "uncertain"; không sửa.
4. Không cố OCR những vùng rác. Với vùng rác, final_text phải là chuỗi rỗng.
5. Context chỉ dùng để hiểu vùng có thuộc nội dung chính hay không, không dùng để suy diễn chữ.

TIÊU CHÍ ACTION:
- keep: crop là text thật thuộc nội dung chính; final_text giữ đúng OCR hiện tại.
- drop: crop không phải text nội dung chính, ví dụ nền trắng, nhiễu, đường kẻ, khung bảng, dấu chấm dẫn dòng, logo, icon, dấu đỏ, con dấu, chữ trang trí, ký hiệu rời, hoặc text rác không đọc được.
- uncertain: không đủ rõ để kết luận keep/drop.

OCR hiện tại:
"{ocr_text}"

Field type:
"{field_type}"

Context gần:
"{context_text}"

Chỉ trả về JSON hợp lệ, không markdown, không giải thích:
{{
  "is_text": true,
  "region_type": "printed|handwriting|mixed|stamp|logo|noise|blank|line|unknown",
  "is_ocr_correct": true,
  "suspicious_parts": [],
  "final_text": "{ocr_text}",
  "action": "keep|drop|uncertain",
  "confidence": 0.0
}}
""".strip()


def load_model(model_name: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # Cân bằng giữa tốc độ và độ rõ cho line crop.
    # Nếu crop dòng rất dài hoặc chữ quá nhỏ, có thể tăng max_pixels.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return model, processor


def generate_from_image(model, processor, image_path: str, prompt: str, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def ocr_image(model, processor, image_path: str, prompt: str):
    return generate_from_image(
        model,
        processor,
        image_path,
        prompt,
        max_new_tokens=256,
    )


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False

    return default


def _extract_json_text(text: str) -> str:
    text = (text or "").strip()

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return text[start:end + 1].strip()

    return text


def _json_from_text(text: str):
    json_text = _extract_json_text(text)
    data = json.loads(json_text)

    action = str(data.get("action", "uncertain")).strip().lower()
    if action not in {"keep", "replace", "drop", "uncertain"}:
        action = "uncertain"

    region_type = str(data.get("region_type", "unknown")).strip().lower()
    valid_region_types = {
        "printed",
        "handwriting",
        "mixed",
        "stamp",
        "logo",
        "noise",
        "blank",
        "line",
        "unknown",
    }

    if region_type not in valid_region_types:
        region_type = "unknown"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    suspicious_parts = data.get("suspicious_parts", [])
    if not isinstance(suspicious_parts, list):
        suspicious_parts = []

    return {
        "is_text": _as_bool(data.get("is_text"), action != "drop"),
        "region_type": region_type,
        "is_ocr_correct": _as_bool(data.get("is_ocr_correct"), action == "keep"),
        "suspicious_parts": [
            str(x).strip()
            for x in suspicious_parts
            if str(x).strip()
        ],
        "final_text": str(data.get("final_text", "")).strip(),
        "action": action,
        "confidence": confidence,
    }


def levenshtein(a: str, b: str) -> int:
    a = a or ""
    b = b or ""

    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))

        previous = current

    return previous[-1]


def edit_distance_ratio(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    denom = max(len(a), len(b), 1)
    return levenshtein(a, b) / denom


def is_obvious_garbage(text: str) -> bool:
    text = (text or "").strip()

    if not text:
        return True

    # Rất ngắn, thường là detector bắt nhầm nét/đường/ký hiệu.
    if len(text) <= 1:
        return True

    alnum_count = sum(1 for c in text if c.isalnum())

    # Toàn ký hiệu, không có chữ/số.
    if alnum_count == 0:
        return True

    allowed_punct = set(" .,;:/()-_+%[]{}")

    weird_count = sum(
        1 for c in text
        if not c.isalnum() and c not in allowed_punct
    )

    # Quá nhiều ký tự lạ so với độ dài.
    if len(text) >= 3 and weird_count / max(len(text), 1) > 0.45:
        return True

    # Pattern OCR rác hay gặp.
    garbage_patterns = [
        r"^[|/_\\\-—–.]+$",
        r"^[=~`'^:;,.]+$",
        r"^[A-Z0-9=()\-_/\\|]{4,}$",
        r"^[Il1|]{1,3}$",
    ]

    for pattern in garbage_patterns:
        if re.match(pattern, text):
            return True

    return False


def decide_final_text(ocr_text: str, verdict: dict):
    """
    Không tin Qwen tuyệt đối.
    Hàm này quyết định có nhận kết quả keep/replace/drop của Qwen hay không.
    """

    ocr_text = ocr_text or ""
    action = verdict.get("action", "uncertain")
    region_type = verdict.get("region_type", "unknown")
    final_text = (verdict.get("final_text") or "").strip()

    try:
        confidence = float(verdict.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if action == "keep":
        return {
            "accepted_text": ocr_text,
            "accepted_action": "keep",
            "accept_qwen": True,
            "reason": "qwen_keep",
        }

    if action == "drop":
        strong_drop_regions = {
            "noise",
            "blank",
            "line",
            "logo",
            "stamp",
        }

        # Drop mạnh nếu Qwen phân loại rõ là rác.
        if region_type in strong_drop_regions and confidence >= 0.65:
            return {
                "accepted_text": "",
                "accepted_action": "drop",
                "accept_qwen": True,
                "reason": f"drop_region={region_type}_confidence={confidence:.2f}",
            }

        # Drop nếu final_text rỗng và confidence tương đối ổn.
        if not final_text and confidence >= 0.70:
            return {
                "accepted_text": "",
                "accepted_action": "drop",
                "accept_qwen": True,
                "reason": f"drop_empty_text_confidence={confidence:.2f}",
            }

        # Nếu OCR ban đầu rõ ràng là rác thì cho drop dễ hơn.
        if is_obvious_garbage(ocr_text) and confidence >= 0.60:
            return {
                "accepted_text": "",
                "accepted_action": "drop",
                "accept_qwen": True,
                "reason": f"drop_obvious_garbage_confidence={confidence:.2f}",
            }

        return {
            "accepted_text": ocr_text,
            "accepted_action": "review",
            "accept_qwen": False,
            "reason": "drop_not_confident_enough",
        }

    if action == "uncertain":
        return {
            "accepted_text": ocr_text,
            "accepted_action": "review",
            "accept_qwen": False,
            "reason": "qwen_uncertain",
        }

    if action == "replace":
        return {
            "accepted_text": ocr_text,
            "accepted_action": "review",
            "accept_qwen": False,
            "reason": "replace_disabled_garbage_filter_only",
        }

    return {
        "accepted_text": ocr_text,
        "accepted_action": "review",
        "accept_qwen": False,
        "reason": "unknown_action",
    }


def verify_line(
    model,
    processor,
    image_path: str,
    ocr_text: str,
    field_type: str = "unknown",
    context_text: str = "",
):
    prompt = VERIFY_PROMPT_VI.format(
        ocr_text=(ocr_text or "").replace('"', '\\"'),
        field_type=field_type or "unknown",
        # Không nên nhét toàn trang quá dài, dễ làm Qwen đoán theo context.
        context_text=(context_text or "").replace('"', '\\"')[:1200],
    )

    raw_text = generate_from_image(
        model,
        processor,
        image_path,
        prompt,
        max_new_tokens=192,
    )

    try:
        verdict = _json_from_text(raw_text)
    except Exception as exc:
        verdict = {
            "is_text": True,
            "region_type": "unknown",
            "is_ocr_correct": False,
            "suspicious_parts": [],
            "final_text": ocr_text or "",
            "action": "uncertain",
            "confidence": 0.0,
            "parse_error": str(exc),
        }

    verdict["raw_response"] = raw_text
    verdict["decision"] = decide_final_text(ocr_text or "", verdict)

    return verdict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Đường dẫn ảnh crop dòng/vùng văn bản",
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Tên model Hugging Face",
    )

    parser.add_argument(
        "--verify-text",
        default=None,
        help="Nếu truyền vào, Qwen sẽ verify/sửa text này thay vì OCR tự do",
    )

    parser.add_argument(
        "--field-type",
        default="unknown",
        help="Loại trường: unknown, person_name, phone, money, date, address, contract_id, free_text...",
    )

    parser.add_argument(
        "--context-text",
        default="",
        help="Context gần của dòng, không nên truyền toàn trang quá dài",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load_model(args.model)

    if args.verify_text is not None:
        result = verify_line(
            model=model,
            processor=processor,
            image_path=args.image,
            ocr_text=args.verify_text,
            field_type=args.field_type,
            context_text=args.context_text,
        )

        result = json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    else:
        result = ocr_image(
            model,
            processor,
            args.image,
            PROMPT_VI,
        )

    print("\n===== QWEN RESULT =====")
    print(result)


if __name__ == "__main__":
    main()
