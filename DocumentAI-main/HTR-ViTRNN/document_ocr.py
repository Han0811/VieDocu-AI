import argparse
import json
import os
import re
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from model import HTR_ViTRNN
from utils import utils


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUTPUT_SUBDIRS = (
    "line_crops",
    "debug_boxes",
    "debug_json",
    "texts",
    "texts_raw",
    "json",
    "metadata",
    "lines",
    "regions",
)


def source_images(source):
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]
    return sorted(p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def htr_np_thumb(img, max_w, max_h):
    height, width = np.shape(img)[:2]
    if height <= 0 or width <= 0:
        return img
    resized_w = min(int(width * max_h / height), max_w)
    return np.array(Image.fromarray(img).resize((resized_w, max_h)))


def order_quad_points(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) != 4:
        return None

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def crop_by_paddle_poly(img, poly):
    quad = order_quad_points(poly)
    if quad is None:
        return None

    tl, tr, br, bl = quad
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    out_w = int(round(max(width_top, width_bottom)))
    out_h = int(round(max(height_right, height_left)))
    if out_w <= 1 or out_h <= 1:
        return None

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def box_from_poly(poly, img_w, img_h, pad_x_ratio=0.0, pad_y_ratio=0.08):
    pts = np.array(poly, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    pad_x = int(w * pad_x_ratio)
    pad_y = int(h * pad_y_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2, y2


def remove_vertical_overlaps(line_items):
    if len(line_items) < 2:
        return line_items

    items = sorted(line_items, key=lambda item: (item["box"][1], item["box"][0]))
    boxes = [list(item["box"]) for item in items]
    centers = [((box[1] + box[3]) / 2.0) for box in boxes]

    for idx in range(len(boxes) - 1):
        boundary = int(round((centers[idx] + centers[idx + 1]) / 2.0))
        if boxes[idx][3] >= boundary:
            boxes[idx][3] = boundary
        if boxes[idx + 1][1] <= boundary:
            boxes[idx + 1][1] = boundary + 1

    clean = []
    for item, box in zip(items, boxes):
        if box[3] > box[1] and box[2] > box[0]:
            item = dict(item)
            item["box"] = tuple(int(v) for v in box)
            clean.append(item)
    return clean


def paddle_result_to_line_items(result, img_w, img_h, pad_x_ratio, pad_y_ratio):
    polys = result.get("dt_polys")
    if polys is None:
        polys = result.get("rec_polys")
    if polys is None:
        polys = []

    scores = result.get("dt_scores")
    if scores is None:
        scores = []

    items = []
    for idx, poly in enumerate(polys):
        pts = np.array(poly, dtype=np.int32)
        if len(pts) < 3:
            continue
        box = box_from_poly(pts, img_w, img_h, pad_x_ratio, pad_y_ratio)
        items.append(
            {
                "poly": [[int(x), int(y)] for x, y in pts.tolist()],
                "box": box,
                "score": float(scores[idx]) if idx < len(scores) else 0.0,
            }
        )

    items = sorted(items, key=lambda item: (item["box"][1], item["box"][0]))
    if pad_x_ratio > 0 or pad_y_ratio > 0:
        return remove_vertical_overlaps(items)
    return items


def estimate_ink_mask(gray):
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None

    border_thickness = max(1, min(h, w) // 12)
    top = gray[:border_thickness, :]
    bottom = gray[-border_thickness:, :]
    left = gray[:, :border_thickness]
    right = gray[:, -border_thickness:]
    border = np.concatenate([top.reshape(-1), bottom.reshape(-1), left.reshape(-1), right.reshape(-1)])

    inner_y1 = h // 5
    inner_y2 = h - inner_y1
    inner_x1 = w // 5
    inner_x2 = w - inner_x1
    center = gray[inner_y1:inner_y2, inner_x1:inner_x2]
    if center.size == 0:
        center = gray

    border_mean = float(border.mean()) if border.size else float(gray.mean())
    center_mean = float(center.mean()) if center.size else float(gray.mean())
    ink = 255 - gray if border_mean >= center_mean else gray

    ink = cv2.GaussianBlur(ink, (5, 5), 0)
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


IMPORTANT_FIELD_KEYWORDS = {
    "document_title": ("giấy xác nhận",),
    "date": ("ngày", "tháng", "năm", "thời gian"),
    "money": ("số tiền", "bằng chữ", "đồng", "triệu", "tỷ"),
    "phone": ("điện thoại", "số điện thoại"),
    "contract_no": ("số hợp đồng", "hợp đồng"),
    "contract_status": ("tình trạng", "đặt cọc", "hoàn thiện", "thanh toán"),
    "address": ("địa chỉ", "xã", "phường", "quận", "huyện", "tỉnh", "thành phố"),
    "name": ("ông", "bà", "người chuyển tiền", "đại diện"),
}

PRINTED_TEXT_KEYWORDS = (
    "cộng hoà",
    "cộng hòa",
    "chủ nghĩa",
    "độc lập",
    "hạnh phúc",
    "công ty",
    "cổ phần",
    "giấy xác nhận",
    "thanh toán",
    "ngân hàng",
    "người chuyển tiền",
    "điện thoại",
    "chứng minh",
    "địa chỉ",
    "số tiền",
    "bằng chữ",
    "thời gian",
    "tình trạng",
    "đặt cọc",
    "hoàn thiện",
    "số hợp đồng",
    "đại diện công ty",
)


def infer_field_type(text):
    normalized = (text or "").casefold()

    if "giấy xác nhận" in normalized:
        return "document_title"
    if any(keyword in normalized for keyword in ("tình trạng hợp đồng", "đặt cọc", "hoàn thiện", "thanh toán")):
        return "contract_status"
    if "số hợp đồng" in normalized:
        return "contract_no"
    if any(keyword in normalized for keyword in ("số tiền", "bằng chữ", "triệu", " tỷ", "vnd", "₫")):
        return "money"
    if any(keyword in normalized for keyword in ("điện thoại", "số điện thoại")):
        return "phone"
    if any(keyword in normalized for keyword in ("ngày", "tháng", "năm", "thời gian")):
        return "date"
    if "địa chỉ" in normalized:
        return "address"
    if any(keyword in normalized for keyword in ("người chuyển tiền", "ông", "bà", "đại diện")):
        return "name"

    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", normalized):
        return "date"
    if re.search(r"\b\d[\d.,]{5,}\b", normalized) and ("đồng" in normalized or "vnd" in normalized):
        return "money"
    return "unknown"


def crop_features(crop_bgr):
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask = estimate_ink_mask(gray)
    foreground_ratio = float((mask > 0).mean()) if mask is not None else 0.0
    red_mask = (crop_bgr[:, :, 2] > 120) & (crop_bgr[:, :, 2] > crop_bgr[:, :, 1] * 1.35) & (crop_bgr[:, :, 2] > crop_bgr[:, :, 0] * 1.35)
    component_count = 0
    component_density = 0.0
    height_cv = 0.0
    width_cv = 0.0
    if mask is not None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = []
        heights = []
        widths = []
        min_area = max(3, int(h * w * 0.00015))
        max_area = max(min_area + 1, int(h * w * 0.35))
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            if comp_w <= 1 or comp_h <= 1:
                continue
            areas.append(area)
            heights.append(comp_h)
            widths.append(comp_w)

        component_count = len(areas)
        component_density = float(component_count / max(1.0, w / max(8.0, h * 0.45)))
        if len(heights) >= 2:
            height_cv = float(np.std(heights) / max(1.0, np.mean(heights)))
        if len(widths) >= 2:
            width_cv = float(np.std(widths) / max(1.0, np.mean(widths)))

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(w / max(1, h)),
        "foreground_ratio": foreground_ratio,
        "red_ratio": float(red_mask.mean()),
        "component_count": int(component_count),
        "component_density": component_density,
        "component_height_cv": height_cv,
        "component_width_cv": width_cv,
    }


def classify_text_region(features):
    if features.get("foreground_ratio", 0.0) < 0.004:
        return "blank"
    if features.get("red_ratio", 0.0) > 0.12:
        return "stamp"

    aspect_ratio = float(features.get("aspect_ratio", 0.0))
    component_density = float(features.get("component_density", 0.0))
    height_cv = float(features.get("component_height_cv", 0.0))
    width_cv = float(features.get("component_width_cv", 0.0))
    component_count = int(features.get("component_count", 0))

    if component_count <= 1 and aspect_ratio > 3.0:
        return "line"

    printed_score = 0
    handwriting_score = 0
    if component_density >= 2.0:
        printed_score += 2
    elif component_density <= 0.9:
        handwriting_score += 1

    if height_cv <= 0.45 and width_cv <= 0.85:
        printed_score += 1
    if height_cv >= 0.65 or width_cv >= 1.15:
        handwriting_score += 1

    if aspect_ratio >= 7.0 and component_density >= 1.3:
        printed_score += 1
    if aspect_ratio < 6.0 and component_density < 1.4:
        handwriting_score += 1

    if printed_score >= handwriting_score + 2:
        return "printed"
    if handwriting_score >= printed_score + 1:
        return "handwriting"
    return "mixed"


def looks_like_printed_text(text, features=None):
    normalized = normalize_output_text(text).casefold()
    if not normalized:
        return False
    if any(keyword in normalized for keyword in PRINTED_TEXT_KEYWORDS):
        return True

    if features:
        word_count = len(re.findall(r"\w+", normalized, flags=re.UNICODE))
        if (
            word_count >= 5
            and len(re.findall(r"[^\W\d_]", normalized, flags=re.UNICODE)) >= 18
            and float(features.get("aspect_ratio", 0.0)) >= 6.0
            and int(features.get("component_count", 0)) >= 12
        ):
            return True

    letters = [char for char in text if char.isalpha()]
    if len(letters) < 8:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / max(1, len(letters))
    return upper_ratio >= 0.65


def refine_region_type_after_recognition(region):
    if region.get("region_type") in {"blank", "line", "stamp", "logo", "noise"}:
        return region

    text = region.get("final_text", region.get("text", ""))
    if region.get("is_full_line_region") and looks_like_printed_text(text, region.get("crop_features", {})):
        region["region_type"] = "printed"
        for segment in region.get("segments", []):
            segment["region_type"] = "printed"
    return region


def valid_date(text):
    for day, month, year in re.findall(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text or ""):
        year = int(year) + 2000 if len(year) == 2 else int(year)
        if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1900 <= year <= 2100:
            return True
    for day, month, year in re.findall(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text or "", flags=re.IGNORECASE):
        if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1900 <= int(year) <= 2100:
            return True
    return False


def valid_money(text):
    normalized = (text or "").casefold().replace(" ", "")
    if not re.search(r"\d", normalized):
        return False
    if re.fullmatch(r".*?\d{1,3}(?:[.,]\d{3})+(?:đ|đồng|vnd)?.*?", normalized, flags=re.IGNORECASE):
        numbers = re.findall(r"\d[\d.,]*", normalized)
        if any(re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", number) for number in numbers):
            return True
        return False
    if re.search(r"\d+\s*(triệu|tỷ|đồng|vnd)", text or "", flags=re.IGNORECASE):
        return True
    return False


def valid_phone(text):
    digits = re.sub(r"\D", "", text or "")
    return 9 <= len(digits) <= 12


def valid_contract_status(text):
    normalized = (text or "").casefold()
    return any(value in normalized for value in ("đặt cọc", "hoàn thiện", "đã thanh toán", "chưa thanh toán", "đang xử lý", "đã hủy"))


def number_tokens(text):
    return re.findall(r"\d[\d.,/-]*", text or "")


def split_field_label(text):
    if ":" not in (text or ""):
        return None, text
    label, value = text.split(":", 1)
    if len(label.strip()) > 80:
        return None, text
    return label.strip(), value.strip()


def preserve_field_label(original_text, replacement_text):
    label, original_value = split_field_label(original_text)
    if not label:
        return replacement_text
    replacement_label, _ = split_field_label(replacement_text)
    if replacement_label and replacement_label.casefold() == label.casefold():
        return replacement_text
    replacement = (replacement_text or "").strip()
    if not replacement:
        return replacement_text
    return f"{label}: {replacement}"


def replacement_loses_numbers(original_text, replacement_text):
    original_numbers = number_tokens(original_text)
    if not original_numbers:
        return False
    replacement_numbers = number_tokens(replacement_text)
    if not replacement_numbers:
        return True
    original_digits = "".join(re.findall(r"\d", original_text or ""))
    replacement_digits = "".join(re.findall(r"\d", replacement_text or ""))
    return len(replacement_digits) < max(1, int(len(original_digits) * 0.6))


def garbage_ratio(text):
    return sum(1 for char in text if char in "|~`^{}[]<>_=+\\") / max(len(text), 1)


def apply_rule_cleanup(line):
    if line.get("source") == "qwen_drop":
        return line

    text = line.get("final_text", line.get("text", ""))
    stripped = text.strip()
    features = line.get("crop_features", {})
    field_type = line.get("field_type", "unknown")
    box = line.get("box") or [0, 0, 0, 0]
    x1, y1 = int(box[0]), int(box[1])

    should_drop = False
    if not stripped:
        should_drop = True
    elif len(stripped) <= 1 and field_type == "unknown":
        should_drop = True
    elif garbage_ratio(stripped) > 0.18 and len(stripped) <= 16:
        should_drop = True
    elif re.fullmatch(r"[\W\d_]{1,16}", stripped, flags=re.UNICODE) and not re.search(r"\d{3,}", stripped):
        should_drop = True
    elif features.get("foreground_ratio", 1.0) < 0.004:
        should_drop = True
    elif (
        features.get("red_ratio", 0.0) > 0.025
        and x1 < 450
        and y1 < 520
        and len(stripped) <= 24
        and field_type in {"unknown", "name"}
    ):
        should_drop = True

    if should_drop:
        line["final_text"] = ""
        line["dropped"] = True
        line["uncertain"] = False
        line["source"] = "rule_drop"
        line["final_confidence"] = 0.92
        return line

    fixed = stripped
    replacements = [
        (r"\bHoài\s+thiện\b", "Hoàn thiện"),
        (r"\bHoan\s+thiên\b", "Hoàn thiện"),
        (r"\bHoài\s+thiên\b", "Hoàn thiện"),
        (r"\bSẦY\s+XÁC\b", "GIẤY XÁC"),
        (r"\bSẤY\s+XÁC\b", "GIẤY XÁC"),
        (r"\bSÀY\s+XÁC\b", "GIẤY XÁC"),
        (r"\bCÓ\s+PHẦN\b", "CỔ PHẦN"),
    ]
    for pattern, replacement in replacements:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

    fixed = re.sub(r"ngày\s*2\s*8", "ngày 28", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\s+\.", " ", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed).strip()

    if fixed != stripped:
        line["final_text"] = fixed
        line["dropped"] = False
        line["uncertain"] = False
        line["source"] = "rule_replace"
        line["final_confidence"] = max(float(line.get("final_confidence", 0.0)), 0.93)

    original_label, _ = split_field_label(line.get("text", ""))
    final_label, _ = split_field_label(line.get("final_text", ""))
    if original_label and not final_label and line.get("final_text"):
        line["final_text"] = preserve_field_label(line.get("text", ""), line["final_text"])
        if line.get("source") == "htr":
            line["source"] = "rule_preserve_label"

    if replacement_loses_numbers(line.get("text", ""), line.get("final_text", "")):
        line["final_text"] = line.get("text", "")
        line["dropped"] = False
        line["uncertain"] = True
        line["source"] = "rule_rejected_number_loss"
    return line


def quality_gate_score(line):
    text = line.get("final_text", line.get("text", ""))
    stripped = text.strip()
    field_type = line.get("field_type", "unknown")
    features = line.get("crop_features", {})
    score = 0
    reasons = []

    if not stripped:
        score += 45
        reasons.append("empty_text")

    if garbage_ratio(text) > 0.08:
        score += 70
        reasons.append("garbage_chars")

    if len(stripped) <= 2 and features.get("aspect_ratio", 0.0) > 3.0:
        score += 60
        reasons.append("too_short_for_wide_crop")

    if features.get("foreground_ratio", 0.0) < 0.01:
        score += 35
        reasons.append("low_ink_ratio")

    if features.get("red_ratio", 0.0) > 0.12:
        score += 40
        reasons.append("red_region_possible_stamp")

    if line.get("segment_count", 1) > 1:
        score += 20
        reasons.append("split_line")

    if field_type != "unknown":
        score += 45
        reasons.append(f"important_field:{field_type}")

    if field_type == "date" and not valid_date(text):
        score += 100
        reasons.append("invalid_date")
    elif field_type == "money" and not valid_money(text):
        score += 100
        reasons.append("invalid_money")
    elif field_type == "phone" and not valid_phone(text):
        score += 100
        reasons.append("invalid_phone")
    elif field_type == "contract_status" and not valid_contract_status(text):
        score += 100
        reasons.append("invalid_contract_status")

    return score, reasons


def qwen_garbage_filter_score(line):
    text = line.get("final_text", line.get("text", ""))
    stripped = text.strip()
    features = line.get("crop_features", {})
    region_type = line.get("region_type", "unknown")
    score = 0
    reasons = []

    if region_type in {"blank", "line", "stamp", "logo", "noise"}:
        score += 90
        reasons.append(f"garbage_region:{region_type}")
    if not stripped:
        score += 65
        reasons.append("empty_text")
    if garbage_ratio(text) > 0.08:
        score += 70
        reasons.append("garbage_chars")
    if len(stripped) <= 2 and features.get("aspect_ratio", 0.0) > 3.0:
        score += 60
        reasons.append("too_short_for_wide_crop")
    if features.get("foreground_ratio", 0.0) < 0.01:
        score += 45
        reasons.append("low_ink_ratio")
    if features.get("red_ratio", 0.0) > 0.08:
        score += 45
        reasons.append("red_region_possible_stamp")

    return score, reasons


def select_qwen_candidates(lines, max_calls_per_page, verify_all_lines=False):
    candidates = []
    for line in lines:
        if line.get("dropped"):
            line["need_qwen_verify"] = False
            line["qwen_priority"] = 0
            line["qwen_reasons"] = ["already_dropped_by_rule"]
            continue
        score, reasons = qwen_garbage_filter_score(line)
        line["need_qwen_verify"] = bool(verify_all_lines or score > 0)
        line["qwen_priority"] = int(score)
        line["qwen_reasons"] = reasons
        if line["need_qwen_verify"]:
            candidates.append(line)

    candidates.sort(key=lambda item: item.get("qwen_priority", 0), reverse=True)
    return candidates[:max(0, max_calls_per_page)]


def apply_qwen_verdict(line, verdict, min_confidence_to_accept=0.85):
    action = verdict.get("action", "uncertain")
    try:
        qwen_confidence = float(verdict.get("confidence", 0.0))
    except (TypeError, ValueError):
        qwen_confidence = 0.0

    line["qwen_verdict"] = verdict
    if action == "drop" and qwen_confidence >= min_confidence_to_accept:
        line["final_text"] = ""
        line["dropped"] = True
        line["uncertain"] = False
        line["source"] = "qwen_drop"
        line["final_confidence"] = qwen_confidence
    else:
        line["final_text"] = line.get("final_text", line.get("text", ""))
        line["dropped"] = False
        line["uncertain"] = bool(line.get("uncertain", False) or action == "uncertain")
        if action in {"replace", "keep"}:
            line["source"] = line.get("source", "htr")
        else:
            line["source"] = line.get("source", "qwen_review")
        line["final_confidence"] = max(float(line.get("final_confidence", 0.0)), qwen_confidence)
    return line


def initialize_final_line_state(lines):
    for line in lines:
        line["field_type"] = infer_field_type(line.get("text", ""))
        line["final_text"] = line.get("text", "")
        line["dropped"] = False
        line["uncertain"] = False
        line["source"] = "htr"
        line["final_confidence"] = line.get("score", 0.0)
        line["need_qwen_verify"] = False
        line["qwen_priority"] = 0
        line["qwen_reasons"] = []


def make_line_records(lines):
    return [
        {
            "index": int(item.get("index", idx)),
            "line_id": int(item.get("index", idx)),
            "parent_line_id": item.get("parent_line_id"),
            "local_box_xyxy": item.get("local_box_xyxy"),
            "local_polygon": item.get("local_polygon"),
            "parent_line_crop_path": item.get("parent_line_crop_path"),
            "box_xyxy": [int(v) for v in item["box"]],
            "polygon": item["poly"],
            "score": float(item.get("score", 0.0)),
            "crop_path": item["crop_path"],
            "line_crop_path": item["line_crop_path"],
            "segment_count": len(item.get("segments", [])),
            "segments": item.get("segments", []),
            "parts": item.get("parts", []),
            "region_ids": item.get("region_ids", []),
            "region_types": item.get("region_types", [item.get("region_type", "unknown")]),
            "region_type": item.get("region_type", "unknown"),
            "is_full_line_region": bool(item.get("is_full_line_region", False)),
            "collapsed_weak_style_split": bool(item.get("collapsed_weak_style_split", False)),
            "ocr_text": item.get("text", ""),
            "raw_text": item.get("raw_text", item.get("text", "")),
            "final_text": item.get("final_text", item.get("text", "")),
            "text": item.get("final_text", item.get("text", "")),
            "recognizer": item.get("recognizer", "htr"),
            "source": item.get("source", "htr"),
            "field_type": item.get("field_type", "unknown"),
            "ocr_confidence": float(item.get("score", 0.0)),
            "final_confidence": float(item.get("final_confidence", 0.0)),
            "dropped": bool(item.get("dropped", False)),
            "uncertain": bool(item.get("uncertain", False)),
            "need_qwen_verify": bool(item.get("need_qwen_verify", False)),
            "qwen_priority": int(item.get("qwen_priority", 0)),
            "qwen_reasons": item.get("qwen_reasons", []),
            "qwen_verdict": item.get("qwen_verdict"),
            "crop_features": item.get("crop_features", {}),
        }
        for idx, item in enumerate(lines)
    ]


def prepare_output_dirs(output_dir, clean_output):
    if clean_output:
        for subdir in OUTPUT_SUBDIRS:
            path = output_dir / subdir
            if path.exists():
                shutil.rmtree(path)

    paths = {subdir: output_dir / subdir for subdir in OUTPUT_SUBDIRS}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_output_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def tsv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def write_tsv(path, rows, columns):
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(tsv_cell(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_clean_line_outputs(line_records):
    rows = []
    for record in line_records:
        text = normalize_output_text(record.get("final_text", ""))
        if record.get("dropped") or not text:
            continue
        parts = []
        for part_idx, part in enumerate(record.get("parts", [])):
            part_text = normalize_output_text(part.get("final_text", part.get("ocr_text", "")))
            parts.append(
                {
                    "part_id": part_idx,
                    "region_id": part.get("region_id"),
                    "region_type": part.get("region_type", "unknown"),
                    "text": part_text,
                    "raw_text": normalize_output_text(part.get("ocr_text", "")),
                    "crop_path": part.get("crop_path"),
                    "box_xyxy": part.get("box_xyxy"),
                    "local_box_xyxy": part.get("local_box_xyxy"),
                    "dropped": bool(part.get("dropped", False)),
                }
            )
        rows.append(
            {
                "line_id": int(record.get("line_id", len(rows))),
                "text": text,
                "raw_text": normalize_output_text(record.get("raw_text", record.get("ocr_text", ""))),
                "line_crop_path": record.get("line_crop_path"),
                "box_xyxy": record.get("box_xyxy"),
                "region_types": record.get("region_types", []),
                "parts": parts,
            }
        )
    return rows


def make_region_outputs(region_records):
    rows = []
    for record in region_records:
        text = normalize_output_text(record.get("final_text", record.get("ocr_text", "")))
        rows.append(
            {
                "region_id": int(record.get("line_id", record.get("index", len(rows)))),
                "line_id": record.get("parent_line_id"),
                "region_type": record.get("region_type", "unknown"),
                "text": text,
                "raw_text": normalize_output_text(record.get("raw_text", record.get("ocr_text", ""))),
                "crop_path": record.get("crop_path"),
                "line_crop_path": record.get("parent_line_crop_path"),
                "box_xyxy": record.get("box_xyxy"),
                "local_box_xyxy": record.get("local_box_xyxy"),
                "dropped": bool(record.get("dropped", False)),
                "source": record.get("source", "htr"),
            }
        )
    return rows


def save_debug(img, line_items, out_path):
    vis = img.copy()
    for idx, item in enumerate(line_items):
        poly = np.array(item["poly"], dtype=np.int32)
        x1, y1, x2, y2 = item["box"]
        cv2.polylines(vis, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, str(idx), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), vis)


def save_status_debug(img, line_items, out_path, mode="final"):
    colors = {
        "htr": (255, 0, 0),
        "qwen_keep": (0, 180, 0),
        "qwen_drop": (0, 0, 255),
        "uncertain": (0, 140, 255),
        "candidate": (255, 0, 255),
    }
    vis = img.copy()
    for item in line_items:
        poly = np.array(item.get("polygon") or item.get("poly"), dtype=np.int32)
        x1, y1, _, _ = item.get("box_xyxy") or item.get("box")
        if mode == "candidate":
            color = colors["candidate"] if item.get("need_qwen_verify") else colors["htr"]
        else:
            color = colors.get(item.get("source", "htr"), colors["htr"])
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)
        cv2.putText(vis, str(item["index"]), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.imwrite(str(out_path), vis)


def save_style_region_debug(img, region_records, out_path):
    colors = {
        "printed": (255, 0, 0),
        "handwriting": (0, 0, 255),
        "mixed": (0, 180, 255),
        "unknown": (160, 160, 160),
    }
    vis = img.copy()
    for record in region_records:
        region_type = record.get("region_type", "unknown")
        color = colors.get(region_type, colors["unknown"])
        poly = np.array(record.get("polygon") or record.get("poly"), dtype=np.int32)
        x1, y1, _, _ = record.get("box_xyxy") or record.get("box")
        parent_id = record.get("parent_line_id")
        region_id = record.get("line_id", record.get("index", ""))
        label = f"{parent_id}.{region_id}:{region_type}" if parent_id is not None else f"{region_id}:{region_type}"
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)
        cv2.putText(vis, label, (int(x1), max(20, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(str(out_path), vis)


def load_charset_alphabet(charset_file):
    path = Path(charset_file)
    if not path.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_file}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        charset = data.get("charset")
        if charset is None:
            charset = "".join(data.get("characters", []))
    else:
        charset = path.read_text(encoding="utf-8").rstrip("\n")

    if not charset:
        raise ValueError(f"Charset file is empty or invalid: {charset_file}")
    return dict(enumerate(charset)), charset


def checkpoint_output_classes(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict_ema") or ckpt.get("model") or ckpt
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key == "head.weight" and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    return None


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict_ema") or ckpt.get("model") or ckpt

    model_dict = OrderedDict()
    pattern = re.compile("module.")
    for key, value in state.items():
        if re.search("module", key):
            model_dict[re.sub(pattern, "", key)] = value
        else:
            model_dict[key] = value
    model.load_state_dict(model_dict, strict=True)
    return model


class LineRecognizer:
    def __init__(self, args):
        self.device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.img_size = args.img_size
        self.ralph, self.charset = load_charset_alphabet(args.charset_file)
        self.charset_source = args.charset_file
        print(f"Loaded HTR charset from file: {args.charset_file}")
        self.charset_size = len(self.charset)
        needed_cls = len(self.ralph) + 1
        self.needed_cls = needed_cls
        self.checkpoint_nb_cls = checkpoint_output_classes(args.checkpoint)
        self.nb_cls = self.checkpoint_nb_cls or args.nb_cls
        if self.checkpoint_nb_cls is not None and args.nb_cls != self.checkpoint_nb_cls:
            print(f"Using nb_cls={self.checkpoint_nb_cls} from checkpoint head.weight; CLI nb_cls={args.nb_cls} is ignored.")
        if self.nb_cls < needed_cls:
            raise ValueError(
                f"Checkpoint/model nb_cls={self.nb_cls} is smaller than viet_hf alphabet + CTC blank ({needed_cls})"
            )
        print(
            f"HTR alphabet: {self.charset_size} viet_hf chars + blank = {needed_cls}; checkpoint output classes = {self.nb_cls}"
        )

        self.converter = utils.CTCLabelConverter(self.ralph.values())
        self.model = HTR_ViTRNN.create_model(
            nb_cls=self.nb_cls,
            img_size=args.img_size[::-1],
            num_layer_RNN=args.num_layers_RNN,
            hidden_dim_RNN=args.hidden_dim_RNN,
        )
        self.model = load_checkpoint(self.model, args.checkpoint, self.device)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_crop_array(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        image_data = np.array(pil_img.convert("L"))
        image_data = htr_np_thumb(image_data, self.img_size[0], self.img_size[1])
        image_data = image_data.astype(np.float32) / 255.0
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.pad(
            image_data,
            ((0, 0), (0, self.img_size[0] - np.shape(image_data)[1]), (0, 0)),
            mode="constant",
            constant_values=(1.0),
        )
        return image_data.transpose((2, 0, 1))

    @torch.no_grad()
    def predict_crops(self, crop_bgrs, batch_size=64):
        if not crop_bgrs:
            return []

        texts = []
        for start in range(0, len(crop_bgrs), batch_size):
            batch = crop_bgrs[start:start + batch_size]
            image_arrays = [self.preprocess_crop_array(crop) for crop in batch]
            image = torch.from_numpy(np.stack(image_arrays, axis=0)).float().to(self.device, non_blocking=True)

            preds = self.model(image).float()
            preds_size = torch.IntTensor([preds.size(1)] * image.size(0))
            preds = preds.permute(1, 0, 2).log_softmax(2)
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            texts.extend(self.converter.decode(preds_index.data, preds_size.data))

        return texts


def create_paddle_detector(args):
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    try:
        import paddle
        from paddleocr._models.text_detection import TextDetection
    except ImportError as exc:
        raise RuntimeError("paddleocr/paddle is not installed in this Python environment.") from exc

    if args.paddle_device:
        try:
            paddle.device.set_device(args.paddle_device)
        except Exception as exc:
            raise RuntimeError(f"Cannot set Paddle device to {args.paddle_device}") from exc

    print(f"Paddle device: {args.paddle_device or paddle.device.get_device()} | CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
    return TextDetection(
        model_name=args.det_model_name,
        model_dir=args.det_model_dir,
        device=args.paddle_device,
        limit_side_len=args.text_det_limit_side_len,
        limit_type=args.text_det_limit_type,
        thresh=args.det_thresh,
        box_thresh=args.det_box_thresh,
        unclip_ratio=args.det_unclip_ratio,
    )


def local_box_to_page_box(local_box, parent_box, crop_w, crop_h):
    px1, py1, px2, py2 = [int(v) for v in parent_box]
    lx1, ly1, lx2, ly2 = [int(v) for v in local_box]
    parent_w = max(1, px2 - px1)
    parent_h = max(1, py2 - py1)
    return (
        int(round(px1 + lx1 / max(1, crop_w) * parent_w)),
        int(round(py1 + ly1 / max(1, crop_h) * parent_h)),
        int(round(px1 + lx2 / max(1, crop_w) * parent_w)),
        int(round(py1 + ly2 / max(1, crop_h) * parent_h)),
    )


def pad_local_box(box, crop_w, crop_h, pad_x_ratio, pad_y_ratio):
    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(round(w * pad_x_ratio))
    pad_y = int(round(h * pad_y_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(int(crop_w), x2 + pad_x),
        min(int(crop_h), y2 + pad_y),
    )


def should_keep_style_box(item, crop_w, crop_h, args):
    x1, y1, x2, y2 = item["box"]
    w = max(0, int(x2) - int(x1))
    h = max(0, int(y2) - int(y1))
    if w <= 0 or h <= 0:
        return False
    if float(item.get("score", 0.0)) < args.style_split_min_score:
        return False
    if w < max(4, int(crop_w * args.style_split_min_width_ratio)):
        return False
    if h < max(4, int(crop_h * args.style_split_min_height_ratio)):
        return False
    area_ratio = (w * h) / max(1.0, float(crop_w * crop_h))
    if area_ratio < args.style_split_min_area_ratio:
        return False
    return True


def normalized_style_type(region_type):
    if region_type == "printed":
        return "printed"
    if region_type == "handwriting":
        return "handwriting"
    return "mixed"


def group_style_items(style_items, crop_w, args):
    if not style_items:
        return []

    ordered = sorted(style_items, key=lambda item: (item["box"][0], item["box"][1]))
    groups = []
    for item in ordered:
        style_type = normalized_style_type(item.get("region_type", "mixed"))
        if not groups:
            groups.append({"region_type": style_type, "items": [item]})
            continue

        prev_group = groups[-1]
        prev_box = union_box(prev_group["items"])
        gap = int(item["box"][0]) - int(prev_box[2])
        max_gap = max(6, int(crop_w * args.style_split_merge_gap_ratio))
        if prev_group["region_type"] == style_type and (style_type == "printed" or gap <= max_gap):
            prev_group["items"].append(item)
        else:
            groups.append({"region_type": style_type, "items": [item]})

    return groups


def component_style_items(crop_bgr, local_box, base_score, args):
    crop = safe_crop(crop_bgr, *local_box)
    if crop is None:
        return []

    crop_h, crop_w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = estimate_ink_mask(gray)
    if mask is None:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    areas = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < args.mixed_split_min_component_area:
            continue
        if w <= 1 or h <= 1:
            continue
        if area > crop_w * crop_h * 0.45:
            continue
        comps.append({"box": (x, y, x + w, y + h), "area": area, "width": w, "height": h})
        areas.append(area)

    if not comps:
        return []

    median_area = float(np.median(areas)) if areas else 1.0
    text_comps = []
    marks = []
    for comp in comps:
        x1, y1, x2, y2 = comp["box"]
        w = comp["width"]
        h = comp["height"]
        area = comp["area"]

        is_tiny_mark = (
            h <= max(4, int(crop_h * args.mixed_split_mark_height_ratio))
            and w <= max(4, int(crop_h * args.mixed_split_mark_width_ratio))
        )
        if is_tiny_mark:
            marks.append(comp)
            continue

        handwriting_score = 0
        printed_score = 0
        if h >= crop_h * args.mixed_split_handwriting_height_ratio:
            handwriting_score += 2
        if w >= crop_h * args.mixed_split_handwriting_width_ratio:
            handwriting_score += 1
        if area >= median_area * args.mixed_split_handwriting_area_factor:
            handwriting_score += 1
        if h <= crop_h * args.mixed_split_printed_max_height_ratio:
            printed_score += 2
        if w <= crop_h * args.mixed_split_printed_max_width_ratio:
            printed_score += 1

        region_type = "handwriting" if handwriting_score > printed_score else "printed"
        text_comps.append({**comp, "region_type": region_type})

    if not text_comps:
        return []

    ordered = sorted(text_comps, key=lambda item: (item["box"][0], item["box"][1]))
    runs = []
    for comp in ordered:
        if not runs:
            runs.append({"region_type": comp["region_type"], "components": [comp]})
            continue

        prev = runs[-1]
        prev_box = union_box([{"box": item["box"]} for item in prev["components"]])
        gap = int(comp["box"][0]) - int(prev_box[2])
        max_gap = max(5, int(crop_w * args.mixed_split_merge_gap_ratio))
        if prev["region_type"] == comp["region_type"] and gap <= max_gap:
            prev["components"].append(comp)
        else:
            runs.append({"region_type": comp["region_type"], "components": [comp]})

    attach_px = max(4, int(crop_h * args.mixed_split_mark_attach_ratio))
    for mark in marks:
        mx = (mark["box"][0] + mark["box"][2]) / 2.0
        my1, my2 = mark["box"][1], mark["box"][3]
        best = None
        for run_idx, run in enumerate(runs):
            run_box = union_box([{"box": item["box"]} for item in run["components"]])
            overlap_y = min(my2, run_box[3]) - max(my1, run_box[1])
            distance = 0
            if mx < run_box[0]:
                distance = run_box[0] - mx
            elif mx > run_box[2]:
                distance = mx - run_box[2]
            if distance <= attach_px and overlap_y >= -attach_px:
                candidate = (distance, run_idx)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            runs[best[1]]["components"].append({**mark, "region_type": runs[best[1]]["region_type"]})

    split_items = []
    offset_x, offset_y = int(local_box[0]), int(local_box[1])
    for run_idx, run in enumerate(runs):
        run_box = union_box([{"box": item["box"]} for item in run["components"]])
        x1, y1, x2, y2 = pad_local_box(
            (run_box[0] + offset_x, run_box[1] + offset_y, run_box[2] + offset_x, run_box[3] + offset_y),
            crop_bgr.shape[1],
            crop_bgr.shape[0],
            args.mixed_split_pad_x_ratio,
            args.mixed_split_pad_y_ratio,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        split_items.append(
            {
                "poly": bbox_polygon((x1, y1, x2, y2)),
                "box": (x1, y1, x2, y2),
                "score": float(base_score),
                "local_index": run_idx,
                "region_type": run["region_type"],
                "mixed_component_split": True,
            }
        )

    min_run_width = max(args.mixed_split_min_run_width_px, int(crop_w * args.mixed_split_min_run_width_ratio))
    split_items = [
        item
        for item in split_items
        if int(item["box"][2]) - int(item["box"][0]) >= min_run_width
    ]
    if len(split_items) > args.mixed_split_max_runs:
        return []
    if len({item["region_type"] for item in split_items}) < 2:
        return []
    return split_items


def small_mark_count_in_window(crop_bgr, x1, x2, args):
    h, w = crop_bgr.shape[:2]
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    if x2 <= x1:
        return 0

    crop = crop_bgr[:, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = estimate_ink_mask(gray)
    if mask is None:
        return 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    count = 0
    for label_idx in range(1, num_labels):
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        if area < args.mixed_split_min_component_area:
            continue
        center_y = y + comp_h / 2.0
        if center_y < h * args.form_split_mark_min_y_ratio:
            continue
        if (
            comp_h <= max(4, int(h * args.mixed_split_mark_height_ratio))
            and comp_w <= max(4, int(h * args.mixed_split_mark_width_ratio))
        ):
            count += 1
    return count


def has_form_separator(crop_bgr, left_box, right_box, args):
    h, w = crop_bgr.shape[:2]
    gap = int(right_box[0]) - int(left_box[2])
    if gap < args.form_split_min_gap_px:
        return False
    if gap >= max(8, int(w * args.form_split_min_gap_ratio)):
        return True

    window_left = int(left_box[2] - h * args.form_split_separator_window_ratio)
    window_right = int(right_box[0] + h * args.form_split_separator_window_ratio)
    return small_mark_count_in_window(crop_bgr, window_left, window_right, args) >= args.form_split_min_mark_count


def handwriting_tail_has_shape_delta(left_box, right_box, line_h, args):
    left_h = max(1, int(left_box[3]) - int(left_box[1]))
    right_h = max(1, int(right_box[3]) - int(right_box[1]))
    if right_h >= left_h * args.form_split_tail_height_factor:
        return True

    y_delta = max(2, int(line_h * args.form_split_tail_y_delta_ratio))
    return int(right_box[1]) <= int(left_box[1]) - y_delta or int(right_box[3]) >= int(left_box[3]) + y_delta


def validate_form_style_split(crop_bgr, split_items, args):
    if len(split_items) < 2 or len(split_items) > args.mixed_split_max_runs:
        return []

    ordered = sorted(split_items, key=lambda item: item["box"][0])
    if ordered[0].get("region_type") != "printed":
        return []

    handwriting_idx = next((idx for idx, item in enumerate(ordered) if item.get("region_type") == "handwriting"), None)
    if handwriting_idx is None or handwriting_idx == 0:
        return []

    left_box = union_box(ordered[:handwriting_idx])
    right_box = union_box(ordered[handwriting_idx:])
    if not has_form_separator(crop_bgr, left_box, right_box, args):
        return []
    if not handwriting_tail_has_shape_delta(left_box, right_box, crop_bgr.shape[0], args):
        return []

    return ordered


def form_anchor_style_items(crop_bgr, base_score, args):
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask = estimate_ink_mask(gray)
    if mask is None:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    areas = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < args.mixed_split_min_component_area or comp_w <= 1 or comp_h <= 1:
            continue
        is_mark = (
            comp_h <= max(4, int(h * args.mixed_split_mark_height_ratio))
            and comp_w <= max(4, int(h * args.mixed_split_mark_width_ratio))
        )
        comps.append(
            {
                "box": (x, y, x + comp_w, y + comp_h),
                "area": area,
                "width": comp_w,
                "height": comp_h,
                "is_mark": is_mark,
            }
        )
        if not is_mark:
            areas.append(area)

    text_comps = [comp for comp in comps if not comp["is_mark"]]
    if len(text_comps) < 3:
        return []

    median_area = float(np.median(areas)) if areas else 1.0
    handwriting_candidates = [
        comp
        for comp in text_comps
        if comp["box"][0] >= w * args.form_split_min_prefix_ratio
        and (
            comp["height"] >= h * args.mixed_split_handwriting_height_ratio
            or comp["width"] >= h * args.mixed_split_handwriting_width_ratio
            or comp["area"] >= median_area * args.mixed_split_handwriting_area_factor
        )
    ]
    if not handwriting_candidates:
        return []

    tail_start = min(comp["box"][0] for comp in handwriting_candidates)
    prefix_comps = [comp for comp in text_comps if comp["box"][2] < tail_start]
    tail_comps = [comp for comp in text_comps if comp["box"][2] >= tail_start]
    if not prefix_comps or not tail_comps:
        return []

    prefix_box = union_box(prefix_comps)
    tail_box = union_box(tail_comps)
    min_prefix_width = max(args.mixed_split_min_run_width_px, int(w * args.form_split_min_prefix_ratio))
    min_tail_width = max(args.mixed_split_min_run_width_px, int(w * args.form_split_min_tail_ratio))
    if prefix_box[2] - prefix_box[0] < min_prefix_width or tail_box[2] - tail_box[0] < min_tail_width:
        return []
    if not has_form_separator(crop_bgr, prefix_box, tail_box, args):
        return []
    if not handwriting_tail_has_shape_delta(prefix_box, tail_box, h, args):
        return []

    prefix_box = pad_local_box(prefix_box, w, h, args.mixed_split_pad_x_ratio, args.mixed_split_pad_y_ratio)
    tail_box = pad_local_box(tail_box, w, h, args.mixed_split_pad_x_ratio, args.mixed_split_pad_y_ratio)
    return [
        {
            "poly": bbox_polygon(prefix_box),
            "box": prefix_box,
            "score": float(base_score),
            "local_index": 0,
            "region_type": "printed",
            "form_anchor_split": True,
        },
        {
            "poly": bbox_polygon(tail_box),
            "box": tail_box,
            "score": float(base_score),
            "local_index": 1,
            "region_type": "handwriting",
            "form_anchor_split": True,
        },
    ]


def expand_mixed_style_groups(groups, line_crop, args):
    expanded = []
    for group in groups:
        if group.get("region_type") != "mixed":
            expanded.append(group)
            continue

        local_box = union_box(group["items"])
        base_score = float(np.mean([float(item.get("score", 0.0)) for item in group["items"]]))
        split_items = validate_form_style_split(line_crop, component_style_items(line_crop, local_box, base_score, args), args)
        if not split_items and local_box == (0, 0, line_crop.shape[1], line_crop.shape[0]):
            split_items = form_anchor_style_items(line_crop, base_score, args)
        if not split_items:
            expanded.append(group)
            continue

        for item in split_items:
            expanded.append({"region_type": item["region_type"], "items": [item]})
    return expanded


def collapse_weak_style_groups(groups, line_crop, args):
    if len(groups) < 2:
        return groups

    ordered = sorted(groups, key=lambda group: union_box(group["items"])[0])
    handwriting_idx = next((idx for idx, group in enumerate(ordered) if group.get("region_type") == "handwriting"), None)
    if handwriting_idx is None or handwriting_idx == 0:
        return groups

    left_box = union_box([item for group in ordered[:handwriting_idx] for item in group["items"]])
    right_box = union_box([item for group in ordered[handwriting_idx:] for item in group["items"]])
    if handwriting_tail_has_shape_delta(left_box, right_box, line_crop.shape[0], args):
        return groups

    h, w = line_crop.shape[:2]
    score_values = [
        float(item.get("score", 0.0))
        for group in ordered
        for item in group["items"]
    ]
    full_item = {
        "poly": bbox_polygon((0, 0, w, h)),
        "box": (0, 0, w, h),
        "score": float(np.mean(score_values)) if score_values else 0.0,
        "local_index": 0,
        "region_type": "mixed",
        "full_line": True,
        "collapsed_weak_style_split": True,
    }
    return [{"region_type": "mixed", "items": [full_item], "collapsed_weak_style_split": True}]


def detect_style_regions_in_line(detector, line_crop, line_crop_path, parent_item, page_crop_dir, line_prefix, args):
    crop_h, crop_w = line_crop.shape[:2]
    line_type = parent_item.get("region_type", "mixed")
    style_line_type = normalized_style_type(line_type)
    full_line_item = {
        "poly": bbox_polygon((0, 0, crop_w, crop_h)),
        "box": (0, 0, crop_w, crop_h),
        "score": float(parent_item.get("score", 0.0)),
        "local_index": 0,
        "region_type": line_type,
        "full_line": True,
    }

    if not args.split_line_by_style:
        sub_items = [full_line_item]
    else:
        base_score = float(parent_item.get("score", 0.0))
        sub_items = validate_form_style_split(
            line_crop,
            component_style_items(line_crop, (0, 0, crop_w, crop_h), base_score, args),
            args,
        )
        if not sub_items:
            sub_items = form_anchor_style_items(line_crop, base_score, args)
        if not sub_items and style_line_type in {"printed", "handwriting"}:
            sub_items = [full_line_item]

    if args.split_line_by_style and args.style_split_use_paddle_subboxes and not sub_items:
        try:
            sub_results = detector.predict(str(line_crop_path))
            sub_items = paddle_result_to_line_items(sub_results[0], crop_w, crop_h, 0.0, 0.0) if sub_results else []
        except Exception as exc:
            print(f"[WARN] Paddle style split failed on {line_crop_path}: {exc}")
            sub_items = []

        for sub_item in sub_items:
            sub_item["box"] = pad_local_box(
                sub_item["box"],
                crop_w,
                crop_h,
                args.style_split_pad_x_ratio,
                args.style_split_pad_y_ratio,
            )

        sub_items = [
            item
            for item in sub_items
            if should_keep_style_box(item, crop_w, crop_h, args)
        ]

    if not sub_items:
        sub_items = [full_line_item]

    style_items = []
    for local_idx, sub_item in enumerate(sub_items):
        local_crop = safe_crop(line_crop, *sub_item["box"])
        if local_crop is None:
            continue
        features = crop_features(local_crop)
        style_items.append(
            {
                **sub_item,
                "local_index": local_idx,
                "crop_features": features,
                "region_type": sub_item.get("region_type") or classify_text_region(features),
            }
        )

    groups = expand_mixed_style_groups(group_style_items(style_items, crop_w, args), line_crop, args)
    groups = collapse_weak_style_groups(groups, line_crop, args)
    regions = []
    for group_idx, group in enumerate(groups):
        local_box = union_box(group["items"])
        region_crop = safe_crop(line_crop, *local_box)
        if region_crop is None:
            continue

        region_type = group["region_type"]
        region_path = page_crop_dir / f"{line_prefix}_part_{group_idx:02d}.png"
        cv2.imwrite(str(region_path), region_crop)
        global_box = local_box_to_page_box(local_box, parent_item["box"], crop_w, crop_h)
        region_features = crop_features(region_crop)
        regions.append(
            {
                "index": None,
                "parent_line_id": int(parent_item["index"]),
                "parent_line_crop_path": parent_item["line_crop_path"],
                "parent_line_box_xyxy": [int(v) for v in parent_item["box"]],
                "local_box_xyxy": [int(v) for v in local_box],
                "local_polygon": bbox_polygon(local_box),
                "box": global_box,
                "poly": bbox_polygon(global_box),
                "score": float(np.mean([float(item.get("score", 0.0)) for item in group["items"]])),
                "crop_path": str(region_path.as_posix()),
                "line_crop_path": str(region_path.as_posix()),
                "crop_features": region_features,
                "region_type": region_type,
                "is_full_line_region": len(group["items"]) == 1 and bool(group["items"][0].get("full_line")),
                "collapsed_weak_style_split": bool(group.get("collapsed_weak_style_split")),
                "recognizer": "htr",
                "segments": [
                    {
                        "index": int(item.get("local_index", idx)),
                        "x1": int(item["box"][0]),
                        "x2": int(item["box"][2]),
                        "crop_path": None,
                        "text": "",
                        "region_type": normalized_style_type(item.get("region_type", "mixed")),
                        "local_box_xyxy": [int(v) for v in item["box"]],
                        "local_polygon": item.get("poly"),
                        "score": float(item.get("score", 0.0)),
                    }
                    for idx, item in enumerate(group["items"])
                ],
                "text": "",
            }
        )

    return regions


def bbox_polygon(box):
    x1, y1, x2, y2 = [int(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def union_box(items):
    x1 = min(int(item["box"][0]) for item in items)
    y1 = min(int(item["box"][1]) for item in items)
    x2 = max(int(item["box"][2]) for item in items)
    y2 = max(int(item["box"][3]) for item in items)
    return x1, y1, x2, y2


def join_region_texts(parts, text_key="final_text", include_dropped=False):
    texts = []
    for part in parts:
        if part.get("dropped") and not include_dropped:
            continue
        text = part.get(text_key)
        if text is None and text_key == "final_text":
            text = part.get("text", "")
        elif text is None:
            text = ""
        text = normalize_output_text(text)
        if text:
            texts.append(text)
    return " ".join(texts)


def build_lines_from_style_regions(parent_lines):
    lines = []
    for parent in sorted(parent_lines, key=lambda item: int(item["index"])):
        parts = sorted(parent.get("parts", []), key=lambda item: item.get("local_box_xyxy", [0])[0])
        kept_parts = [part for part in parts if not part.get("dropped")]
        raw_text = join_region_texts(parts, text_key="text", include_dropped=True)
        text = join_region_texts(parts, text_key="final_text", include_dropped=False)
        region_types = [part.get("region_type", "unknown") for part in parts]
        line = {
            "index": int(parent["index"]),
            "line_id": int(parent["index"]),
            "poly": parent["poly"],
            "box": parent["box"],
            "score": float(parent.get("score", 0.0)),
            "crop_path": parent["crop_path"],
            "line_crop_path": parent["line_crop_path"],
            "segments": [
                {
                    "index": int(part.get("index", idx)),
                    "x1": int(part.get("local_box_xyxy", [0, 0, 0, 0])[0]),
                    "x2": int(part.get("local_box_xyxy", [0, 0, 0, 0])[2]),
                    "crop_path": part.get("crop_path"),
                    "text": part.get("text", ""),
                    "raw_text": part.get("text", ""),
                    "final_text": part.get("final_text", ""),
                    "region_type": part.get("region_type", "unknown"),
                    "recognizer": part.get("recognizer", "htr"),
                    "source": part.get("source", "htr"),
                    "box_xyxy": [int(v) for v in part["box"]],
                    "polygon": part.get("poly"),
                    "local_box_xyxy": part.get("local_box_xyxy"),
                    "dropped": bool(part.get("dropped", False)),
                }
                for idx, part in enumerate(parts)
            ],
            "parts": [
                {
                    "region_id": int(part.get("index", idx)),
                    "parent_line_id": int(parent["index"]),
                    "box_xyxy": [int(v) for v in part["box"]],
                    "polygon": part.get("poly"),
                    "local_box_xyxy": part.get("local_box_xyxy"),
                    "local_polygon": part.get("local_polygon"),
                    "region_type": part.get("region_type", "unknown"),
                    "recognizer": part.get("recognizer", "htr"),
                    "ocr_text": part.get("text", ""),
                    "raw_text": part.get("text", ""),
                    "final_text": part.get("final_text", ""),
                    "source": part.get("source", "htr"),
                    "crop_path": part.get("crop_path"),
                    "dropped": bool(part.get("dropped", False)),
                    "uncertain": bool(part.get("uncertain", False)),
                    "qwen_verdict": part.get("qwen_verdict"),
                }
                for idx, part in enumerate(parts)
            ],
            "raw_text": raw_text,
            "text": text,
            "final_text": text,
            "field_type": infer_field_type(text),
            "dropped": not bool(text.strip()),
            "uncertain": any(part.get("uncertain") for part in parts),
            "source": "style_metadata_merge",
            "recognizer": "htr_style_regions",
            "final_confidence": float(np.mean([float(part.get("final_confidence", 0.0)) for part in kept_parts])) if kept_parts else 0.0,
            "need_qwen_verify": False,
            "qwen_priority": 0,
            "qwen_reasons": [],
            "qwen_verdict": None,
            "crop_features": parent.get("crop_features", {}),
            "region_types": region_types,
            "region_ids": [int(part.get("index", idx)) for idx, part in enumerate(parts)],
        }
        lines.append(line)
    return lines


def process_pages(args):
    output_dir = Path(args.out)
    paths = prepare_output_dirs(output_dir, clean_output=args.clean_output)
    crop_root = paths["line_crops"]
    debug_root = paths["debug_boxes"]
    debug_json_root = paths["debug_json"]
    text_root = paths["texts"]
    raw_text_root = paths["texts_raw"]
    json_root = paths["json"]
    metadata_root = paths["metadata"]
    lines_root = paths["lines"]
    regions_root = paths["regions"]

    detector = create_paddle_detector(args)
    recognizer = LineRecognizer(args)
    print(f"HTR device: {recognizer.device}")
    charset_metadata = {
        "source": recognizer.charset_source,
        "charset": recognizer.charset,
        "charset_size": recognizer.charset_size,
        "ctc_blank": 1,
        "needed_cls": recognizer.needed_cls,
        "checkpoint_output_classes": recognizer.checkpoint_nb_cls,
        "model_nb_cls": recognizer.nb_cls,
        "checkpoint": args.checkpoint,
    }
    (metadata_root / "viet_hf_charset.json").write_text(
        json.dumps(charset_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    qwen_model = None
    qwen_processor = None
    htr_released_for_qwen = False
    if args.qwen_enabled and args.qwen_max_calls_per_page <= 0:
        print("Qwen verifier enabled but max calls per page is 0; skipping model load.")

    total_pages = 0
    total_lines = 0
    total_qwen_calls = 0
    started_at = time.time()

    for img_path in source_images(args.source):
        page_started_at = time.time()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] Cannot read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        page_name = img_path.stem
        results = detector.predict(str(img_path))
        line_items = paddle_result_to_line_items(results[0], img_w, img_h, args.pad_x_ratio, args.pad_y_ratio) if results else []

        page_crop_dir = crop_root / page_name
        page_crop_dir.mkdir(parents=True, exist_ok=True)
        htr_crops = []
        htr_refs = []
        parent_lines = []
        valid_regions = []

        for idx, item in enumerate(line_items):
            item["index"] = idx
            crop = crop_by_paddle_poly(img, item["poly"])
            if crop is None:
                crop = safe_crop(img, *item["box"])
            if crop is None:
                continue

            line_prefix = f"line_{idx:03d}"
            full_line_path = page_crop_dir / f"{line_prefix}_full.png"
            cv2.imwrite(str(full_line_path), crop)
            item["line_crop_path"] = str(full_line_path.as_posix())
            item["crop_features"] = crop_features(crop)
            item["region_type"] = classify_text_region(item["crop_features"])
            item["crop_path"] = item["line_crop_path"]
            item["parts"] = []
            item["text"] = ""
            item["recognizer"] = "htr_style_regions"

            style_regions = detect_style_regions_in_line(
                detector,
                crop,
                full_line_path,
                item,
                page_crop_dir,
                line_prefix,
                args,
            )
            if not style_regions:
                continue

            for region in style_regions:
                region["index"] = len(valid_regions)
                item["parts"].append(region)
                valid_regions.append(region)
                region_crop = cv2.imread(region["crop_path"])
                if region_crop is not None:
                    htr_crops.append(region_crop)
                    htr_refs.append(region)

            parent_lines.append(item)

        htr_texts = recognizer.predict_crops(htr_crops, batch_size=args.htr_batch_size)
        for region, text in zip(htr_refs, htr_texts):
            region["text"] = text
            region["segments"][0]["text"] = text

        initialize_final_line_state(valid_regions)
        raw_region_records = make_line_records(valid_regions)
        for region in valid_regions:
            apply_rule_cleanup(region)
        qwen_context_text = "\n".join(region.get("final_text", region.get("text", "")) for region in valid_regions if not region.get("dropped"))
        qwen_candidates = []
        qwen_verdicts = []
        if args.qwen_enabled and args.qwen_max_calls_per_page > 0:
            qwen_candidates = select_qwen_candidates(
                valid_regions,
                max_calls_per_page=args.qwen_max_calls_per_page,
                verify_all_lines=args.qwen_verify_all_lines,
            )
        if qwen_candidates:
            from qwen import load_model as load_qwen_model
            from qwen import verify_line

            if not htr_released_for_qwen:
                recognizer.model.to("cpu")
                htr_released_for_qwen = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if qwen_model is None:
                print(f"Loading Qwen verifier: {args.qwen_model}")
                qwen_model, qwen_processor = load_qwen_model(args.qwen_model)

            for region in qwen_candidates:
                qwen_input_text = region.get("final_text", region.get("text", ""))
                verdict = verify_line(
                    qwen_model,
                    qwen_processor,
                    region["line_crop_path"],
                    qwen_input_text,
                    region.get("field_type", "unknown"),
                    qwen_context_text,
                )
                apply_qwen_verdict(region, verdict, min_confidence_to_accept=args.qwen_min_confidence_to_accept)
                qwen_verdicts.append(
                    {
                        "region_id": int(region["index"]),
                        "crop_path": region["line_crop_path"],
                        "ocr_text": qwen_input_text,
                        "field_type": region.get("field_type", "unknown"),
                        "region_type": region.get("region_type", "unknown"),
                        "verdict": verdict,
                    }
                )
            total_qwen_calls += len(qwen_verdicts)
            del qwen_model
            del qwen_processor
            qwen_model = None
            qwen_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if htr_released_for_qwen:
                recognizer.model.to(recognizer.device)
                htr_released_for_qwen = False

        for region in valid_regions:
            apply_rule_cleanup(region)
            refine_region_type_after_recognition(region)

        lines = build_lines_from_style_regions(parent_lines)

        raw_page_text = "\n".join(normalize_output_text(item.get("raw_text", item.get("text", ""))) for item in lines)
        page_text = "\n".join(normalize_output_text(item["final_text"]) for item in lines if not item.get("dropped"))
        (raw_text_root / f"{page_name}.txt").write_text(raw_page_text + ("\n" if raw_page_text else ""), encoding="utf-8")
        (text_root / f"{page_name}.txt").write_text(page_text + ("\n" if page_text else ""), encoding="utf-8")

        line_records = make_line_records(lines)
        region_records = make_line_records(valid_regions)
        clean_line_records = make_clean_line_outputs(line_records)
        clean_region_records = make_region_outputs(region_records)

        metadata = {
            "page": str(img_path.as_posix()),
            "page_name": page_name,
            "mode": args.mode,
            "image_size": {"width": int(img_w), "height": int(img_h)},
            "method": "paddleocr_v5_line_detection_style_split_htr_metadata_merge_qwen_garbage_filter",
            "fallback_method": "polygon_bounding_rectangle",
            "detector_model_name": args.det_model_name,
            "detector_model_dir": args.det_model_dir,
            "det_thresh": args.det_thresh,
            "det_box_thresh": args.det_box_thresh,
            "det_unclip_ratio": args.det_unclip_ratio,
            "pad_x_ratio": args.pad_x_ratio,
            "pad_y_ratio": args.pad_y_ratio,
            "split_line_by_style": bool(args.split_line_by_style),
            "style_split_use_paddle_subboxes": bool(args.style_split_use_paddle_subboxes),
            "style_split_min_score": args.style_split_min_score,
            "style_split_min_width_ratio": args.style_split_min_width_ratio,
            "style_split_min_height_ratio": args.style_split_min_height_ratio,
            "style_split_min_area_ratio": args.style_split_min_area_ratio,
            "style_split_merge_gap_ratio": args.style_split_merge_gap_ratio,
            "mixed_split": {
                "min_component_area": args.mixed_split_min_component_area,
                "mark_height_ratio": args.mixed_split_mark_height_ratio,
                "mark_width_ratio": args.mixed_split_mark_width_ratio,
                "mark_attach_ratio": args.mixed_split_mark_attach_ratio,
                "handwriting_height_ratio": args.mixed_split_handwriting_height_ratio,
                "handwriting_width_ratio": args.mixed_split_handwriting_width_ratio,
                "handwriting_area_factor": args.mixed_split_handwriting_area_factor,
                "printed_max_height_ratio": args.mixed_split_printed_max_height_ratio,
                "printed_max_width_ratio": args.mixed_split_printed_max_width_ratio,
                "merge_gap_ratio": args.mixed_split_merge_gap_ratio,
                "max_runs": args.mixed_split_max_runs,
                "min_run_width_ratio": args.mixed_split_min_run_width_ratio,
                "min_run_width_px": args.mixed_split_min_run_width_px,
                "pad_x_ratio": args.mixed_split_pad_x_ratio,
                "pad_y_ratio": args.mixed_split_pad_y_ratio,
            },
            "form_split": {
                "min_gap_ratio": args.form_split_min_gap_ratio,
                "min_gap_px": args.form_split_min_gap_px,
                "separator_window_ratio": args.form_split_separator_window_ratio,
                "min_mark_count": args.form_split_min_mark_count,
                "mark_min_y_ratio": args.form_split_mark_min_y_ratio,
                "min_prefix_ratio": args.form_split_min_prefix_ratio,
                "min_tail_ratio": args.form_split_min_tail_ratio,
                "tail_height_factor": args.form_split_tail_height_factor,
                "tail_y_delta_ratio": args.form_split_tail_y_delta_ratio,
            },
            "paddle_device": args.paddle_device or "auto",
            "htr_device": str(recognizer.device),
            "checkpoint": args.checkpoint,
            "htr_charset": {
                "source": recognizer.charset_source,
                "charset_size": recognizer.charset_size,
                "needed_cls": recognizer.needed_cls,
                "checkpoint_output_classes": recognizer.checkpoint_nb_cls,
                "model_nb_cls": recognizer.nb_cls,
                "metadata_path": str((metadata_root / "viet_hf_charset.json").as_posix()),
            },
            "qwen": {
                "enabled": bool(args.qwen_enabled),
                "model": args.qwen_model if args.qwen_enabled else None,
                "max_calls_per_page": args.qwen_max_calls_per_page,
                "min_confidence_to_accept": args.qwen_min_confidence_to_accept,
                "verify_all_lines": bool(args.qwen_verify_all_lines),
                "mode": "garbage_filter_only",
                "num_calls": len(qwen_verdicts),
            },
            "line_count": len(lines),
            "parent_line_count": len(parent_lines),
            "region_count": len(valid_regions),
            "segment_count": len(valid_regions),
            "num_printed_regions": sum(1 for item in valid_regions if item.get("region_type") == "printed"),
            "num_handwriting_regions": sum(1 for item in valid_regions if item.get("region_type") == "handwriting"),
            "num_mixed_regions": sum(1 for item in valid_regions if item.get("region_type") == "mixed"),
            "num_qwen_calls": len(qwen_verdicts),
            "num_dropped": sum(1 for item in valid_regions if item.get("dropped")),
            "num_replaced": sum(1 for item in valid_regions if item.get("source") == "rule_replace"),
            "num_qwen_replaced": 0,
            "num_rule_replaced": sum(1 for item in valid_regions if item.get("source") == "rule_replace"),
            "num_qwen_dropped": sum(1 for item in valid_regions if item.get("source") == "qwen_drop"),
            "num_rule_dropped": sum(1 for item in valid_regions if item.get("source") == "rule_drop"),
            "num_uncertain": sum(1 for item in valid_regions if item.get("uncertain")),
            "raw_text": raw_page_text,
            "text": page_text,
            "full_text": page_text,
            "clean_lines": clean_line_records,
            "clean_regions": clean_region_records,
            "regions": region_records,
            "lines": line_records,
        }
        (json_root / f"{page_name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (metadata_root / f"{page_name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (lines_root / f"{page_name}.json").write_text(
            json.dumps({"page_name": page_name, "lines": clean_line_records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (regions_root / f"{page_name}.json").write_text(
            json.dumps({"page_name": page_name, "regions": clean_region_records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_tsv(lines_root / f"{page_name}.tsv", clean_line_records, ["line_id", "text"])
        write_tsv(
            regions_root / f"{page_name}.tsv",
            clean_region_records,
            ["line_id", "region_id", "region_type", "text", "crop_path", "local_box_xyxy", "box_xyxy"],
        )
        (debug_json_root / f"{page_name}_ocr_regions_raw.json").write_text(json.dumps(raw_region_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_json_root / f"{page_name}_ocr_lines_raw.json").write_text(json.dumps(line_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_json_root / f"{page_name}_qwen_candidates.json").write_text(json.dumps(
            [
                {
                    "region_id": int(region["index"]),
                    "ocr_text": region.get("text", ""),
                    "field_type": region.get("field_type", "unknown"),
                    "region_type": region.get("region_type", "unknown"),
                    "priority": int(region.get("qwen_priority", 0)),
                    "reasons": region.get("qwen_reasons", []),
                    "crop_path": region.get("line_crop_path"),
                }
                for idx, region in enumerate(qwen_candidates)
            ],
            ensure_ascii=False,
            indent=2,
        ), encoding="utf-8")
        (debug_json_root / f"{page_name}_qwen_verdicts.json").write_text(json.dumps(qwen_verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_json_root / f"{page_name}_final_regions.json").write_text(json.dumps(region_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_json_root / f"{page_name}_final_lines.json").write_text(json.dumps(line_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_json_root / f"{page_name}_page_result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        save_debug(img, lines, debug_root / f"{page_name}_boxes.jpg")
        save_status_debug(img, region_records, debug_root / f"{page_name}_qwen_candidates.jpg", mode="candidate")
        save_status_debug(img, line_records, debug_root / f"{page_name}_final_lines.jpg", mode="final")
        save_style_region_debug(img, region_records, debug_root / f"{page_name}_style_regions.jpg")

        print(f"[OK] {page_name}: {len(lines)} merged lines, {len(valid_regions)} detected regions, {len(qwen_verdicts)} Qwen calls in {time.time() - page_started_at:.2f}s")
        total_pages += 1
        total_lines += len(lines)

    print("\nDone.")
    print(f"Pages processed: {total_pages}")
    print(f"Total merged lines: {total_lines}")
    print(f"Total Qwen calls: {total_qwen_calls}")
    print(f"Text output: {text_root}")
    print(f"Raw text output: {raw_text_root}")
    print(f"Clean line output: {lines_root}")
    print(f"Region output: {regions_root}")
    print(f"JSON output: {json_root}")
    print(f"Metadata output: {metadata_root}")
    print(f"Debug boxes: {debug_root}")
    print(f"Debug JSON: {debug_json_root}")
    print(f"Elapsed: {time.time() - started_at:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Full-page Vietnamese handwriting OCR: PaddleOCR line detection + HTR-ViTRNN recognition.")
    parser.add_argument("--source", required=True, help="Page image file/folder")
    parser.add_argument("--out", default="./outputs_dococr", help="Output folder")
    parser.add_argument("--mode", choices=["FAST", "BALANCED", "ACCURATE"], default="BALANCED")
    clean_group = parser.add_mutually_exclusive_group()
    clean_group.add_argument("--clean-output", dest="clean_output", action="store_true", help="Remove this pipeline's generated subfolders inside --out before running")
    clean_group.add_argument("--no-clean-output", dest="clean_output", action="store_false", help="Keep existing files inside --out")
    parser.set_defaults(clean_output=True)
    parser.add_argument("--checkpoint", default="./output/viet_lr1e4_64_layers_1024_dim/best_CER.pth")
    parser.add_argument(
        "--charset-file",
        default="./output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json",
        help="Production charset JSON/TXT. Document OCR never scans train data for charset.",
    )
    parser.add_argument("--nb-cls", default=512, type=int)
    parser.add_argument("--img-size", default=[512, 64], type=int, nargs=2)
    parser.add_argument("--num_layers_RNN", default=1, type=int)
    parser.add_argument("--hidden_dim_RNN", default=1024, type=int)
    parser.add_argument("--device", default=None, help="PyTorch HTR device, default uses cuda:0 when available")
    parser.add_argument("--pad-x-ratio", type=float, default=0.0)
    parser.add_argument("--pad-y-ratio", type=float, default=0.08)
    parser.add_argument("--htr-batch-size", type=int, default=64)
    parser.add_argument("--paddle-device", default="gpu:0", help="PaddleOCR detection device, e.g. gpu:0 or cpu")
    parser.add_argument("--text-det-limit-side-len", type=int, default=1536)
    parser.add_argument("--text-det-limit-type", default="max")
    parser.add_argument("--det-model-name", default="PP-OCRv5_server_det", help="PaddleOCR v5 detection model name")
    parser.add_argument("--det-model-dir", default=None, help="Optional local PaddleOCR detection model directory")
    parser.add_argument("--det-thresh", type=float, default=0.2, help="Paddle text detection pixel threshold")
    parser.add_argument("--det-box-thresh", type=float, default=0.4, help="Paddle text detection box threshold")
    parser.add_argument("--det-unclip-ratio", type=float, default=2.2, help="Paddle text detection unclip ratio")
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument("--split-line-by-style", dest="split_line_by_style", action="store_true", help="Split only confirmed mixed lines into printed/handwriting runs before HTR")
    style_group.add_argument("--no-split-line-by-style", dest="split_line_by_style", action="store_false", help="Send each full line crop directly to HTR")
    parser.set_defaults(split_line_by_style=True)
    parser.add_argument("--style-split-use-paddle-subboxes", action="store_true", help="Fallback to PaddleOCRv5 sub-boxes when component mixed split cannot confirm a printed/handwriting split")
    parser.add_argument("--style-split-min-score", type=float, default=0.20, help="Minimum Paddle detection score for a style sub-box")
    parser.add_argument("--style-split-min-width-ratio", type=float, default=0.015, help="Minimum style sub-box width as a ratio of line crop width")
    parser.add_argument("--style-split-min-height-ratio", type=float, default=0.35, help="Minimum style sub-box height as a ratio of line crop height")
    parser.add_argument("--style-split-min-area-ratio", type=float, default=0.002, help="Minimum style sub-box area as a ratio of line crop area")
    parser.add_argument("--style-split-merge-gap-ratio", type=float, default=0.025, help="Maximum gap for merging adjacent same-style sub-boxes")
    parser.add_argument("--style-split-pad-x-ratio", type=float, default=0.01)
    parser.add_argument("--style-split-pad-y-ratio", type=float, default=0.08)
    parser.add_argument("--mixed-split-min-component-area", type=int, default=4)
    parser.add_argument("--mixed-split-mark-height-ratio", type=float, default=0.18)
    parser.add_argument("--mixed-split-mark-width-ratio", type=float, default=0.22)
    parser.add_argument("--mixed-split-mark-attach-ratio", type=float, default=0.35)
    parser.add_argument("--mixed-split-handwriting-height-ratio", type=float, default=0.36)
    parser.add_argument("--mixed-split-handwriting-width-ratio", type=float, default=0.55)
    parser.add_argument("--mixed-split-handwriting-area-factor", type=float, default=1.70)
    parser.add_argument("--mixed-split-printed-max-height-ratio", type=float, default=0.42)
    parser.add_argument("--mixed-split-printed-max-width-ratio", type=float, default=0.75)
    parser.add_argument("--mixed-split-merge-gap-ratio", type=float, default=0.035)
    parser.add_argument("--mixed-split-max-runs", type=int, default=3)
    parser.add_argument("--mixed-split-min-run-width-ratio", type=float, default=0.06)
    parser.add_argument("--mixed-split-min-run-width-px", type=int, default=18)
    parser.add_argument("--mixed-split-pad-x-ratio", type=float, default=0.03)
    parser.add_argument("--mixed-split-pad-y-ratio", type=float, default=0.12)
    parser.add_argument("--form-split-min-gap-ratio", type=float, default=0.045)
    parser.add_argument("--form-split-min-gap-px", type=int, default=6)
    parser.add_argument("--form-split-separator-window-ratio", type=float, default=0.75)
    parser.add_argument("--form-split-min-mark-count", type=int, default=2)
    parser.add_argument("--form-split-mark-min-y-ratio", type=float, default=0.25)
    parser.add_argument("--form-split-min-prefix-ratio", type=float, default=0.10)
    parser.add_argument("--form-split-min-tail-ratio", type=float, default=0.12)
    parser.add_argument("--form-split-tail-height-factor", type=float, default=1.25, help="Minimum height ratio for accepting a handwriting tail after printed text")
    parser.add_argument("--form-split-tail-y-delta-ratio", type=float, default=0.18, help="Minimum vertical overhang ratio for accepting a handwriting tail when it is not taller")
    qwen_group = parser.add_mutually_exclusive_group()
    qwen_group.add_argument("--qwen-enabled", dest="qwen_enabled", action="store_true", help="Enable Qwen verifier after HTR")
    qwen_group.add_argument("--no-qwen", dest="qwen_enabled", action="store_false", help="Disable Qwen verifier")
    parser.set_defaults(qwen_enabled=False)
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Qwen VL verifier model")
    parser.add_argument("--qwen-max-calls-per-page", type=int, default=None, help="Top-K suspicious regions checked by Qwen garbage filter")
    parser.add_argument("--qwen-min-confidence-to-accept", type=float, default=0.80)
    parser.add_argument("--qwen-verify-all-lines", action="store_true", help="Run Qwen garbage filtering on every detected region, capped by max calls")
    parser.add_argument("--paddle-lang", default="en", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mode == "FAST":
        args.qwen_enabled = False
    if args.qwen_max_calls_per_page is None:
        args.qwen_max_calls_per_page = 16 if args.mode == "ACCURATE" else 12

    if not Path(args.source).exists():
        raise SystemExit(f"Source not found: {args.source}")
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    process_pages(args)


if __name__ == "__main__":
    main()
