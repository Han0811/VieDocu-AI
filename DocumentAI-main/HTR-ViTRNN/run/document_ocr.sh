#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/mpeclab/torch-env/bin/python}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
QWEN_MAX_CALLS=${QWEN_MAX_CALLS:-12}
QWEN_MIN_CONF=${QWEN_MIN_CONF:-0.80}
QWEN_VERIFY_ALL=${QWEN_VERIFY_ALL:-0}
CHARSET_FILE=${CHARSET_FILE:-./output/viet_lr1e4_64_layers_1024_dim/viet_hf_charset.json}

QWEN_EXTRA_ARGS=()
if [[ "$QWEN_VERIFY_ALL" == "1" ]]; then
  QWEN_EXTRA_ARGS+=(--qwen-verify-all-lines)
fi

"$PYTHON" document_ocr.py \
--source ../img \
--out outputs_dococr_paddle \
--mode BALANCED \
--checkpoint ./output/viet_lr1e4_64_layers_1024_dim/best_CER.pth \
--charset-file "$CHARSET_FILE" \
--img-size 512 64 \
--nb-cls 512 \
--num_layers_RNN 1 \
--hidden_dim_RNN 1024 \
--pad-x-ratio 0.0 \
--pad-y-ratio 0.08 \
--htr-batch-size 64 \
--paddle-device gpu:0 \
--text-det-limit-side-len 1536 \
--text-det-limit-type max \
--det-model-name PP-OCRv5_server_det \
--det-thresh 0.2 \
--det-box-thresh 0.4 \
--det-unclip-ratio 2.2 \
--split-line-by-style \
--qwen-enabled \
--qwen-model "$QWEN_MODEL" \
--qwen-max-calls-per-page "$QWEN_MAX_CALLS" \
--qwen-min-confidence-to-accept "$QWEN_MIN_CONF" \
"${QWEN_EXTRA_ARGS[@]}"
