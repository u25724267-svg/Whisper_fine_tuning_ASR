#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
HF_HOME_VALUE="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"
INPUT="/ext_data/casper/asr_data/rq4_calibration_c0_train_v1/predictions.jsonl"
OUTPUT_DIR="/ext_data/casper/asr_data/rq4_proxy_calibration_v1"
MIN_FREE_GPU_MB="${RQ4_PROXY_MIN_FREE_GPU_MB:-18000}"

required_paths=(
    "$PYTHON"
    "$ROOT_DIR/generate_rq4_proxy_predictions.py"
    "$ROOT_DIR/documents/data/rq4_seamless_proxy_review_v1.md"
    "$INPUT"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing RQ4 proxy-calibration prerequisite: $path" >&2
        exit 1
    fi
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "RQ4 proxy calibration is already complete: $OUTPUT_DIR"
    exit 0
fi

free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < MIN_FREE_GPU_MB )); then
    echo "Insufficient free GPU memory: ${free_gpu_mb:-unknown} MB; require $MIN_FREE_GPU_MB MB" >&2
    exit 1
fi

HF_HOME="$HF_HOME_VALUE" exec "$PYTHON" "$ROOT_DIR/generate_rq4_proxy_predictions.py" \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --chunk-size 64 \
    --max-new-tokens 256 \
    --seed 42 \
    --device cuda \
    --confirm-noncommercial-license