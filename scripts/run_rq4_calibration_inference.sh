#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
MANIFEST="/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/train.jsonl"
TEACHER_DIR="/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2"
OUTPUT_DIR="/ext_data/casper/asr_data/rq4_calibration_c0_train_v1"
MIN_FREE_GPU_MB="${RQ4_MIN_FREE_GPU_MB:-20000}"

required_paths=(
    "$PYTHON"
    "$ROOT_DIR/generate_rq4_calibration_predictions.py"
    "$ROOT_DIR/documents/data/rq2_rq4_parameter_sweep_amendment_v2.md"
    "$MANIFEST"
    "$TEACHER_DIR/model.safetensors"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing RQ4 calibration prerequisite: $path" >&2
        exit 1
    fi
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "RQ4 calibration inference is already complete: $OUTPUT_DIR"
    exit 0
fi

free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < MIN_FREE_GPU_MB )); then
    echo "Insufficient free GPU memory: ${free_gpu_mb:-unknown} MB; require $MIN_FREE_GPU_MB MB" >&2
    exit 1
fi

exec "$PYTHON" "$ROOT_DIR/generate_rq4_calibration_predictions.py" \
    --manifest "$MANIFEST" \
    --teacher-dir "$TEACHER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --chunk-size 256 \
    --max-length 225 \
    --seed 42 \
    --device cuda