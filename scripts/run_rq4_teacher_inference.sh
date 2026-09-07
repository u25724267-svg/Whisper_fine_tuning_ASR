#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
MANIFEST="/ext_data/casper/asr_data/waxal_shona_unlabeled_admission_v1/eligible.jsonl"
ADMISSION_SUMMARY="/ext_data/casper/asr_data/waxal_shona_unlabeled_admission_v1/summary.json"
AUDIT_ROOT="/ext_data/casper/asr_data/waxal_shona_unlabeled_v1"
TEACHER_DIR="/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2"
OUTPUT_DIR="/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1"
MIN_FREE_GPU_MB="${RQ4_MIN_FREE_GPU_MB:-20000}"
MIN_FREE_DISK_GB="${RQ4_MIN_FREE_DISK_GB:-20}"

required_paths=(
    "$PYTHON"
    "$ROOT_DIR/generate_rq4_teacher_labels.py"
    "$ROOT_DIR/documents/data/rq4_pseudolabel_protocol_v1.md"
    "$MANIFEST"
    "$ADMISSION_SUMMARY"
    "$AUDIT_ROOT/inventory_full.json"
    "$TEACHER_DIR/model.safetensors"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing RQ4 teacher-inference prerequisite: $path" >&2
        exit 1
    fi
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "RQ4 teacher inference is already complete: $OUTPUT_DIR"
    exit 0
fi

free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < MIN_FREE_GPU_MB )); then
    echo "Insufficient free GPU memory: ${free_gpu_mb:-unknown} MB; require $MIN_FREE_GPU_MB MB" >&2
    exit 1
fi

free_disk_kb="$(df -Pk "${OUTPUT_DIR%/*}" | awk 'NR==2 {print $4}')"
required_disk_kb=$((MIN_FREE_DISK_GB * 1024 * 1024))
if [[ ! "$free_disk_kb" =~ ^[0-9]+$ ]] || (( free_disk_kb < required_disk_kb )); then
    echo "Insufficient free disk for RQ4 teacher inference" >&2
    exit 1
fi

exec "$PYTHON" "$ROOT_DIR/generate_rq4_teacher_labels.py" \
    --manifest "$MANIFEST" \
    --admission-summary "$ADMISSION_SUMMARY" \
    --audit-root "$AUDIT_ROOT" \
    --teacher-dir "$TEACHER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --chunk-size 256 \
    --max-length 225 \
    --seed 42 \
    --device cuda