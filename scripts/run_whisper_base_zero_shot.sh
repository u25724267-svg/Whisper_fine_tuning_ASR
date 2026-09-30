#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
ASR_CLI="$ROOT_DIR/asr.py"
CONFIG="$ROOT_DIR/experiments/baselines/whisper_base_zero_shot/config.json"
OUTPUT_ROOT="$ROOT_DIR/artifacts/experiment_outputs"
OUTPUT_DIR="$OUTPUT_ROOT/baselines/whisper_base_zero_shot_waxal_fleurs_v1"
LOG_DIR="$OUTPUT_ROOT/baselines/logs"
FLEURS_ROOT="${FLEURS_PROTOCOL_ROOT:-/ext_data/casper/asr_data/fleurs_shona_corrected_v2}"
MODEL_ID="openai/whisper-base"
MODEL_REVISION="e37978b90ca9030d5170a5c07aadb050351a65bb"

required_files=(
    "$PYTHON"
    "$ASR_CLI"
    "$CONFIG"
    "/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/validation.jsonl"
    "/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/test.jsonl"
    "$FLEURS_ROOT/validation.jsonl"
    "$FLEURS_ROOT/test.jsonl"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing zero-shot prerequisite: $path" >&2
        exit 1
    fi
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "Whisper Base zero-shot evaluation already complete: $OUTPUT_DIR"
    exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Incomplete zero-shot output exists; refusing overwrite: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
HF_HOME="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}" \
    "$PYTHON" -u "$ASR_CLI" evaluate_predictions \
    --config "$CONFIG" \
    --model-id "$MODEL_ID" \
    --model-revision "$MODEL_REVISION" \
    --output-dir "$OUTPUT_DIR" \
    --splits validation test \
    --manifest "fleurs_validation=$FLEURS_ROOT/validation.jsonl" \
    --manifest "fleurs_test=$FLEURS_ROOT/test.jsonl" \
    2>&1 | tee -a "$LOG_DIR/whisper_base_zero_shot_waxal_fleurs_v1.log"

if [[ ! -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "Zero-shot evaluation exited without a summary: $OUTPUT_DIR" >&2
    exit 1
fi