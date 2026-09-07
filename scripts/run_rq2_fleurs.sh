#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
HF_HOME_VALUE="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"
FLEURS_ROOT="${FLEURS_PROTOCOL_ROOT:-/ext_data/casper/asr_data/fleurs_shona_corrected_v2}"
FLEURS_VALIDATION="$FLEURS_ROOT/validation.jsonl"
FLEURS_TEST="$FLEURS_ROOT/test.jsonl"

experiments=(
    "experiments/rq1/c0_random_seed42_v2|rq1/c0_random_seed42_v2"
    "experiments/rq1/c0_random_seed43_v2|rq1/c0_random_seed43_v2"
    "experiments/rq1/c0_random_seed44_v2|rq1/c0_random_seed44_v2"
    "experiments/rq1/c3_snr_duration_seed42|rq1/c3_snr_duration_seed42"
    "experiments/rq2/a2_waveform_random_seed42|rq2/a2_waveform_random_seed42"
    "experiments/rq2/a2_waveform_random_seed43|rq2/a2_waveform_random_seed43"
    "experiments/rq2/a2_waveform_random_seed44|rq2/a2_waveform_random_seed44"
    "experiments/rq2/a3_waveform_c3_seed42|rq2/a3_waveform_c3_seed42"
    "experiments/rq2/a3_waveform_c3_seed43|rq2/a3_waveform_c3_seed43"
    "experiments/rq2/a3_waveform_c3_seed44|rq2/a3_waveform_c3_seed44"
)

required_files=(
    "$PYTHON"
    "$ROOT_DIR/evaluate_predictions.py"
    "$FLEURS_ROOT/preparation_summary.json"
    "$FLEURS_VALIDATION"
    "$FLEURS_TEST"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing FLEURS evaluation prerequisite: $path" >&2
        exit 1
    fi
done

for specification in "${experiments[@]}"; do
    IFS='|' read -r experiment_dir relative_output <<<"$specification"
    config_path="$ROOT_DIR/$experiment_dir/config.json"
    model_dir="$OUTPUT_ROOT/$relative_output"
    prediction_dir="$model_dir/fleurs_corrected_v2"
    if [[ ! -f "$config_path" ]]; then
        echo "Missing experiment config: $config_path" >&2
        exit 1
    fi
    if [[ ! -f "$model_dir/model.safetensors" ]]; then
        echo "Missing trained model: $model_dir" >&2
        exit 1
    fi
    if [[ -e "$prediction_dir" && ! -f "$prediction_dir/summary.json" ]]; then
        echo "Incomplete FLEURS output exists; refusing overwrite: $prediction_dir" >&2
        exit 1
    fi
done

echo "Preflight passed for ${#experiments[@]} distinct frozen checkpoints."

for specification in "${experiments[@]}"; do
    IFS='|' read -r experiment_dir relative_output <<<"$specification"
    config_path="$ROOT_DIR/$experiment_dir/config.json"
    model_dir="$OUTPUT_ROOT/$relative_output"
    prediction_dir="$model_dir/fleurs_corrected_v2"
    experiment_name="${relative_output//\//-}"

    echo "===== $experiment_name ====="
    if [[ -f "$prediction_dir/summary.json" ]]; then
        echo "FLEURS predictions already complete; skipping."
        continue
    fi
    mkdir -p "$model_dir/logs"
    HF_HOME="$HF_HOME_VALUE" "$PYTHON" -u "$ROOT_DIR/evaluate_predictions.py" \
        --config "$config_path" \
        --model-dir "$model_dir" \
        --output-dir "$prediction_dir" \
        --manifest "fleurs_validation=$FLEURS_VALIDATION" \
        --manifest "fleurs_test=$FLEURS_TEST" \
        2>&1 | tee -a "$model_dir/logs/fleurs_corrected_v2.log"
    if [[ ! -f "$prediction_dir/summary.json" ]]; then
        echo "Evaluation exited without a summary: $prediction_dir" >&2
        exit 1
    fi
done

echo "All frozen RQ2 checkpoints have corrected FLEURS predictions."
echo "STOP: aggregate and interpret OOD results before launching another stage."