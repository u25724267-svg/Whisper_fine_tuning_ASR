#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
ARTIFACT_ROOT="$ROOT_DIR/artifacts/experiment_outputs"
HF_HOME_VALUE="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"
FLEURS_ROOT="${FLEURS_PROTOCOL_ROOT:-/ext_data/casper/asr_data/fleurs_shona_corrected_v2}"
FLEURS_VALIDATION="$FLEURS_ROOT/validation.jsonl"
FLEURS_TEST="$FLEURS_ROOT/test.jsonl"
MIN_FREE_GPU_MB="${FLEURS_MIN_FREE_GPU_MB:-20000}"
PREFLIGHT_ONLY=false

if (( $# > 1 )); then
    echo "Usage: $0 [--preflight-only]" >&2
    exit 1
fi
if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=true
elif [[ -n "${1:-}" ]]; then
    echo "Unknown argument: $1" >&2
    exit 1
fi

experiments=(
    "experiments/rq1/c1_sortagrad_seed42|rq1/c1_sortagrad_seed42|checkpoint-6906"
    "experiments/rq1/c1_sortagrad_seed43|rq1/c1_sortagrad_seed43|."
    "experiments/rq1/c1_sortagrad_seed44|rq1/c1_sortagrad_seed44|."
    "experiments/rq2/s2_specaug_lb_random_seed42|rq2/s2_specaug_lb_random_seed42|."
    "experiments/rq2/s3_specaug_lb_sortagrad_seed42|rq2/s3_specaug_lb_sortagrad_seed42|."
    "experiments/rq2/s2_specaug_lb_random_seed43|rq2/s2_specaug_lb_random_seed43|."
    "experiments/rq2/s3_specaug_lb_sortagrad_seed43|rq2/s3_specaug_lb_sortagrad_seed43|."
    "experiments/rq2/s2_specaug_lb_random_seed44|rq2/s2_specaug_lb_random_seed44|."
    "experiments/rq2/s3_specaug_lb_sortagrad_seed44|rq2/s3_specaug_lb_sortagrad_seed44|."
    "experiments/rq2/s4_specaug_ld_random_seed42|rq2/s4_specaug_ld_random_seed42|."
    "experiments/rq2/s5_specaug_ld_sortagrad_seed42|rq2/s5_specaug_ld_sortagrad_seed42|."
    "experiments/rq2/s4_specaug_ld_random_seed43|rq2/s4_specaug_ld_random_seed43|."
    "experiments/rq2/s5_specaug_ld_sortagrad_seed43|rq2/s5_specaug_ld_sortagrad_seed43|."
    "experiments/rq2/s4_specaug_ld_random_seed44|rq2/s4_specaug_ld_random_seed44|."
    "experiments/rq2/s5_specaug_ld_sortagrad_seed44|rq2/s5_specaug_ld_sortagrad_seed44|."
)

required_files=(
    "$PYTHON"
    "$ROOT_DIR/evaluate_predictions.py"
    "$ROOT_DIR/mirror_experiment_artifacts.py"
    "$ROOT_DIR/documents/data/rq2_specaugment_fleurs_protocol.md"
    "$FLEURS_ROOT/preparation_summary.json"
    "$FLEURS_VALIDATION"
    "$FLEURS_TEST"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing SpecAugment FLEURS prerequisite: $path" >&2
        exit 1
    fi
done

for specification in "${experiments[@]}"; do
    IFS='|' read -r experiment_dir relative_output model_relative <<<"$specification"
    config_path="$ROOT_DIR/$experiment_dir/config.json"
    run_dir="$OUTPUT_ROOT/$relative_output"
    model_dir="$run_dir/$model_relative"
    prediction_dir="$run_dir/fleurs_corrected_v2"
    artifact_dir="$ARTIFACT_ROOT/$relative_output"
    if [[ ! -f "$config_path" ]]; then
        echo "Missing experiment config: $config_path" >&2
        exit 1
    fi
    if [[ ! -f "$model_dir/model.safetensors" ]]; then
        echo "Missing selected model: $model_dir" >&2
        exit 1
    fi
    if [[ ! -d "$artifact_dir" ]]; then
        echo "Missing repository artifact run: $artifact_dir" >&2
        exit 1
    fi
    if [[ -e "$prediction_dir" && ! -f "$prediction_dir/summary.json" ]]; then
        echo "Incomplete FLEURS output exists; refusing overwrite: $prediction_dir" >&2
        exit 1
    fi
done

free_gpu_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
)"
if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < MIN_FREE_GPU_MB )); then
    echo "Insufficient free GPU memory: ${free_gpu_mb:-unknown} MB; require $MIN_FREE_GPU_MB MB" >&2
    exit 1
fi

echo "Preflight passed for ${#experiments[@]} frozen checkpoints."
if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
    exit 0
fi

for specification in "${experiments[@]}"; do
    IFS='|' read -r experiment_dir relative_output model_relative <<<"$specification"
    config_path="$ROOT_DIR/$experiment_dir/config.json"
    run_dir="$OUTPUT_ROOT/$relative_output"
    model_dir="$run_dir/$model_relative"
    prediction_dir="$run_dir/fleurs_corrected_v2"
    artifact_dir="$ARTIFACT_ROOT/$relative_output"
    experiment_name="${relative_output//\//-}"

    echo "===== $experiment_name ====="
    if [[ -f "$prediction_dir/summary.json" ]]; then
        echo "Corrected FLEURS predictions already complete; skipping inference."
    else
        mkdir -p "$run_dir/logs"
        HF_HOME="$HF_HOME_VALUE" "$PYTHON" -u "$ROOT_DIR/evaluate_predictions.py" \
            --config "$config_path" \
            --model-dir "$model_dir" \
            --output-dir "$prediction_dir" \
            --manifest "fleurs_validation=$FLEURS_VALIDATION" \
            --manifest "fleurs_test=$FLEURS_TEST" \
            2>&1 | tee -a "$run_dir/logs/fleurs_corrected_v2.log"
    fi
    if [[ ! -f "$prediction_dir/summary.json" ]]; then
        echo "Evaluation exited without a summary: $prediction_dir" >&2
        exit 1
    fi

    "$PYTHON" "$ROOT_DIR/mirror_experiment_artifacts.py" \
        --source-dir "$run_dir" \
        --output-dir "$artifact_dir" \
        --refresh
    if [[ ! -f "$artifact_dir/fleurs_corrected_v2/summary.json" ]]; then
        echo "Repository FLEURS mirror is incomplete: $artifact_dir" >&2
        exit 1
    fi
    echo "Completed and mirrored $experiment_name"
done

echo "All confirmatory SpecAugment FLEURS evaluations completed successfully."