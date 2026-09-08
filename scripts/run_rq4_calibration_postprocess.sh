#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CALIBRATION_INPUT="/ext_data/casper/asr_data/rq4_calibration_c0_train_v1/predictions.jsonl"
CALIBRATION_INPUT_SUMMARY="/ext_data/casper/asr_data/rq4_calibration_c0_train_v1/summary.json"
CALIBRATION_OUTPUT="/ext_data/casper/asr_data/rq4_diagnostic_calibration_v1"
SHORTLIST_OUTPUT="/ext_data/casper/asr_data/rq4_proxy_shortlist_v1"

required_paths=(
    "$PYTHON"
    "$ROOT_DIR/calibrate_rq4_teacher_diagnostics.py"
    "$ROOT_DIR/build_rq4_proxy_shortlist.py"
    "$ROOT_DIR/documents/data/rq2_rq4_parameter_sweep_amendment_v2.md"
    "$CALIBRATION_INPUT"
    "$CALIBRATION_INPUT_SUMMARY"
    "/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1/predictions.jsonl"
    "/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1/summary.json"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing completed RQ4 calibration prerequisite: $path" >&2
        exit 1
    fi
done

run_immutable() {
    local output_dir="$1"
    shift
    if [[ -f "$output_dir/summary.json" ]]; then
        echo "Already complete; skipping $output_dir"
        return
    fi
    local staging_dir="${output_dir%/*}/.${output_dir##*/}.staging"
    if [[ -e "$output_dir" || -e "$staging_dir" ]]; then
        echo "Incomplete immutable output exists: $output_dir" >&2
        exit 1
    fi
    "$@"
}

run_immutable "$CALIBRATION_OUTPUT" \
    "$PYTHON" "$ROOT_DIR/calibrate_rq4_teacher_diagnostics.py" \
    --input "$CALIBRATION_INPUT" \
    --output-dir "$CALIBRATION_OUTPUT" \
    --outer-folds 5 \
    --inner-folds 4 \
    --seed 42

run_immutable "$SHORTLIST_OUTPUT" \
    "$PYTHON" "$ROOT_DIR/build_rq4_proxy_shortlist.py" \
    --predictions /ext_data/casper/asr_data/rq4_teacher_labels_c0_v1/predictions.jsonl \
    --teacher-summary /ext_data/casper/asr_data/rq4_teacher_labels_c0_v1/summary.json \
    --calibration-dir "$CALIBRATION_OUTPUT" \
    --output-dir "$SHORTLIST_OUTPUT"

cat <<EOF
RQ4 grouped diagnostic calibration and proxy shortlist are complete.
STOP: inspect cross-validated AUCs and shortlist diversity before approving
SeamlessM4T-v2 licensing and proxy inference.
EOF