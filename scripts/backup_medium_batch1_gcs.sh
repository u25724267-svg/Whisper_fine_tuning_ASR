#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
BACKUP_DATE="${ASR_BACKUP_DATE:-$(date -u +%F)}"
GCS_ROOT="${ASR_GCS_BACKUP_ROOT:-gs://casper/completed-runs/$BACKUP_DATE}"
LOG_DIR="${ASR_BACKUP_LOG_DIR:-$ROOT_DIR/artifacts/experiment_outputs/medium/logs/backup-batch1-$(date -u +%Y%m%dT%H%M%SZ)}"
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export CLOUDSDK_STORAGE_PROCESS_COUNT=2
export CLOUDSDK_STORAGE_THREAD_COUNT=4

mkdir -p "$LOG_DIR"
trap 'status=$?; printf "backup_exit_status=%s\n" "$status" | tee "$LOG_DIR/status.txt"' EXIT

ordered_outputs="$("$PYTHON" -c '
import json
import sys

queue = json.load(open(sys.argv[1]))
assert len(queue["runs"]) == 27, "Unexpected Medium queue size"
print("\n".join(entry["output_dir"] for entry in queue["runs"][6:16]))
' "$ROOT_DIR/experiments/medium/queue.json")"
mapfile -t outputs <<< "$ordered_outputs"
if (( ${#outputs[@]} != 10 )); then
    echo "Expected ten batch-one runs." >&2
    exit 1
fi

for relative_output in "${outputs[@]}"; do
    if [[ "$relative_output" != medium/* || "$relative_output" == *..* ]]; then
        echo "Unsafe output path: $relative_output" >&2
        exit 1
    fi
    for file in train_results.json validation_results.json test_results.json model.safetensors \
        trainer_state.json item_predictions/summary.json item_predictions/validation.jsonl \
        item_predictions/test.jsonl checkpoint-2302/model.safetensors \
        checkpoint-4604/model.safetensors checkpoint-6906/model.safetensors; do
        if [[ ! -s "$OUTPUT_ROOT/$relative_output/$file" ]]; then
            echo "Incomplete source: $relative_output/$file" >&2
            exit 1
        fi
    done
    if [[ ! -s "$ROOT_DIR/artifacts/experiment_outputs/$relative_output/artifact_manifest.json" ]]; then
        echo "Missing completed mirror: $relative_output" >&2
        exit 1
    fi
done

printf 'Destination: %s\nLogs: %s\n' "$GCS_ROOT" "$LOG_DIR"
for relative_output in "${outputs[@]}"; do
    source_dir="$OUTPUT_ROOT/$relative_output"
    destination="${GCS_ROOT%/}/$relative_output/"
    run_name="${relative_output##*/}"
    echo "Uploading $relative_output"
    gcloud storage rsync "$source_dir/" "$destination" \
        --recursive --ignore-symlinks --exclude='(^|.*/)(secrets(/.*)?|[.]env)$' --quiet \
        2>&1 | tee "$LOG_DIR/$run_name.upload.log"
    echo "Verifying $relative_output"
    verification_log="$LOG_DIR/$run_name.verify.log"
    gcloud storage rsync "$source_dir/" "$destination" \
        --recursive --ignore-symlinks --exclude='(^|.*/)(secrets(/.*)?|[.]env)$' \
        --checksums-only --dry-run --quiet 2>&1 | tee "$verification_log"
    if grep -Eq 'Would (copy|delete|update)' "$verification_log"; then
        echo "Verification proposed changes: $relative_output" >&2
        exit 1
    fi
    printf '%s VERIFIED\n' "$relative_output" | tee -a "$LOG_DIR/verified-runs.txt"
done

echo "All ten batch-one runs backed up and checksum-verified. Local outputs retained."