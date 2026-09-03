#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_FILE="${1:?Usage: run_experiment.sh CONFIG_FILE SESSION_NAME}"
SESSION_NAME="${2:?Usage: run_experiment.sh CONFIG_FILE SESSION_NAME}"
ASR_OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
HF_CACHE_DIR="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Experiment configuration not found at $CONFIG_FILE" >&2
    exit 1
fi

relative_output="$($PYTHON -c '
import json
import pathlib
import sys

value = pathlib.PurePosixPath(json.load(open(sys.argv[1]))["output_dir"])
if value.is_absolute() or ".." in value.parts or str(value) in {"", "."}:
    raise SystemExit("output_dir must be a non-empty relative path without ..")
print(value)
' "$CONFIG_FILE")"
runner="$($PYTHON -c '
import json
import sys

runner = json.load(open(sys.argv[1])).get("runner", "train_full.py")
allowed = {
    "train_full.py",
    "train_s2s_curriculum.py",
    "train_snr_curriculum.py",
    "train_sortagrad.py",
}
if runner not in allowed:
    raise SystemExit(f"Unsupported experiment runner: {runner}")
print(runner)
' "$CONFIG_FILE")"
RUNNER_FILE="$ROOT_DIR/$runner"
if [[ ! -f "$RUNNER_FILE" ]]; then
    echo "Experiment runner not found at $RUNNER_FILE" >&2
    exit 1
fi
OUTPUT_DIR="$ASR_OUTPUT_ROOT/$relative_output"
LOG_DIR="$OUTPUT_DIR/logs"
WANDB_DIR="$OUTPUT_DIR/wandb"
LOG_FILE="$LOG_DIR/train.log"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists; refusing to overwrite or resume: $OUTPUT_DIR" >&2
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
fi

required_free_gpu_mb="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("resources", {}).get("min_free_gpu_mb", 0))' "$CONFIG_FILE")"
required_free_disk_gb="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("resources", {}).get("min_free_disk_gb", 0))' "$CONFIG_FILE")"

if (( required_free_gpu_mb > 0 )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required to validate GPU capacity." >&2
        exit 1
    fi
    free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if (( free_gpu_mb < required_free_gpu_mb )); then
        echo "Insufficient GPU memory: ${free_gpu_mb} MiB free, ${required_free_gpu_mb} MiB required." >&2
        exit 1
    fi
fi

mkdir -p "$ASR_OUTPUT_ROOT"
if (( required_free_disk_gb > 0 )); then
    free_disk_kb="$(df -Pk "$ASR_OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
    required_disk_kb=$((required_free_disk_gb * 1024 * 1024))
    if (( free_disk_kb < required_disk_kb )); then
        echo "Insufficient disk: ${free_disk_kb} KiB free, ${required_disk_kb} KiB required." >&2
        exit 1
    fi
fi

mkdir -p "$LOG_DIR" "$WANDB_DIR"
train_command=(
    env
    "HF_HOME=$HF_CACHE_DIR"
    "WANDB_DIR=$WANDB_DIR"
    "$PYTHON"
    -u
    "$RUNNER_FILE"
    --config "$CONFIG_FILE"
    --output-dir "$OUTPUT_DIR"
)
printf -v quoted_command '%q ' "${train_command[@]}"
printf -v quoted_log '%q' "$LOG_FILE"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" \
    "set -o pipefail; ${quoted_command}2>&1 | tee -a ${quoted_log}"
tmux set-window-option -t "$SESSION_NAME" remain-on-exit on >/dev/null

echo "Started experiment: $SESSION_NAME"
echo "Config: $CONFIG_FILE"
echo "Runner: $RUNNER_FILE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"