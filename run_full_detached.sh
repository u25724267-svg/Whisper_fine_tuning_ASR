#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${WHISPER_CONFIG:-$ROOT_DIR/configs/whisper-base-shona-3epochs.json}"
LOG_DIR="$ROOT_DIR/logs"
PYTHON="$ROOT_DIR/.venv/bin/python"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but was not found." >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Experiment configuration not found at $CONFIG_FILE" >&2
    exit 1
fi

experiment_name="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["experiment_name"])' "$CONFIG_FILE")"
configured_output="${WHISPER_OUTPUT_DIR:-$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$CONFIG_FILE")}"
required_free_gpu_mb="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("resources", {}).get("min_free_gpu_mb", 0))' "$CONFIG_FILE")"
SESSION_NAME="${TMUX_SESSION_NAME:-$experiment_name}"
if [[ "$configured_output" = /* ]]; then
    OUTPUT_DIR="$configured_output"
else
    OUTPUT_DIR="$ROOT_DIR/$configured_output"
fi
LOG_FILE="$LOG_DIR/$experiment_name.log"

if (( required_free_gpu_mb > 0 )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required to validate GPU capacity." >&2
        exit 1
    fi
    free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if (( free_gpu_mb < required_free_gpu_mb )); then
        echo "Insufficient free GPU memory: ${free_gpu_mb} MiB available, ${required_free_gpu_mb} MiB required." >&2
        exit 1
    fi
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

mkdir -p "$LOG_DIR"
printf '\n===== launch %s =====\n' "$(date --iso-8601=seconds)" >> "$LOG_FILE"

train_command=("$PYTHON" -u "$ROOT_DIR/train_full.py" --config "$CONFIG_FILE" --output-dir "$OUTPUT_DIR")
latest_checkpoint=""
if [[ -d "$OUTPUT_DIR" ]]; then
    latest_checkpoint="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -1)"
fi
if [[ -n "$latest_checkpoint" ]]; then
    train_command+=(--resume-from-checkpoint "$latest_checkpoint")
fi

printf -v quoted_command '%q ' "${train_command[@]}"
printf -v quoted_log '%q' "$LOG_FILE"
tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" \
    "set -o pipefail; ${quoted_command}2>&1 | tee -a ${quoted_log}"
tmux set-window-option -t "$SESSION_NAME" remain-on-exit on >/dev/null

echo "Started detached session: $SESSION_NAME"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
if [[ -n "$latest_checkpoint" ]]; then
    echo "Resuming: $latest_checkpoint"
fi
echo "Attach: tmux attach -t $SESSION_NAME"