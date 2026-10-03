#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
export ASR_OUTPUT_ROOT="$OUTPUT_ROOT"
export HF_HOME="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"

if (( $# > 1 )) || [[ -n "${1:-}" && "${1:-}" != "--preflight-only" ]]; then
    echo "Usage: $0 [--preflight-only]" >&2
    exit 1
fi

ordered_dirs="$("$PYTHON" "$ROOT_DIR/asr.py" prepare_medium_queue --list)"
mapfile -t experiments <<<"$ordered_dirs"
if (( ${#experiments[@]} != 27 )); then
    echo "Expected 27 Medium runs, found ${#experiments[@]}." >&2
    exit 1
fi
for experiment in "${experiments[@]}"; do
    if [[ ! -x "$ROOT_DIR/$experiment/run.sh" ]]; then
        echo "Missing executable launcher: $experiment/run.sh" >&2
        exit 1
    fi
done

required_disk_gib="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["estimated_disk_reserve_gib"])' "$ROOT_DIR/experiments/medium/queue.json")"
disk_parent="$OUTPUT_ROOT"
while [[ ! -d "$disk_parent" ]]; do disk_parent="$(dirname "$disk_parent")"; done
free_disk_kib="$(df -Pk "$disk_parent" | awk 'NR == 2 {print $4}')"
if (( free_disk_kib < required_disk_gib * 1024 * 1024 )); then
    echo "Insufficient external disk for queue reserve: require $required_disk_gib GiB." >&2
    exit 1
fi
echo "Preflight passed: 27 Medium runs; ${required_disk_gib} GiB external reserve."
if [[ "${1:-}" == "--preflight-only" ]]; then
    echo "No training launched."
    exit 0
fi

session="whisper-medium-rq1-rq2"
if tmux has-session -t "$session" 2>/dev/null; then
    echo "Medium controller already exists: $session" >&2
    exit 1
fi
free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < 20000 )); then
    echo "Require at least 20000 MiB free GPU memory; got ${free_gpu_mb:-unknown}." >&2
    exit 1
fi
log_dir="$ROOT_DIR/artifacts/experiment_outputs/medium/logs"
mkdir -p "$log_dir"
log_file="$log_dir/sequence-$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v sequence_command '%q ' env \
    "ASR_OUTPUT_ROOT=$OUTPUT_ROOT" \
    "WHISPER_HF_HOME=$HF_HOME" \
    "$ROOT_DIR/scripts/run_sequence.sh" "${experiments[@]}"
printf -v quoted_log '%q' "$log_file"
tmux new-session -d -s "$session" -c "$ROOT_DIR" \
    "set -o pipefail; ${sequence_command}2>&1 | tee -a ${quoted_log}"
tmux set-window-option -t "$session" remain-on-exit on >/dev/null
echo "Started $session; log: $log_file"