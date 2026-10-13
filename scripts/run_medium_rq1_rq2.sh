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
mapfile -t gpu_ids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if (( ${#gpu_ids[@]} < 2 )); then
    echo "Require two GPUs for independent Medium runs." >&2
    exit 1
fi
for gpu_index in 0 1; do
    free_gpu_mb="$(nvidia-smi --id="${gpu_ids[$gpu_index]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < 20000 )); then
        echo "GPU $gpu_index requires at least 20000 MiB free; got ${free_gpu_mb:-unknown}." >&2
        exit 1
    fi
done
echo "Preflight passed: 27 Medium runs; two single-GPU lanes; ${required_disk_gib} GiB external reserve."
if [[ "${1:-}" == "--preflight-only" ]]; then
    echo "No training launched."
    exit 0
fi

session="whisper-medium-rq1-rq2"
if tmux has-session -t "$session" 2>/dev/null; then
    echo "Medium controller already exists: $session" >&2
    exit 1
fi
for gpu_index in 0 1; do
    if tmux has-session -t "$session-gpu$gpu_index" 2>/dev/null; then
        echo "Medium controller already exists: $session-gpu$gpu_index" >&2
        exit 1
    fi
done
log_dir="$ROOT_DIR/artifacts/experiment_outputs/medium/logs"
mkdir -p "$log_dir"
for gpu_index in 0 1; do
    lane_experiments=()
    for (( experiment_index=gpu_index; experiment_index<${#experiments[@]}; experiment_index+=2 )); do
        lane_experiments+=("${experiments[$experiment_index]}")
    done
    lane_session="$session-gpu$gpu_index"
    log_file="$log_dir/sequence-gpu$gpu_index-$(date -u +%Y%m%dT%H%M%SZ).log"
    printf -v sequence_command '%q ' env \
        "CUDA_VISIBLE_DEVICES=${gpu_ids[$gpu_index]}" \
        "ASR_OUTPUT_ROOT=$OUTPUT_ROOT" \
        "WHISPER_HF_HOME=$HF_HOME" \
        "$ROOT_DIR/scripts/run_sequence.sh" "${lane_experiments[@]}"
    printf -v quoted_log '%q' "$log_file"
    tmux new-session -d -s "$lane_session" -c "$ROOT_DIR" \
        "set -o pipefail; ${sequence_command}2>&1 | tee -a ${quoted_log}"
    tmux set-window-option -t "$lane_session" remain-on-exit on >/dev/null
    echo "Started $lane_session on ${gpu_ids[$gpu_index]}; log: $log_file"
done