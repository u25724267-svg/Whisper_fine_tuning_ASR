#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
ASR_CLI="$ROOT_DIR/asr.py"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
FLEURS_ROOT="${FLEURS_PROTOCOL_ROOT:-/ext_data/casper/asr_data/fleurs_shona_corrected_v2}"
HF_HOME_VALUE="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"
ARTIFACT_ROOT="$ROOT_DIR/artifacts/experiment_outputs"
LOG_DIR="$ARTIFACT_ROOT/medium/logs"
SESSION="medium-fleurs-after-batch2"
QUEUE_PATH="$ROOT_DIR/experiments/medium/queue.json"
TRAINING_PREFIX="medium-batch2-gpu"
QUEUE_SIZE=27
START_INDEX=16
EXPECTED_RUNS=11
OUTPUT_PREFIX="medium/"
A100_CONTROLS=false

usage() {
    echo "Usage: $0 [--a100-controls] [--preflight-only | --controller | --local-only | --local-preflight-only]" >&2
    exit 2
}

mode="--launch"
mode_set=false
for argument in "$@"; do
    case "$argument" in
        --a100-controls)
            A100_CONTROLS=true
            QUEUE_PATH="$ROOT_DIR/experiments/medium_a100/queue.json"
            TRAINING_PREFIX="whisper-medium-a100-controls-gpu"
            QUEUE_SIZE=5
            START_INDEX=0
            EXPECTED_RUNS=5
            OUTPUT_PREFIX="medium_a100/"
            SESSION="medium-a100-fleurs-after-controls"
            LOG_DIR="$ARTIFACT_ROOT/medium_a100/logs"
            ;;
        --launch|--preflight-only|--controller|--local-only|--local-preflight-only)
            if [[ "$mode_set" == true ]]; then usage; fi
            mode="$argument"
            mode_set=true
            ;;
        *) usage ;;
    esac
done

for path in "$PYTHON" "$ASR_CLI" "$FLEURS_ROOT/preparation_summary.json" \
    "$FLEURS_ROOT/validation.jsonl" "$FLEURS_ROOT/test.jsonl"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing FLEURS prerequisite: $path" >&2
        exit 1
    fi
done

ordered_outputs="$("$PYTHON" - "$QUEUE_PATH" "$OUTPUT_ROOT" "$ARTIFACT_ROOT" "$mode" "$START_INDEX" "$QUEUE_SIZE" "$OUTPUT_PREFIX" "$EXPECTED_RUNS" <<'PY'
import json
import sys
from pathlib import Path

queue = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(queue["runs"]) == int(sys.argv[6]), "Unexpected frozen queue size"
local_only = sys.argv[4] in {"--local-only", "--local-preflight-only"}
outputs = []
for run in queue["runs"] if local_only else queue["runs"][int(sys.argv[5]):]:
    output = run["output_dir"]
    if local_only:
        directory = Path(sys.argv[2]) / output
        required = ("model.safetensors", "train_results.json", "validation_results.json",
                    "test_results.json", "item_predictions/summary.json")
        if not all((directory / name).is_file() for name in required):
            continue
        if not (Path(sys.argv[3]) / output / "artifact_manifest.json").is_file():
            continue
    outputs.append(output)
assert outputs and len(set(outputs)) == len(outputs), "No eligible unique local models"
if not local_only:
    assert len(outputs) == int(sys.argv[8])
for output in outputs:
    assert output.startswith(sys.argv[7]) and ".." not in output.split("/")
print("\n".join(outputs))
PY
)"
mapfile -t outputs <<< "$ordered_outputs"

for relative_output in "${outputs[@]}"; do
    config_path="$ROOT_DIR/experiments/$relative_output/config.json"
    if [[ ! -f "$config_path" ]]; then
        echo "Missing Medium config: $config_path" >&2
        exit 1
    fi
    prediction_dir="$OUTPUT_ROOT/$relative_output/fleurs_corrected_v2"
    if [[ -d "$prediction_dir" && ! -f "$prediction_dir/summary.json" ]]; then
        echo "Incomplete FLEURS output exists; inspect before restarting: $prediction_dir" >&2
        exit 1
    fi
done

if [[ "$mode" == "--preflight-only" || "$mode" == "--local-preflight-only" ]]; then
    echo "Preflight passed for ${#outputs[@]} Medium checkpoints; no evaluation launched."
    printf '%s\n' "${outputs[@]}"
    exit 0
fi

if [[ "$mode" == "--launch" ]]; then
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "FLEURS controller already exists: $SESSION" >&2
        exit 1
    fi
    for lane in 0 1; do
        if ! tmux has-session -t "=$TRAINING_PREFIX$lane" 2>/dev/null; then
            echo "Missing training controller $TRAINING_PREFIX$lane" >&2
            exit 1
        fi
    done
    mkdir -p "$LOG_DIR"
    log_file="$LOG_DIR/$SESSION-$(date -u +%Y%m%dT%H%M%SZ).log"
    controller_arguments=(bash "$ROOT_DIR/scripts/run_medium_fleurs_after_batch2.sh")
    if [[ "$A100_CONTROLS" == true ]]; then controller_arguments+=(--a100-controls); fi
    printf -v controller_command '%q ' "${controller_arguments[@]}" --controller
    printf -v quoted_log '%q' "$log_file"
    tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" \
        "set -o pipefail; ${controller_command}2>&1 | tee -a ${quoted_log}"
    tmux set-window-option -t "$SESSION" remain-on-exit on >/dev/null
    echo "Queued FLEURS evaluation for ${#outputs[@]} runs in $SESSION"
    echo "Controller log: $log_file"
    exit 0
fi

lane_count=2
if [[ "$mode" == "--local-only" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        echo "Local-only evaluation requires exactly one GPU in CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    gpu_ids=("$CUDA_VISIBLE_DEVICES")
    lane_count=1
else
echo "Waiting for both $TRAINING_PREFIX training controllers to finish successfully."
for lane in 0 1; do
    session="$TRAINING_PREFIX$lane"
    while true; do
        if ! tmux has-session -t "=$session" 2>/dev/null; then
            echo "Training controller disappeared: $session" >&2
            exit 1
        fi
        pane_dead="$(tmux display-message -p -t "$session:0.0" '#{pane_dead}')"
        if [[ "$pane_dead" == "1" ]]; then break; fi
        sleep 30
    done
    status="$(tmux display-message -p -t "$session:0.0" '#{pane_dead_status}')"
    if [[ "$status" != "0" ]]; then
        echo "Training controller $session failed with status $status; FLEURS not started." >&2
        exit 1
    fi
    echo "$session completed successfully."
done

mapfile -t gpu_ids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if (( ${#gpu_ids[@]} < 2 )); then
    echo "Two GPUs are required for the two evaluation lanes." >&2
    exit 1
fi
fi
for (( lane=0; lane<lane_count; lane++ )); do
    if [[ "$A100_CONTROLS" == true ]]; then
        gpu_name="$(nvidia-smi --id="${gpu_ids[$lane]}" --query-gpu=name --format=csv,noheader)"
        if [[ "$gpu_name" != "NVIDIA A100-SXM4-40GB" ]]; then
            echo "Expected the A100 control device; got $gpu_name." >&2
            exit 1
        fi
        busy_gpu_ids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
        if printf '%s\n' "$busy_gpu_ids" | grep -Fxq "${gpu_ids[$lane]}"; then
            echo "A100 GPU lane $lane still has an active compute process; evaluation not started." >&2
            exit 1
        fi
    fi
    free_gpu_mb="$(nvidia-smi --id="${gpu_ids[$lane]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$free_gpu_mb" =~ ^[0-9]+$ ]] || (( free_gpu_mb < 20000 )); then
        echo "GPU $lane has only ${free_gpu_mb:-unknown} MiB free; require 20000 MiB." >&2
        exit 1
    fi
done

for relative_output in "${outputs[@]}"; do
    model_dir="$OUTPUT_ROOT/$relative_output"
    if [[ ! -s "$model_dir/model.safetensors" \
        || ! -s "$model_dir/item_predictions/summary.json" \
        || ! -s "$ARTIFACT_ROOT/$relative_output/artifact_manifest.json" ]]; then
        echo "Incomplete trained run; FLEURS not started: $relative_output" >&2
        exit 1
    fi
done

run_lane() {
    local lane="$1"
    local index relative_output run_dir prediction_dir artifact_dir
    export CUDA_VISIBLE_DEVICES="${gpu_ids[$lane]}"
    export HF_HOME="$HF_HOME_VALUE"
    for (( index=lane; index<${#outputs[@]}; index+=lane_count )); do
        relative_output="${outputs[$index]}"
        run_dir="$OUTPUT_ROOT/$relative_output"
        prediction_dir="$run_dir/fleurs_corrected_v2"
        artifact_dir="$ARTIFACT_ROOT/$relative_output"
        echo "Evaluating $relative_output on GPU lane $lane (${gpu_ids[$lane]})"
        mkdir -p "$run_dir/logs"
        if [[ ! -f "$prediction_dir/summary.json" ]]; then
            "$PYTHON" -u "$ASR_CLI" evaluate_predictions \
                --config "$ROOT_DIR/experiments/$relative_output/config.json" \
                --model-dir "$run_dir" \
                --output-dir "$prediction_dir" \
                --manifest "fleurs_validation=$FLEURS_ROOT/validation.jsonl" \
                --manifest "fleurs_test=$FLEURS_ROOT/test.jsonl" \
                > "$run_dir/logs/fleurs_corrected_v2.log" 2>&1
        fi
        if [[ ! -s "$prediction_dir/summary.json" ]]; then
            echo "FLEURS summary missing: $prediction_dir" >&2
            return 1
        fi
        "$PYTHON" "$ASR_CLI" mirror_experiment_artifacts \
            --source-dir "$run_dir" --output-dir "$artifact_dir" --refresh \
            > "$run_dir/logs/fleurs_mirror.log" 2>&1
        if [[ ! -s "$artifact_dir/fleurs_corrected_v2/summary.json" ]]; then
            echo "FLEURS mirror incomplete: $artifact_dir" >&2
            return 1
        fi
        echo "Completed and mirrored $relative_output"
    done
}

mkdir -p "$LOG_DIR"
if [[ "$mode" == "--local-only" ]]; then
    run_lane 0
    echo "All ${#outputs[@]} eligible local Medium FLEURS evaluations completed; no models restored or training launched."
    exit 0
fi
run_lane 0 > "$LOG_DIR/medium-fleurs-gpu0.log" 2>&1 &
pid0=$!
run_lane 1 > "$LOG_DIR/medium-fleurs-gpu1.log" 2>&1 &
pid1=$!
status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
echo "FLEURS lane statuses: GPU 0=$status0, GPU 1=$status1"
if (( status0 != 0 || status1 != 0 )); then exit 1; fi
echo "All ${#outputs[@]} queued Medium FLEURS evaluations completed."
