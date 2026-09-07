#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at $PYTHON" >&2
    exit 1
fi

if (( $# == 0 )); then
    echo "Usage: $0 EXPERIMENT_DIR [EXPERIMENT_DIR ...]" >&2
    echo "Only already-implemented experiment directories are accepted." >&2
    exit 1
fi

read_config_value() {
    local config_path="$1"
    local expression="$2"
    "$PYTHON" - "$config_path" "$expression" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
value = config
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

wait_for_session() {
    local session_name="$1"
    while true; do
        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            echo "Experiment session disappeared before its status could be read: $session_name" >&2
            exit 1
        fi
        local pane_dead
        pane_dead="$(tmux display-message -p -t "$session_name" '#{pane_dead}')"
        if [[ "$pane_dead" == "1" ]]; then
            break
        fi
        printf 'Waiting for %s...\n' "$session_name"
        read -r -t 30 || true
    done
    local status
    status="$(tmux display-message -p -t "$session_name" '#{pane_dead_status}' 2>/dev/null || true)"
    if [[ "$status" != "0" ]]; then
        echo "Experiment session $session_name exited with status ${status:-unknown}." >&2
        exit 1
    fi
}

training_complete() {
    local output_dir="$1"
    [[ -f "$output_dir/train_results.json" \
        && -f "$output_dir/validation_results.json" \
        && -f "$output_dir/test_results.json" \
        && -f "$output_dir/model.safetensors" ]]
}

experiment_dirs=()
for requested_dir in "$@"; do
    experiment_dir="$(realpath -m "$ROOT_DIR/$requested_dir")"
    config_path="$experiment_dir/config.json"
    run_script="$experiment_dir/run.sh"
    if [[ ! -f "$config_path" || ! -x "$run_script" ]]; then
        echo "Unimplemented experiment: $requested_dir" >&2
        echo "Expected executable run.sh and config.json in $experiment_dir" >&2
        echo "No experiments were launched. Implement and validate every requested stage first." >&2
        exit 1
    fi
    read_config_value "$config_path" experiment_name >/dev/null
    read_config_value "$config_path" output_dir >/dev/null
    experiment_dirs+=("$experiment_dir")
done

echo "Preflight passed for ${#experiment_dirs[@]} implemented experiment(s)."

for experiment_dir in "${experiment_dirs[@]}"; do
    config_path="$experiment_dir/config.json"
    run_script="$experiment_dir/run.sh"

    experiment_name="$(read_config_value "$config_path" experiment_name)"
    relative_output="$(read_config_value "$config_path" output_dir)"
    output_dir="$OUTPUT_ROOT/$relative_output"
    session_name="$experiment_name"
    model_dir="$output_dir"
    prediction_dir="$output_dir/item_predictions"

    echo "===== $experiment_name ====="
    if ! training_complete "$output_dir"; then
        if tmux has-session -t "$session_name" 2>/dev/null; then
            pane_dead="$(tmux display-message -p -t "$session_name" '#{pane_dead}')"
            if [[ "$pane_dead" == "1" ]]; then
                status="$(tmux display-message -p -t "$session_name" '#{pane_dead_status}')"
                echo "Incomplete output with dead session $session_name (status $status)." >&2
                echo "Refusing to relaunch or overwrite $output_dir." >&2
                exit 1
            fi
            echo "Waiting for existing training session: $session_name"
        else
            if [[ -e "$output_dir" ]]; then
                echo "Incomplete output exists without a live session: $output_dir" >&2
                echo "Refusing to relaunch or overwrite it." >&2
                exit 1
            fi
            echo "Launching $experiment_name"
            "$run_script"
        fi
        wait_for_session "$session_name"
        if ! training_complete "$output_dir"; then
            echo "Session exited successfully but required training artifacts are missing: $output_dir" >&2
            exit 1
        fi
    else
        echo "Training artifacts already complete; skipping training."
    fi

    if [[ ! -f "$prediction_dir/summary.json" ]]; then
        prediction_session="${session_name}-item-predictions"
        if tmux has-session -t "$prediction_session" 2>/dev/null; then
            echo "Waiting for existing prediction session: $prediction_session"
        else
            echo "Exporting item-level predictions for $experiment_name"
            prediction_command=(
                env
                "HF_HOME=${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"
                "$PYTHON"
                -u
                "$ROOT_DIR/evaluate_predictions.py"
                --config "$config_path"
                --model-dir "$model_dir"
                --output-dir "$prediction_dir"
                --splits validation test
            )
            printf -v quoted_prediction '%q ' "${prediction_command[@]}"
            printf -v quoted_prediction_log '%q' "$output_dir/logs/predictions.log"
            tmux new-session -d -s "$prediction_session" -c "$ROOT_DIR" \
                "set -o pipefail; ${quoted_prediction}2>&1 | tee -a ${quoted_prediction_log}"
            tmux set-window-option -t "$prediction_session" remain-on-exit on >/dev/null
        fi
        wait_for_session "$prediction_session"
        if [[ ! -f "$prediction_dir/summary.json" ]]; then
            echo "Prediction session exited without summary: $prediction_dir" >&2
            exit 1
        fi
    else
        echo "Item-level prediction artifacts already complete; skipping export."
    fi
    echo "Completed $experiment_name"
done

echo "All requested experiments completed successfully."