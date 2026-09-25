#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
UPSTREAM_SESSION="rq1-c1-sortagrad-seeds43-44"

if tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    while [[ "$(tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead}')" != "1" ]]; do
        echo "Waiting for upstream SortaGrad controller: $UPSTREAM_SESSION"
        read -r -t 30 || true
    done
    upstream_status="$(tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead_status}')"
    if [[ "$upstream_status" != "0" ]]; then
        echo "Upstream SortaGrad controller failed with status $upstream_status" >&2
        exit 1
    fi
fi

for seed in 43 44; do
    summary="$OUTPUT_ROOT/rq1/c1_sortagrad_seed${seed}/item_predictions/summary.json"
    if [[ ! -f "$summary" ]]; then
        echo "Missing completed upstream SortaGrad prediction summary: $summary" >&2
        exit 1
    fi
done

exec "$ROOT_DIR/scripts/run_sequence.sh" \
    experiments/rq1/c2_snr_seed43 \
    experiments/rq1/c2_snr_seed44 \
    experiments/rq1/c3_snr_duration_seed44 \
    experiments/rq1/c6_wer_margin_seed43 \
    experiments/rq1/c6_wer_margin_seed44