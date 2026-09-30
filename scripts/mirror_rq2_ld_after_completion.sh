#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
ASR_CLI="$ROOT_DIR/asr.py"
SOURCE_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
ARTIFACT_ROOT="$ROOT_DIR/artifacts/experiment_outputs"
UPSTREAM_SESSION="rq2-specaugment-ld-extension"

if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    echo "Required LD controller is missing: $UPSTREAM_SESSION" >&2
    exit 1
fi

while [[ "$(tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead}')" != "1" ]]; do
    echo "Waiting for upstream LD controller: $UPSTREAM_SESSION"
    read -r -t 30 || true
done

upstream_status="$(
    tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead_status}'
)"
if [[ "$upstream_status" != "0" ]]; then
    echo "Upstream LD controller failed with status $upstream_status" >&2
    exit 1
fi

runs=(
    rq2/s4_specaug_ld_random_seed42
    rq2/s5_specaug_ld_sortagrad_seed42
    rq2/s4_specaug_ld_random_seed43
    rq2/s5_specaug_ld_sortagrad_seed43
    rq2/s4_specaug_ld_random_seed44
    rq2/s5_specaug_ld_sortagrad_seed44
)
for relative_output in "${runs[@]}"; do
    echo "Mirroring $relative_output"
    "$PYTHON" "$ASR_CLI" mirror_experiment_artifacts \
        --source-dir "$SOURCE_ROOT/$relative_output" \
        --output-dir "$ARTIFACT_ROOT/$relative_output"
done

echo "All LD scientific outputs are mirrored in the repository."