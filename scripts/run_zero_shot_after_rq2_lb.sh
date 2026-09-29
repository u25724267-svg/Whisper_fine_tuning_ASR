#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_SESSION="rq2-specaugment-lb-factorial"

if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    echo "Required upstream controller is missing: $UPSTREAM_SESSION" >&2
    exit 1
fi

while [[ "$(tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead}')" != "1" ]]; do
    echo "Waiting for upstream RQ2 LB controller: $UPSTREAM_SESSION"
    read -r -t 30 || true
done

upstream_status="$(
    tmux display-message -p -t "$UPSTREAM_SESSION" '#{pane_dead_status}'
)"
if [[ "$upstream_status" != "0" ]]; then
    echo "Upstream RQ2 LB controller failed with status $upstream_status" >&2
    exit 1
fi

exec "$ROOT_DIR/scripts/run_whisper_base_zero_shot.sh"