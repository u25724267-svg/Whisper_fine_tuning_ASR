#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
exec "$ROOT_DIR/scripts/run_experiment.sh" "$EXPERIMENT_DIR/config.json" "rq1-c5-s2s-margin-seed43"