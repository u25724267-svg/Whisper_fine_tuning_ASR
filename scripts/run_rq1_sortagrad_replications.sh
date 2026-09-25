#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$ROOT_DIR/scripts/run_sequence.sh" \
    experiments/rq1/c1_sortagrad_seed43 \
    experiments/rq1/c1_sortagrad_seed44