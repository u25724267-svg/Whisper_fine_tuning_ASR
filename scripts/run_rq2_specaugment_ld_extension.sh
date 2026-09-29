#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$ROOT_DIR/scripts/run_sequence.sh" \
    experiments/rq2/s4_specaug_ld_random_seed42 \
    experiments/rq2/s5_specaug_ld_sortagrad_seed42 \
    experiments/rq2/s4_specaug_ld_random_seed43 \
    experiments/rq2/s5_specaug_ld_sortagrad_seed43 \
    experiments/rq2/s4_specaug_ld_random_seed44 \
    experiments/rq2/s5_specaug_ld_sortagrad_seed44