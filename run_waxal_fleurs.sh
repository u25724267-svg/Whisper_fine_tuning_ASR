#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
DATA_DIR="/ext_data/casper/whisper_data/waxal_fleurs/sna_asr"
CONFIG_FILE="$ROOT_DIR/configs/whisper-base-shona-waxal-fleurs-3epochs-seed42.json"
HF_CACHE_DIR="/ext_data/casper/huggingface_cache"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$DATA_DIR/preparation_summary.json" ]]; then
    HF_HOME="$HF_CACHE_DIR" "$PYTHON" "$ROOT_DIR/prepare_waxal_fleurs.py" \
        --output-dir "$DATA_DIR"
fi

WHISPER_CONFIG="$CONFIG_FILE" \
WHISPER_HF_HOME="$HF_CACHE_DIR" \
TMUX_SESSION_NAME="whisper-base-shona-waxal-fleurs-3epochs-seed42" \
exec "$ROOT_DIR/run_full_detached.sh"