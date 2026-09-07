#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
DATA_ROOT="${ASR_DATA_ROOT:-/ext_data/casper/asr_data}"
ASSET_ROOT="$DATA_ROOT/augmentation/slr28/RIRS_NOISES"
ARCHIVE="$DATA_ROOT/augmentation/slr28/rirs_noises.zip"
SOURCE_MANIFEST="$DATA_ROOT/waxal_shona_speaker_disjoint_v2/validation.jsonl"
FOLLOWUP_ROOT="$DATA_ROOT/rq2_followup_v1"
PARTITION_ROOT="$FOLLOWUP_ROOT/asset_partitions"
VALIDATION_ROOT="$FOLLOWUP_ROOT/validation_battery"
ARCHIVE_SHA256="3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"
SEED="${RQ2_FOLLOWUP_VALIDATION_SEED:-42}"

required_paths=(
    "$PYTHON"
    "$ROOT_DIR/prepare_rq2_asset_partitions.py"
    "$ROOT_DIR/prepare_rq2_followup_data.py"
    "$ASSET_ROOT"
    "$ARCHIVE"
    "$SOURCE_MANIFEST"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing RQ2 follow-up prerequisite: $path" >&2
        exit 1
    fi
done

run_immutable() {
    local output_dir="$1"
    shift
    if [[ -f "$output_dir/summary.json" ]]; then
        echo "Already complete; skipping $output_dir"
        return
    fi
    local staging_dir="${output_dir%/*}/.${output_dir##*/}.staging"
    if [[ -e "$output_dir" || -e "$staging_dir" ]]; then
        echo "Incomplete immutable output exists: $output_dir" >&2
        echo "Inspect it manually; this launcher will not delete or overwrite it." >&2
        exit 1
    fi
    "$@"
}

run_immutable "$PARTITION_ROOT" \
    "$PYTHON" "$ROOT_DIR/prepare_rq2_asset_partitions.py" \
    --asset-root "$ASSET_ROOT" \
    --archive "$ARCHIVE" \
    --expected-archive-sha256 "$ARCHIVE_SHA256" \
    --output-dir "$PARTITION_ROOT"

mkdir -p "$VALIDATION_ROOT"

materialize_validation() {
    local name="$1"
    local condition="$2"
    shift 2
    local output_dir="$VALIDATION_ROOT/$name"
    run_immutable "$output_dir" \
        "$PYTHON" "$ROOT_DIR/prepare_rq2_followup_data.py" \
        --source-manifest "$SOURCE_MANIFEST" \
        --output-dir "$output_dir" \
        --condition "$condition" \
        --seed "$SEED" \
        "$@"
}

materialize_validation d0_clean_source_v2 clean

for snr_db in 5 10 15; do
    materialize_validation "dn_${snr_db}db_no_loop_v2" noise \
        --snr-db "$snr_db" \
        --asset-root "$ASSET_ROOT" \
        --asset-manifest "$PARTITION_ROOT/validation.jsonl" \
        --asset-partition validation
done

materialize_validation dr_rir rir \
    --asset-root "$ASSET_ROOT" \
    --asset-manifest "$PARTITION_ROOT/validation.jsonl" \
    --asset-partition validation

for snr_db in 5 10 15; do
    materialize_validation "dnr_${snr_db}db_no_loop_v2" noise-rir \
        --snr-db "$snr_db" \
        --asset-root "$ASSET_ROOT" \
        --asset-manifest "$PARTITION_ROOT/validation.jsonl" \
        --asset-partition validation
done

cat <<EOF
RQ2 follow-up validation assets are complete under:
$FOLLOWUP_ROOT

STOP: audit summary hashes and corruption metrics before creating B-series
training manifests. This script does not open test assets, train models, or
contact W&B.
EOF