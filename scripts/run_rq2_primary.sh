#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
DATA_ROOT="${ASR_DATA_ROOT:-/ext_data/casper/asr_data}"
OUTPUT_ROOT="${ASR_OUTPUT_ROOT:-/ext_data/casper/asr_experiment_outputs}"
HF_HOME_VALUE="${WHISPER_HF_HOME:-/ext_data/casper/huggingface_cache}"
POLICY="$ROOT_DIR/configs/rq2-waveform-mild.json"
ASSET_ROOT="$DATA_ROOT/augmentation/slr28/RIRS_NOISES"
CLEAN_TRAIN="$DATA_ROOT/waxal_shona_speaker_disjoint_v2/train.jsonl"
COMPARISON_ROOT="$OUTPUT_ROOT/rq2/comparisons"
WORKERS="${RQ2_MATERIALIZATION_WORKERS:-8}"

required_paths=(
    "$PYTHON"
    "$ROOT_DIR/prepare_waveform_augmentation.py"
    "$ROOT_DIR/prepare_rq2_replication_configs.py"
    "$ROOT_DIR/compare_predictions_bootstrap.py"
    "$ROOT_DIR/scripts/run_sequence.sh"
    "$POLICY"
    "$CLEAN_TRAIN"
    "$ASSET_ROOT"
    "$OUTPUT_ROOT/rq1/c0_random_seed42_v2/item_predictions/summary.json"
    "$OUTPUT_ROOT/rq1/c0_random_seed43_v2/item_predictions/summary.json"
    "$OUTPUT_ROOT/rq1/c0_random_seed44_v2/item_predictions/summary.json"
    "$OUTPUT_ROOT/rq1/c3_snr_duration_seed42/item_predictions/summary.json"
    "$OUTPUT_ROOT/rq2/a2_waveform_random_seed42/item_predictions/summary.json"
    "$OUTPUT_ROOT/rq2/a3_waveform_c3_seed42/item_predictions/summary.json"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing RQ2 primary prerequisite: $path" >&2
        exit 1
    fi
done

mkdir -p "$COMPARISON_ROOT"

run_comparison() {
    local baseline="$1"
    local candidate="$2"
    local output="$3"
    local seed="$4"
    if [[ -f "$output" ]]; then
        echo "Comparison already complete; skipping $output"
        return
    fi
    "$PYTHON" "$ROOT_DIR/compare_predictions_bootstrap.py" \
        --baseline "$baseline" \
        --candidate "$candidate" \
        --output "$output" \
        --replicates 10000 \
        --seed "$seed"
}

compare_seed() {
    local seed="$1"
    local c0="$OUTPUT_ROOT/rq1/c0_random_seed${seed}_v2/item_predictions"
    local c3="$OUTPUT_ROOT/rq1/c3_snr_duration_seed42/item_predictions"
    local a2="$OUTPUT_ROOT/rq2/a2_waveform_random_seed${seed}/item_predictions"
    local a3="$OUTPUT_ROOT/rq2/a3_waveform_c3_seed${seed}/item_predictions"
    for split in validation test; do
        run_comparison \
            "$c0/$split.jsonl" "$a2/$split.jsonl" \
            "$COMPARISON_ROOT/a2_seed${seed}_vs_c0_seed${seed}_${split}.json" "$seed"
        run_comparison \
            "$a2/$split.jsonl" "$a3/$split.jsonl" \
            "$COMPARISON_ROOT/a3_seed${seed}_vs_a2_seed${seed}_${split}.json" "$seed"
        run_comparison \
            "$c3/$split.jsonl" "$a3/$split.jsonl" \
            "$COMPARISON_ROOT/a3_seed${seed}_vs_c3_${split}.json" "$seed"
    done
}

prepare_seed() {
    local seed="$1"
    local data_dir="$DATA_ROOT/rq2_waveform_mild_seed${seed}"
    if [[ ! -f "$data_dir/summary.json" ]]; then
        if [[ -e "$data_dir" || -e "${data_dir%/*}/.${data_dir##*/}.staging" ]]; then
            echo "Incomplete materialization path exists for seed $seed: $data_dir" >&2
            echo "Inspect it manually; automatic deletion or overwrite is forbidden." >&2
            exit 1
        fi
        echo "Materializing RQ2 waveform data for seed $seed"
        "$PYTHON" "$ROOT_DIR/prepare_waveform_augmentation.py" \
            --train-manifest "$CLEAN_TRAIN" \
            --policy "$POLICY" \
            --asset-root "$ASSET_ROOT" \
            --output-dir "$data_dir" \
            --seed "$seed" \
            --workers "$WORKERS"
    else
        echo "Materialization already complete for seed $seed"
    fi

    "$PYTHON" "$ROOT_DIR/prepare_rq2_replication_configs.py" --seed "$seed"

    local a2_dir="experiments/rq2/a2_waveform_random_seed${seed}"
    local a3_dir="experiments/rq2/a3_waveform_c3_seed${seed}"
    local a2_output="$OUTPUT_ROOT/rq2/a2_waveform_random_seed${seed}"
    local a3_output="$OUTPUT_ROOT/rq2/a3_waveform_c3_seed${seed}"
    if [[ ! -f "$a2_output/train_results.json" ]]; then
        HF_HOME="$HF_HOME_VALUE" "$PYTHON" "$ROOT_DIR/train_full.py" \
            --config "$ROOT_DIR/$a2_dir/config.json" \
            --output-dir "$a2_output" \
            --dry-run
    fi
    if [[ ! -f "$a3_output/train_results.json" ]]; then
        HF_HOME="$HF_HOME_VALUE" "$PYTHON" "$ROOT_DIR/train_snr_curriculum.py" \
            --config "$ROOT_DIR/$a3_dir/config.json" \
            --output-dir "$a3_output" \
            --dry-run
    fi

    "$ROOT_DIR/scripts/run_sequence.sh" "$a2_dir" "$a3_dir"
    compare_seed "$seed"
}

echo "===== Closing RQ2 seed 42 comparisons ====="
compare_seed 42

for seed in 43 44; do
    echo "===== RQ2 primary seed $seed ====="
    prepare_seed "$seed"
done

echo "===== RQ2 descriptive factorial aggregation ====="
"$PYTHON" "$ROOT_DIR/aggregate_rq2_factorial.py" \
    --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/rq2/aggregate"

cat <<'EOF'
RQ2 A2/A3 primary runs, item exports, comparisons, and descriptive aggregation are complete.
STOP: scientific interpretation and promotion decisions are required next.
Do not automatically open FLEURS or launch SpecAug, Medium, RQ3, or RQ4 stages.
EOF