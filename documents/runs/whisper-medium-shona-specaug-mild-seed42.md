# Whisper Medium Shona Mild SpecAugment Run

## Run identity

| Field | Value |
|---|---|
| Experiment | `whisper-medium-shona-specaug-mild-seed42` |
| Status | Completed |
| Start time (UTC) | 2026-08-20 11:37:04 |
| tmux session | `whisper-medium-shona-specaug-mild-seed42` |
| Trainer PID at launch | `2064901` |
| Configuration | `configs/whisper-medium-shona-specaug-mild-seed42.json` |
| Intended output | `output_dir_whisper_medium_shona_specaug_mild_seed42` |
| Intended log | `logs/whisper-medium-shona-specaug-mild-seed42.log` |
| W&B project | `whisper-shona-multilingual` |
| W&B run name | `whisper-medium-shona-specaug-mild-seed42` |

## Research question

Does always-on mild SpecAugment improve Whisper Medium Shona ASR relative to the
existing unaugmented Whisper Medium run when model revision, data, seed,
optimization, effective batch, decoding, and evaluation are held fixed?

## Baseline

The comparable unaugmented Medium experiment used the same pinned model revision,
manifests, seed 42, three epochs, per-device batch 2, gradient accumulation 3,
learning rate 1e-5, 500 warm-up steps, FP16, and 1,000-step evaluation.

| Baseline metric | Value |
|---|---:|
| Validation loss | 0.24881 |
| Validation WER | 22.9450% |
| Test loss | 0.24767 |
| Test WER | 23.7739% |

The augmentation run is directly comparable for model behavior. Checkpoint
retention differs only because disk pressure requires model-only checkpoints;
this does not alter forward/backward optimization.

## Model

| Field | Value |
|---|---|
| Model | `openai/whisper-medium` |
| Revision | `abdf7c39ab9d0397620ccaea8974cc764cd0953e` |
| Parameters | 763,857,920 |
| Language/task | Shona transcription |

## Data

| Split | Rows | Approximate duration |
|---|---:|---:|
| Train | 13,799 | 79.28 h |
| Validation | 1,683 | 9.70 h |
| Test | 1,711 | 9.93 h |

Manifest hashes are captured automatically in `run_manifest.json` when the run
starts. Inputs are decoded and resampled to 16 kHz, then converted to 80-bin,
3,000-frame Whisper log-Mel features. Features remain in RAM.

The WAXAL v1 speaker-overlap limitation remains: 1,700 of 1,711 cleaned test
utterances belong to speakers represented in training. Results are in-domain and
do not prove speaker generalization.

## Augmentation

Every training presentation receives online mild SpecAugment:

```json
{
  "application_probability": 1.0,
  "mask_time_prob": 0.05,
  "mask_time_length": 8,
  "mask_time_min_masks": 1,
  "mask_feature_prob": 0.05,
  "mask_feature_length": 8,
  "mask_feature_min_masks": 1
}
```

Masks are generated online and set selected feature cells to zero. Validation,
test, and inference are clean because masking is active only in training mode.
No waveform augmentation, time warping, curriculum learning, or text
augmentation is used.

## Training

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Seed | 42 |
| Train/eval batch | 2 / 2 |
| Gradient accumulation | 3 |
| Effective train batch | 6 |
| Learning rate | 1e-5 |
| Warm-up | 500 steps |
| Optimizer | AdamW Torch |
| Gradient checkpointing | Enabled |
| Precision | FP16 |
| Logging/evaluation/checkpoint interval | 1,000 optimizer steps |
| Best model | Minimum validation WER |
| Checkpoint retention | One model-only checkpoint |

Model-only checkpoints are necessary because only about 13 GB disk was free
when this experiment was configured. Optimizer state is not saved, so this run
cannot resume after interruption; the launcher rejects resume attempts and a
fresh deterministic rerun is required.

## Reproducibility

The trainer writes the following before feature extraction:

- resolved `experiment_config.json`;
- `run_manifest.json` with model revision;
- SHA-256 hashes for trainer, config, dependencies, and all data manifests;
- exact package, Python, PyTorch, CUDA, and GPU versions;
- effective seed, epochs, and output path.

The W&B project allowlist permits only `whisper-shona-multilingual`.

### Captured source and data hashes

| Artifact | SHA-256 |
|---|---|
| Trainer | `4866d6c8919984dfb1c87b4ec8572000c2cc735c524b45c76d48b032bc7bc85b` |
| Experiment config | `1fe01388acf83651d7aa5496f88c7aa34be2e4e0a926eedc340515fd2b4d6eea` |
| Dependency pins | `1eefe5d5d2fd57e5bb4b2cd0d709cf0934779fc3410ed988975eba3ab4c7e873` |
| Train manifest | `f0e300fe88de9b77d550940b2a1d537313dc73d1cdbfa299eaa426810cfd8469` |
| Validation manifest | `597f503ed58aae149be8258c02cce4553d6bde33f7811be79d434446e19b6941` |
| Test manifest | `a2fe73a95184e89842ef598111644e16eae056086e5afcdbed9fd2978025f203` |

### Launch command

```bash
WHISPER_CONFIG=configs/whisper-medium-shona-specaug-mild-seed42.json \
./run_full_detached.sh
```

### Fresh reproduction

```bash
WHISPER_CONFIG=configs/whisper-medium-shona-specaug-mild-seed42.json \
WHISPER_OUTPUT_DIR=output_dir_whisper_medium_shona_specaug_mild_seed42_rerun \
TMUX_SESSION_NAME=whisper-medium-shona-specaug-mild-seed42-rerun \
./run_full_detached.sh
```

## Evaluation

The full validation split is evaluated every 1,000 optimizer steps. After three
epochs, the best checkpoint is evaluated again on validation and then on the full
test split. Training rows are never used for WER evaluation.

## Results

| Metric | Value |
|---|---:|
| Best validation WER | 22.2636% at step 6,000 |
| Final validation loss | 0.23761 |
| Final validation WER | 22.2636% |
| Final test loss | 0.23611 |
| Final test WER | 22.7487% |
| Train loss | 0.27140 |
| Train runtime | 9,185.79 s (2:33:05.79) |
| W&B URL | `https://wandb.ai/dsfsi/whisper-shona-multilingual/runs/t4n9425x` |

### Validation trajectory

| Step | Epoch | Learning rate | Train loss | Validation WER |
|---:|---:|---:|---:|---:|
| 1,000 | 0.435 | 9.2266e-6 | 0.7266 | 29.2561% |
| 2,000 | 0.870 | 7.6641e-6 | 0.3070 | 24.9126% |
| 3,000 | 1.304 | 6.1016e-6 | 0.2252 | 23.4707% |
| 4,000 | 1.739 | 4.5391e-6 | 0.2037 | 23.8127% |
| 5,000 | 2.174 | 2.9766e-6 | 0.1688 | 22.9220% |
| 6,000 | 2.609 | 1.4141e-6 | 0.1269 | 22.2636% |

The temporary WER increase at step 4,000 did not persist. Step 6,000 was selected
as the best checkpoint and used for final validation and test evaluation.

### Baseline comparison

| Policy | Validation WER | Test WER |
|---|---:|---:|
| Unaugmented Whisper Medium | 22.9450% | 23.7739% |
| Mild SpecAugment Whisper Medium | 22.2636% | 22.7487% |
| Absolute change | -0.6814 points | -1.0252 points |
| Relative WER reduction | 2.97% | 4.31% |

The improvement exceeds the predeclared 0.5-point caution threshold on both
validation and test. It is evidence that this policy is promising for this seed
and v1 split, but not yet a speaker-generalization claim.

## Planned stage-2 extension

The retained checkpoint is model-only; optimizer and scheduler states were not
saved. A mathematically exact continuation is impossible. The extension is
therefore a separately documented stage initialized from the selected model,
with a fresh AdamW optimizer, learning rate 2e-6, 100 warm-up steps, and exactly
2,000 optimizer steps. See
`configs/whisper-medium-shona-specaug-mild-stage2-2000-seed42.json`.

## Decision rule

Compare only with the unaugmented Medium baseline. Treat absolute WER changes
below 0.5 points as inconclusive without additional seeds. Do not compare this
run causally against Base or Large because model capacity changes.

## Limitations

1. One seed cannot estimate run variance.
2. WAXAL v1 has severe speaker overlap.
3. Clean WER does not measure noise robustness.
4. Time and frequency masks are not independently ablated.
5. Model-only checkpoints prevent interruption recovery.