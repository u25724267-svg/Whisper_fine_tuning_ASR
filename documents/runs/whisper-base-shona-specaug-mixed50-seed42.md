# Whisper Base Shona 50/50 Clean–SpecAugment Run

## Run identity

| Field | Value |
|---|---|
| Experiment | `whisper-base-shona-specaug-mixed50-seed42` |
| Status at documentation time | Running; GPU training active |
| Start time (UTC) | 2026-08-20 10:01:41 |
| tmux session | `whisper-base-shona-specaug-mixed50-seed42` |
| Trainer PID at launch | `2016899` |
| Configuration | `configs/whisper-base-shona-specaug-mixed50-seed42.json` |
| Output directory | `output_dir_whisper_base_shona_specaug_mixed50_seed42` |
| Persistent log | `logs/whisper-base-shona-specaug-mixed50-seed42.log` |
| W&B project | `whisper-shona-multilingual` |
| W&B run name | `whisper-base-shona-specaug-mixed50-seed42` |
| W&B run | `https://wandb.ai/dsfsi/whisper-shona-multilingual/runs/bzaaig2m` |

The W&B project is protected by `configs/wandb-projects.json`. The trainer fails
before authentication if a configuration names an unapproved project.

## Research question

Does presenting a deterministic stochastic mixture of clean and mildly masked
log-Mel features improve Whisper Base Shona ASR relative to:

1. an otherwise identical deterministic clean control; and
2. mild SpecAugment applied to every training presentation?

This experiment addresses the concern that always-on masking never presents a
completely clean feature matrix during training. It keeps the physical dataset
unchanged and selects clean or augmented features independently for each sample
presentation.

## Hypothesis

A 50/50 mixture should preserve clean-speech behavior while retaining some
regularization and robustness benefit from SpecAugment. It may outperform the
always-augmented policy on clean WER if always-on masking is too strong, but it
may provide less robustness benefit under acoustic corruption. This run measures
clean in-domain WER only; robustness claims require separate noisy evaluation.

## Related experiment matrix

| Experiment | Augmentation application |
|---|---|
| `whisper-base-shona-specaug-control-seed42` | 100% clean |
| `whisper-base-shona-specaug-mild-seed42` | 100% mild SpecAugment |
| `whisper-base-shona-specaug-mixed50-seed42` | 50% clean / 50% mild SpecAugment |
| `whisper-base-shona-specaug-paper9-seed42` | 100% strong paper-9 SpecAugment |

All four configurations pin the same Whisper Base revision, manifests, seed,
training schedule, batches, decoding behavior, and metric. Only the augmentation
block changes.

## Literature rationale

The original SpecAugment paper found that time and frequency masking supplied
most of the gain, while time warping supplied the smallest controlled benefit.
This run therefore uses masking only.

The closest attached Whisper study, using approximately 60 hours per language,
reports a stronger policy with time probability 0.05, time length 8, frequency
probability 0.15, frequency length 15, and at least three frequency masks. On an
80-bin Whisper input, that policy can remove a large part of the frequency axis.
We use the milder one-by-eight-bin policy and vary only how often it is applied.

Other attached Whisper studies do not isolate SpecAugment from simultaneous
changes in data, optimization, regularization, or alignment. Their headline gains
are not treated as causal evidence for a specific mask policy. The full synthesis
is in `documents/specaugment_experiment_plan.md`.

## Data

| Split | Cleaned rows | Approximate duration | Manifest |
|---|---:|---:|---|
| Train | 13,799 | 79.28 h | `sna_asr_train.normalized.json` |
| Validation | 1,683 | 9.70 h | `sna_asr_validation.normalized.json` |
| Test | 1,711 | 9.93 h | `sna_asr_test.normalized.json` |

Audio is decoded from local files, resampled to 16 kHz, and converted into
Whisper's 80-bin log-Mel representation. Inputs are padded or truncated to 3,000
frames, corresponding to 30 seconds. The text targets are cleaned lowercase Shona
transcripts from the normalized WAXAL manifests.

### Manifest integrity

| Artifact | SHA-256 |
|---|---|
| Train manifest | `f0e300fe88de9b77d550940b2a1d537313dc73d1cdbfa299eaa426810cfd8469` |
| Validation manifest | `597f503ed58aae149be8258c02cce4553d6bde33f7811be79d434446e19b6941` |
| Test manifest | `a2fe73a95184e89842ef598111644e16eae056086e5afcdbed9fd2978025f203` |

### Split limitation

The run uses WAXAL v1 splits. There is no duplicate audio across train and test,
but 101 of 110 test speakers occur in training, and 1,700 of 1,711 cleaned test
utterances belong to seen speakers. Results therefore measure in-domain
adaptation rather than robust speaker generalization. A later study must retrain
and evaluate with the speaker-disjoint WAXAL v2 protocol.

## Model

| Field | Value |
|---|---|
| Model | `openai/whisper-base` |
| Immutable revision | `e37978b90ca9030d5170a5c07aadb050351a65bb` |
| Parameters | 72,593,920 |
| Language | Shona |
| Task | Transcription |
| Forced decoder IDs | Cleared |
| Suppressed token list | Cleared |

Evaluation generation is explicitly configured for Shona transcription.

## Augmentation policy

```json
{
  "type": "specaugment",
  "enabled": true,
  "application_probability": 0.5,
  "mask_time_prob": 0.05,
  "mask_time_length": 8,
  "mask_time_min_masks": 1,
  "mask_feature_prob": 0.05,
  "mask_feature_length": 8,
  "mask_feature_min_masks": 1
}
```

### Per-example selection

For each example in each training batch, the trainer draws one Bernoulli random
value using PyTorch's seeded RNG:

- probability 0.5: return the original clean log-Mel features;
- probability 0.5: return a newly generated mild SpecAugment view.

The built-in Whisper masking routine first generates a masked batch. A
deterministic per-row selector then chooses either the clean clone or masked row.
The selection and masks are regenerated on every presentation. Seed 42 and full
deterministic mode fix the random sequence for this environment.

A 100-example implementation test with seed 42 produced 53 clean examples and
47 augmented examples. Evaluation changed zero examples.

### Mask geometry

For an augmented 3,000-frame input, the time policy selects approximately 18 or
19 spans of eight frames, subject to overlap. The frequency policy forces at
least one eight-bin span over the 80-bin frequency axis. Masked cells are set to
zero. In a direct always-augmented probe, the combined masks zeroed approximately
14.45% of feature cells; the exact fraction varies with overlap and valid audio
length.

### Training versus evaluation

SpecAugment is active only while `model.training` is true. Validation, test, and
inference receive unmodified features. This was verified directly: 0 evaluation
examples changed in the 100-example implementation test.

### Effective dataset exposure

The physical training set remains 13,799 utterances and 79.28 hours. Across three
epochs there are 41,397 sample presentations. In expectation, each utterance is
presented about 1.5 times clean and 1.5 times augmented. No augmented audio or
features are persisted to disk.

## Training configuration

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Seed / data seed | 42 |
| Full deterministic mode | Enabled |
| Train batch per device | 6 |
| Evaluation batch per device | 6 |
| Gradient accumulation | 1 |
| Effective train batch | 6 |
| Learning rate | 1e-5 |
| Warm-up steps | 500 |
| Scheduler | Linear |
| Optimizer | AdamW Torch |
| Weight decay | 0.0 |
| Max gradient norm | 1.0 |
| Gradient checkpointing | Enabled, non-reentrant |
| Precision | FP16 |
| Generation max length | 225 tokens |
| W&B/training logging | Every 1,000 optimizer steps |
| Validation | Every 1,000 optimizer steps |
| Checkpointing | Every 1,000 optimizer steps |
| Checkpoint retention | 1, subject to best-model retention |
| Best model | Lowest validation WER |

Feature preprocessing is kept in RAM. No curriculum learning, waveform noise,
time warping, speed perturbation, pitch shift, reverberation, clipping, or text
augmentation is used.

## Evaluation protocol

Every 1,000 optimizer steps, the trainer generates predictions for all 1,683
validation utterances and computes corpus WER. After training, it reloads the
best validation-WER checkpoint, reevaluates validation, and evaluates all 1,711
test utterances. Training rows are never used for WER evaluation.

References are normalized manifest transcripts. Predictions are decoded with
special tokens removed, without an additional post-decoding normalizer. This
matches the clean control and always-augmented runs.

## Reproducibility

### Source hashes captured at launch

| Artifact | SHA-256 |
|---|---|
| Trainer | `4866d6c8919984dfb1c87b4ec8572000c2cc735c524b45c76d48b032bc7bc85b` |
| Experiment config | `0efa43e78a5a7c3882bcfbeacd233c602b660d8200b242a6bc4352421b06864e` |
| Dependency pins | `1eefe5d5d2fd57e5bb4b2cd0d709cf0934779fc3410ed988975eba3ab4c7e873` |

These hashes were independently rechecked against the current files and matched
at documentation time.

### Runtime environment

| Component | Version/value |
|---|---|
| Python | 3.10.12, GCC 11.4.0 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime | 12.4 |
| GPU | NVIDIA GeForce RTX 4090 |
| Transformers | 4.46.3 |
| Datasets | 2.20.0 |
| Accelerate | 1.1.1 |
| Evaluate | 0.4.2 |
| JiWER | 3.0.4 |
| NumPy | 1.26.4 |
| W&B | 0.28.2 |

Machine-readable provenance is stored in
`output_dir_whisper_base_shona_specaug_mixed50_seed42/run_manifest.json`.

### Repository state caveat

At documentation time, augmentation source/config/report files were present as
uncommitted working-tree changes. The run manifest hashes preserve the exact
executed content, but these files should be committed before moving or cleaning
the workspace.

### Reproduction command

```bash
WHISPER_CONFIG=configs/whisper-base-shona-specaug-mixed50-seed42.json \
TMUX_SESSION_NAME=whisper-base-shona-specaug-mixed50-seed42-rerun \
WHISPER_OUTPUT_DIR=output_dir_whisper_base_shona_specaug_mixed50_seed42_rerun \
./run_full_detached.sh
```

Use a new output directory for a fresh reproduction. The W&B allowlist prevents
the run from creating a new project.

## Execution timeline

1. The 50/50 implementation was validated on 100 synthetic feature rows.
2. Configuration, project allowlist, manifests, disk, and GPU were validated.
3. The run launched detached at 2026-08-20 10:01 UTC.
4. Feature extraction was performed in RAM.
5. W&B run `bzaaig2m` initialized in `whisper-shona-multilingual`.
6. At documentation time, GPU training was active and no final metrics had been
   written.

## Monitoring

```bash
tail -f logs/whisper-base-shona-specaug-mixed50-seed42.log
```

```bash
tmux attach -t whisper-base-shona-specaug-mixed50-seed42
```

Detach with `Ctrl+B`, then `D`.

W&B: `https://wandb.ai/dsfsi/whisper-shona-multilingual/runs/bzaaig2m`

## Results

**Pending.** Do not report a result until metric files are present in the output
directory.

| Metric | Value |
|---|---:|
| Best validation WER | Pending |
| Final validation WER | Pending |
| Final test WER | Pending |
| Train loss | Pending |
| Train runtime | Pending |

## Existing comparison values

| Whisper Base policy | Validation WER | Test WER |
|---|---:|---:|
| Earlier unaugmented Base | 29.0647% | 28.4893% |
| Always-on mild SpecAugment | 29.0443% | 28.5689% |
| 50/50 mixed policy | Pending | Pending |

The always-on run changed validation by -0.0204 points and test by +0.0796 points
relative to the same-size Base model. These differences are below the predeclared
0.5-point caution threshold and are inconclusive. Medium and Large scores are not
valid augmentation controls because model capacity changes simultaneously.

## Decision rule

The mixed policy is promising only if it improves validation and test WER
consistently relative to the deterministic clean control. Differences below 0.5
absolute WER points require additional seeds or bootstrap confidence intervals.
If the mixed policy helps, repeat it with at least two additional seeds before
testing the strong paper-9 policy. If it does not help, prioritize waveform noise
or speaker-disjoint evaluation rather than increasing mask severity immediately.

## Known limitations

1. WAXAL v1 has severe train/test speaker overlap.
2. One seed does not quantify training variance.
3. Clean WER does not establish noise robustness.
4. Time and frequency masking effects are not separately ablated.
5. The 50/50 draw is per presentation, not a guarantee that each utterance is
   seen exactly once clean and once augmented.
6. CUDA deterministic mode improves repeatability but may not guarantee bitwise
   identity across different GPUs, drivers, or kernels.