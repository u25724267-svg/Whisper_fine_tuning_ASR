# Whisper Base Shona Mild SpecAugment Run

## Run identity

| Field | Value |
|---|---|
| Experiment | `whisper-base-shona-specaug-mild-seed42` |
| Status | Completed |
| Start time (UTC) | 2026-08-20 08:29:23 |
| tmux session | `whisper-base-shona-specaug-mild-seed42` |
| Trainer PID at launch | `1973113` |
| Configuration | `configs/whisper-base-shona-specaug-mild-seed42.json` |
| Output directory | `output_dir_whisper_base_shona_specaug_mild_seed42` |
| Persistent log | `logs/whisper-base-shona-specaug-mild-seed42.log` |
| W&B project | `whisper-shona-multilingual` |
| W&B run name | `whisper-base-shona-specaug-mild-seed42` |
| Canonical W&B run | `https://wandb.ai/dsfsi/whisper-shona-multilingual/runs/kjw0am46` |

The first launch accidentally used a new W&B project and was stopped after
checkpoint 1,000. Training was then resumed from that checkpoint under the
existing `whisper-shona-multilingual` project. The first launch's run ID was
`dft2yvpt`; it must not be used as the canonical experiment record.
All later runs are protected by `configs/wandb-projects.json`; unapproved project
names fail before W&B authentication.

## Research question

Does moderate, online SpecAugment improve Whisper Base transcription of cleaned
WAXAL Shona speech relative to an otherwise identical deterministic no-augmentation
control, without materially degrading clean validation or test WER?

The intended causal comparison changes only the `augmentation` block. The clean
control is `configs/whisper-base-shona-specaug-control-seed42.json`. A stronger
replication policy is stored in
`configs/whisper-base-shona-specaug-paper9-seed42.json` and should be run only
after the mild policy.

## Literature rationale

### SpecAugment

Park et al., *SpecAugment: A Simple Data Augmentation Method for Automatic Speech
Recognition* (2019), apply online time warping, time masking, and frequency masking
to normalized log-Mel features. Their controlled ablation found that time and
frequency masking supplied most of the benefit, while time warping supplied the
smallest gain and incurred additional implementation cost. This run therefore
omits time warping.

### Closest attached Whisper study

*ASR Under Noise: Exploring Robustness for Sundanese and Javanese* reports a
best Whisper policy using time probability 0.05, time-mask length 8, minimum one
time mask, frequency probability 0.15, frequency-mask length 15, and minimum
three frequency masks. The corpus size, approximately 60 hours per language, is
closer to the 79-hour Shona training corpus than the other attached studies.

That frequency policy is aggressive for Whisper's 80 Mel bins: Transformers
forces three spans of 15 bins, potentially masking a large fraction of the
frequency axis. This first run instead uses one eight-bin frequency span. The
paper policy is retained as a separate ablation rather than copied as the default.

### Other attached studies

- The Swiss German real-world Whisper study states that it uses SpecAugment but
  does not report exact masking parameters or isolate its effect.
- The subtitle-aligned Swiss German study enables SpecAugment together with
  optimizer, regularization, and batch-size changes, so its gain is confounded.
- Bangla-WhisperDiar uses waveform noise, echo, reverberation, clipping,
  telephone filtering, pitch shift, and time stretch, but does not report a
  feature-level SpecAugment experiment or augmentation-only ablation.

The complete literature synthesis and planned experiment matrix are in
`documents/specaugment_experiment_plan.md`.

## Data

| Split | Cleaned rows | Approximate duration | Manifest |
|---|---:|---:|---|
| Train | 13,799 | 79.28 h | `sna_asr_train.normalized.json` |
| Validation | 1,683 | 9.70 h | `sna_asr_validation.normalized.json` |
| Test | 1,711 | 9.93 h | `sna_asr_test.normalized.json` |

The manifests contain local audio paths, durations, normalized lowercase text,
utterance IDs, speaker IDs, language, gender, and raw text. Audio is decoded and
resampled to 16 kHz. Whisper converts each utterance into 80-bin log-Mel features
padded or truncated to 3,000 frames (30 seconds).

### Data integrity hashes

| Artifact | SHA-256 |
|---|---|
| Train manifest | `f0e300fe88de9b77d550940b2a1d537313dc73d1cdbfa299eaa426810cfd8469` |
| Validation manifest | `597f503ed58aae149be8258c02cce4553d6bde33f7811be79d434446e19b6941` |
| Test manifest | `a2fe73a95184e89842ef598111644e16eae056086e5afcdbed9fd2978025f203` |

### Split limitation

This experiment uses the original WAXAL v1 splits. The split audit found no
duplicate audio across train/test, but 101 of 110 test speakers also occur in
training; 1,700 of 1,711 cleaned test utterances belong to speakers seen during
training. Test WER is therefore an in-domain result and is not sufficient evidence
of speaker generalization. Publishable generalization claims require retraining
and evaluation with WAXAL v2 speaker-disjoint splits.

## Model

| Field | Value |
|---|---|
| Hugging Face model | `openai/whisper-base` |
| Immutable revision | `e37978b90ca9030d5170a5c07aadb050351a65bb` |
| Parameters | 72,593,920 |
| Language | Shona |
| Task | Transcription |
| Forced decoder IDs | Cleared |
| Suppressed token list | Cleared |

The model and processor are loaded from the pinned revision. Evaluation
generation is explicitly configured for Shona transcription.

## Augmentation policy

```json
{
  "type": "specaugment",
  "enabled": true,
  "mask_time_prob": 0.05,
  "mask_time_length": 8,
  "mask_time_min_masks": 1,
  "mask_feature_prob": 0.05,
  "mask_feature_length": 8,
  "mask_feature_min_masks": 1
}
```

Transformers 4.46.3 applies masking inside `WhisperModel` only while
`model.training` is true. Validation, test, and generated predictions are never
augmented.

For a 3,000-frame input, the time policy selects approximately 18 or 19 spans of
eight frames, subject to random placement and overlap, for an upper bound near
5% of the time axis. For the 80-bin frequency axis, the minimum forces one span
of eight bins, or 10% of the frequency axis. Time and frequency masks may overlap.
Masked feature values are set to zero.

A direct implementation probe produced:

| Mode | Zero-valued feature fraction |
|---|---:|
| Control training | 0.00% |
| Mild SpecAugment training | 14.45% |
| Mild SpecAugment evaluation | 0.00% |

The observed combined fraction varies with random mask placement. Seed 42 and
deterministic mode fix the random sequence for this run.

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
| Scheduler | Linear (Transformers default) |
| Optimizer | AdamW Torch (Transformers default) |
| Weight decay | 0.0 (Transformers default) |
| Max gradient norm | 1.0 (Transformers default) |
| Gradient checkpointing | Enabled, non-reentrant |
| Precision | FP16 |
| Generation max length | 225 tokens |
| Logging interval | 1,000 optimizer steps |
| Validation interval | 1,000 optimizer steps |
| Checkpoint interval | 1,000 optimizer steps |
| Checkpoint retention | 1, except best-model retention rules |
| Best-model metric | Minimum validation WER |

Feature preprocessing is kept in RAM to avoid additional disk cache pressure.
No curriculum learning, waveform augmentation, noise mixing, time warping,
speed perturbation, pitch shift, reverberation, or text augmentation is used.

## Evaluation protocol

At each 1,000-step interval, the trainer generates transcriptions for the full
1,683-row validation split and computes corpus WER. After training, the selected
best checkpoint is evaluated once more on validation and then on all 1,711 test
rows. The training split is never used for WER evaluation.

References are the normalized manifest text. Predictions are decoded with
special tokens removed. The primary metric does not apply an additional text
normalizer after decoding, matching the existing Base/Medium/Large experiments.

## Reproducibility

### Source hashes

| Artifact | SHA-256 |
|---|---|
| Trainer | `7a1a701938ab6e8677502cf282f94b9d7e79c40eef64db3e69a28eae4d807595` |
| Experiment config | `9c253055d1d526e912ead42291904761a48781605533aaee5dd2e1f1019b65a0` |
| Dependency pins | `1eefe5d5d2fd57e5bb4b2cd0d709cf0934779fc3410ed988975eba3ab4c7e873` |

### Runtime

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

The machine-readable source of truth is
`output_dir_whisper_base_shona_specaug_mild_seed42/run_manifest.json`.

### Reproduction command

```bash
WHISPER_CONFIG=configs/whisper-base-shona-specaug-mild-seed42.json \
TMUX_SESSION_NAME=whisper-base-shona-specaug-mild-seed42-rerun \
WHISPER_OUTPUT_DIR=output_dir_whisper_base_shona_specaug_mild_seed42_rerun \
./run_full_detached.sh
```

Use a new output directory for a fresh reproduction. If a resumable checkpoint
exists in the configured output directory, the launcher resumes it automatically.

## Execution timeline

1. Configuration, dependencies, manifests, and GPU were validated.
2. A direct model probe verified masking in training and no masking in evaluation.
3. The run launched in detached tmux on 2026-08-20 at approximately 08:29 UTC.
4. At documentation time, feature extraction was still running in RAM. No
   optimizer step, checkpoint, WER metric, or final model had been produced.

Monitor the run with:

```bash
tail -f logs/whisper-base-shona-specaug-mild-seed42.log
```

or:

```bash
tmux attach -t whisper-base-shona-specaug-mild-seed42
```

Detach using `Ctrl+B`, then `D`.

## Results

| Metric | Value |
|---|---:|
| Final validation loss | 0.32937 |
| Final validation WER | 29.0443% |
| Final test loss | 0.32851 |
| Final test WER | 28.5689% |
| Reported train loss | 0.30173 |
| Reported resumed-segment train runtime | 2,813.32 s |
| W&B run | `https://wandb.ai/dsfsi/whisper-shona-multilingual/runs/kjw0am46` |

The comparable unaugmented Whisper Base run scored 29.0647% validation WER and
28.4893% test WER. Mild always-on SpecAugment therefore changed validation WER
by -0.0204 points and test WER by +0.0796 points. Both differences are far below
the predeclared 0.5-point caution threshold and should be treated as inconclusive,
not as evidence of a meaningful regression or improvement.

Medium and Large results must not be used as augmentation baselines because model
capacity changes simultaneously. The interrupted/resumed execution means the
reported trainer runtime covers the resumed segment rather than complete wall time.

## Comparison and decision rule

The primary comparison is the deterministic clean control with the same seed and
training settings. The older Base run is useful context but is not the formal
control because its logging/evaluation schedule and deterministic settings differ.

The mild policy is considered promising only if it improves validation and test
WER consistently. Differences below 0.5 absolute WER points should be treated as
inconclusive without additional seeds or bootstrap confidence intervals. If mild
masking helps, run the strong paper-9 policy and at least two additional seeds.
If it harms clean WER, reduce time probability or remove frequency masking before
testing stronger waveform augmentation.

## Known limitations

1. WAXAL v1 has severe train/test speaker overlap.
2. One seed cannot quantify run-to-run variance.
3. Only clean in-domain WER is measured; no real-noise or cross-corpus test is
   included.
4. The effect of time and frequency masking is not separated in this first run.
5. The current experiment does not test waveform-level augmentation or combined
   waveform plus feature augmentation.
6. FP16/CUDA deterministic mode improves repeatability but bitwise identity can
   still depend on hardware, drivers, and external library kernels.