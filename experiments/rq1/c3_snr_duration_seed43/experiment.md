# RQ1-C3: Joint SNR-Duration, Seed 43

## Purpose

Replicate the strongest seed-42 static curriculum using the shared seed-43
control and fixed speaker-disjoint WAXAL v2 protocol.

## Contract

- Control: C0 seed 43
- Training and data seed: 43
- Fixed C3 acoustic metadata and equal duration/inverse-SNR weights
- Strict easy-to-hard order in every epoch
- No replacement, pacing, augmentation, or difficulty mixing
- Identical model, optimizer, update budget, and evaluation to C3 seed 42
- Hashed item-level validation/test predictions required

## Status

Validated and launched on 2026-09-03 after C0 seed-43 item evaluation completed.
The 13,807-row dry run passed with fixed acoustic order hash `86ff768edcc8`, and
all behavior-controlling fields match C3 seed 42 except the shared seed and
control identity. The run progressed beyond optimizer step 35 without runtime,
curriculum, or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `frv5gwmi`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c3_snr_duration_seed43`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c3_snr_duration_seed43/logs/train.log`

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully.

- Validation WER/CER: 32.8368 / 7.1317
- Test WER/CER: 30.4564 / 6.4664

The final model, all three checkpoint models, all presentation orders, metrics,
and item predictions are byte-identical to C3 seed 42. Strict C3 ordering and
zero model dropout make the configured seed behaviorally inactive. A seed-44
C3 retrain would therefore be redundant unless the treatment definition is
changed.