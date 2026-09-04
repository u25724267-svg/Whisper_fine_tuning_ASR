# RQ1-C5: Dynamic S2S-Margin, Seed 44

## Purpose

Complete the independent-seed replication of C5 using the shared seed-44
control on the fixed speaker-disjoint WAXAL v2 protocol.

## Contract

- Control: C0 seed 44
- Training and data seed: 44
- Epoch 1: seed-44 random permutation
- Epochs 2-3: ascending prior-epoch online NLL-per-second score
- Every row appears exactly once per epoch
- No replacement, pacing, augmentation, or difficulty mixing
- Identical model, optimizer, update budget, and evaluation to C5 seeds 42/43
- Hashed item-level validation/test predictions required

## Status

Validated and launched on 2026-09-03 after C0 seed-44 item evaluation completed.
The 13,807-row dry run passed with epoch-1 random order hash `26be5736e7b0`, and
all behavior-controlling fields match C5 seed 43 except the shared seed and
control identity. The run progressed beyond optimizer step 35 without
dynamic-score, alignment, or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `mgw7qkdk`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed44`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed44/logs/train.log`

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully.

- Validation WER/CER: 34.1963 / 7.9271
- Test WER/CER: 31.4118 / 6.9905
- Test WER change versus C0 seed 44: 0.4232 points worse