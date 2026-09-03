# RQ1-C5: Dynamic S2S-Margin, Seed 43

## Purpose

Independently replicate C5 using the shared seed-43 control on the fixed
speaker-disjoint WAXAL v2 protocol.

## Contract

- Control: C0 seed 43
- Training and data seed: 43
- Epoch 1: seed-43 random permutation
- Epochs 2-3: ascending prior-epoch online NLL-per-second score
- Every row appears exactly once per epoch
- No replacement, pacing, augmentation, or difficulty mixing
- Identical model, optimizer, update budget, and evaluation to C5 seed 42
- Hashed item-level validation/test predictions required

## Status

Validated and launched on 2026-09-03 after C3 seed-43 item evaluation completed.
The 13,807-row dry run passed with epoch-1 random order hash `8eb867d6b358`.
All behavior-controlling fields match C5 seed 42 except the shared seed and
control identity; the 20 GB admission threshold enforces sequential execution.
The run progressed beyond optimizer step 35 without dynamic-score, alignment,
or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `1710akai`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed43`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed43/logs/train.log`