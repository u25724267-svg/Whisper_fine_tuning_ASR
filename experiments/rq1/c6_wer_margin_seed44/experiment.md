# C6 WER-margin, Seed 44

## Purpose

Measure post hoc seed stability after the original seed-42 curriculum screen.
This run was not selected by the original promotion rule and is reported as a
post-outcome replication extension.

## Matched design

- Control: `rq1-c0-random-seed44-v2`.
- Model revision, speaker-disjoint-v2 manifests, optimizer, update budget, batch
  size, decoding, checkpoint boundaries, and augmentation match the source run.
- Only seed-specific identity, control, output, and W&B tracking metadata differ.

Epoch 1 uses the seed-specific random control order before WER-based ordering in epochs 2 and 3, so this run supplies an independent ordering perturbation.

## Required artifacts

- Three epoch checkpoints and audited curriculum orders/scores.
- Train, validation, and test result JSON files.
- Hashed item-level validation/test predictions with WER and CER.

## Status

Configured on 2026-09-25. Training has not yet completed.
