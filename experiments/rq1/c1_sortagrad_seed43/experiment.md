# RQ1-C1: Duration SortaGrad, Seed 43

## Purpose

Measure post hoc seed stability for C1 after the original seed-42 screen. This
run was not selected by the original promotion rule and is reported as a
post-outcome replication extension.

## Treatment and control

- Epoch 1: ascending duration with utterance-ID and dataset-index tie-breakers.
- Epochs 2--3: complete seed-43 random permutations.
- Control: `rq1-c0-random-seed43-v2`.
- All non-ordering settings and speaker-disjoint-v2 manifest hashes match C1
  seed 42 and the seed-43 control.

## Required artifacts

- Three complete curriculum-order JSONL files.
- Epoch checkpoints at steps 2,302, 4,604, and 6,906.
- Train, validation, and test result JSON files.
- Hashed item-level validation/test predictions with WER and CER.

## Status

Configured on 2026-09-25. Training has not yet completed.