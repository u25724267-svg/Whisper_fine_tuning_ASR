# RQ1-C4: Acoustic Cumulative Pacing, Seed 42

## Purpose

Test progressive easy-to-hard acoustic eligibility while holding total sample
presentations and optimizer updates equal to C0-v2 and C4R.

## Treatment

- Priority: C3 joint SNR-duration difficulty
- Epoch 1: easiest 4,603 unique rows, cycled to 13,807 presentations
- Epoch 2: easiest 9,205 unique rows, cycled to 13,807 presentations
- Epoch 3: all 13,807 rows once
- Within-pool order: seeded random
- Repetition imbalance within a stage: at most one presentation

## Controls

C4R uses identical stage sizes, presentation counts, repetition, shuffling,
model, data, and optimization but assigns priority randomly. C0-v2 uses all rows
from the start. C4 cannot be interpreted without C4R.

## Required artifacts

- Priority ranking with first eligible epoch
- Three item-level presentation-order files
- Three epoch checkpoints
- Validation/test metrics and hashed item predictions

## Status

The initial launch failed before optimizer step 1 because concurrent sampler
iterators used the same temporary audit filename. A first correction introduced
collision-safe temporary files, but the retry also failed before step 1 because
the audit parent directory did not exist. The sampler now creates that directory
at write time and retains the collision-safe atomic writes. Neither failed launch
created a checkpoint.

Restarted successfully from step 0 on 2026-09-02 in tmux session
`rq1-c4-acoustic-cumulative-seed42`, using the same output directory and W&B run
ID `4z5ujtti`. The priority and epoch-1 audit files each contain 13,807 rows, and
training progressed beyond step 200. Both failures remain in the centralized log.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c4_acoustic_cumulative_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c4_acoustic_cumulative_seed42/logs/train.log`

Results must not be added until metric, priority, presentation-order, and
item-prediction artifacts are present.

## Results

- Validation WER/CER: 34.4698 / 7.8065
- Test WER/CER: 32.5855 / 7.4686
- Test WER change versus C0-v2: 1.2295 points worse
- Item-level validation and test predictions: complete and hashed

C4 does not improve on C0-v2. Interpretation against pacing requires the
mandatory C4R result.