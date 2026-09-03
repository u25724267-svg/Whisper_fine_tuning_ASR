# RQ1-C4R: Random-Priority Pacing Control, Seed 42

## Purpose

Control for reduced early eligible-pool size and repeated exposure in C4 without
using acoustic difficulty to select examples.

## Treatment

- Priority: one fixed seed-42 random ranking
- Epoch 1: first 4,603 unique rows, cycled to 13,807 presentations
- Epoch 2: first 9,205 unique rows, cycled to 13,807 presentations
- Epoch 3: all 13,807 rows once
- Within-pool order: seeded random
- Repetition imbalance within a stage: at most one presentation

## Matched comparison

C4 and C4R differ only in acoustic versus random priority. Stage sizes,
presentation counts, repetition policy, shuffling, manifests, model, optimizer,
and evaluation are identical.

## Required artifacts

- Fixed random priority ranking
- Three item-level presentation-order files
- Three epoch checkpoints
- Validation/test metrics and hashed item predictions

## Status

Paired contract and full dry run validated on 2026-09-02. Launched in tmux
session `rq1-c4r-random-pacing-seed42` while C4 completed its final evaluation,
then ran concurrently with C4 item-level prediction export. C4R uses W&B run
`tbov402z` in the existing `whisper-shona-multilingual` project. With both CUDA
workloads active, measured allocation was 6,757 MiB with 17,327 MiB free.

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully on 2026-09-02.

- Validation WER/CER: 34.3928 / 7.8264
- Test WER/CER: 32.4604 / 7.4451
- Test WER change versus C0-v2 item-level metric: 1.1044 points worse