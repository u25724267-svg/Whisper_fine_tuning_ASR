# RQ1-C1: Duration SortaGrad, Seed 42

## Purpose

Test whether a duration-ordered first epoch improves Whisper Base fine-tuning
relative to the C0-v2 random control on the same speaker-disjoint Shona data.

## Treatment

- Epoch 1: all training utterances ordered by ascending duration
- Duration ties: utterance ID, then original dataset index
- Epochs 2 and 3: complete seeded random permutations
- Per-epoch order: saved as item-level JSONL artifacts

## Matched control

The control is `rq1-c0-random-seed42-v2`. Model revision, data manifests,
preprocessing, augmentation, optimizer, seed, batch size, epoch count, learning
rate schedule, checkpoint boundaries, decoding, validation, and test evaluation
are unchanged. Only training-example order differs.

## Data

- Protocol: `waxal_shona_speaker_disjoint_v2`
- Train: 13,807 utterances, 79.76 hours, 113 speakers
- Validation: 1,715 utterances, 9.59 hours, 24 speakers
- Test: 1,671 utterances, 9.56 hours, 24 speakers
- Speaker overlap: zero

## Required artifacts

- Epoch checkpoints at steps 2,302, 4,604, and 6,906
- Three complete curriculum-order JSONL files
- Validation and test metrics
- Hashed item-level validation and test predictions
- Runner and sampler source hashes in the run manifest

## Status

Training completed on 2026-09-01. A `SIGINT` interrupted the original final test
pass after all three epoch checkpoints and final validation were saved. Final
item-level validation and test predictions were recovered from the selected
epoch-3 checkpoint without retraining.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c1_sortagrad_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c1_sortagrad_seed42/logs/train.log`
- Canonical checkpoint: epoch 3, step 6,906
- Trainer validation WER: 32.4730%
- Recovered item-level validation WER/CER: 32.5872% / 7.2072%
- Recovered item-level test WER/CER: 31.0153% / 6.7940%

The frozen v2 protocol contains 520 training utterances (3.77%) longer than
Whisper's 30-second feature window. They are retained because C0-v2 used the
same manifests; changing them only for C1 would invalidate the one-factor
comparison. This shared limitation is reported in analysis.

All three curriculum-order files and hashed item-prediction artifacts are
present. The checkpoint directory, rather than the incomplete root model export,
is the canonical model artifact.

## Comparison with C0-v2

| Metric | C0-v2 random | C1 SortaGrad | C1 change |
|---|---:|---:|---:|
| Item-level validation WER | 32.5182% | 32.5872% | +0.0690 |
| Item-level test WER | 31.3560% | 31.0153% | -0.3407 |
| Item-level test CER | 7.0103% | 6.7940% | -0.2163 |

C1 improves seed-42 test WER by 0.3407 points but remains below the
preregistered 0.5-point practical threshold and does not improve recovered
validation WER. SortaGrad is therefore retained as a completed screen condition
but is not promoted or extended before the remaining curriculum screen is run.