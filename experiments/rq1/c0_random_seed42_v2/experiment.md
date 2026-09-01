# RQ1-C0: Random Control, Seed 42, Protocol v2

## Purpose

Establish the confirmatory conventional-shuffle Whisper Base reference using a
balanced speaker-disjoint WAXAL Shona protocol and epoch-boundary checkpointing.

## Treatment

- Curriculum: none; seeded Transformers random sampler
- Augmentation: none
- Model: pinned multilingual Whisper Base
- Training: three epochs, effective batch size six, seed 42
- Checkpoints: epoch boundaries at steps 2,302, 4,604, and 6,906
- Selection: lowest epoch-boundary validation WER
- Prediction export: required after training

## Data

- Protocol: `waxal_shona_speaker_disjoint_v2`
- Train: 13,807 utterances, 79.76 hours, 113 speakers
- Validation: 1,715 utterances, 9.59 hours, 24 speakers
- Test: 1,671 utterances, 9.56 hours, 24 speakers
- Largest validation speaker share: 19.5% of rows
- Largest test speaker share: 17.2% of rows
- Speaker overlap: zero across all splits

## Status

Completed on 2026-09-01.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2/logs/train.log`
- Best checkpoint: epoch 3, step 6,906
- Validation WER: 32.5262%
- Test WER: 31.3400%
- Item-level validation WER/CER: 32.5182% / 7.0830%
- Item-level test WER/CER: 31.3560% / 7.0103%

Metric and hashed item-prediction artifacts are present. This run is the
confirmatory speaker-disjoint random control unless the matched non-disjoint
diagnostic identifies an implementation issue rather than the expected
speaker-generalization gap.