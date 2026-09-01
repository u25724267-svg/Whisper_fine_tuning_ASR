# RQ1-C0: Random Control, Seed 42

## Purpose

Establish the conventional shuffled Whisper Base reference for all RQ1
curriculum treatments on the speaker-disjoint WAXAL Shona protocol.

## Treatment

- Curriculum: none; seeded Transformers random sampler
- Augmentation: none
- Model: pinned multilingual Whisper Base
- Training: three epochs, effective batch size six, seed 42
- Selection: lowest validation WER, evaluated every 1,000 optimizer steps

## Data

- Protocol: `waxal_shona_speaker_disjoint_v1`
- Train: 13,778 utterances, 79.48 hours, 129 speakers
- Validation: 1,696 utterances, 9.71 hours, 16 speakers
- Test: 1,719 utterances, 9.72 hours, 16 speakers
- Speaker overlap: zero across all splits

## Status

Launched on 2026-09-01 in tmux session `rq1-c0-random-seed42`.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42/logs/train.log`

Results must not be added until saved metric files are present.