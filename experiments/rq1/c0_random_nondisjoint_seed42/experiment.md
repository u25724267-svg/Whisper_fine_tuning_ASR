# RQ1 Diagnostic: Non-Disjoint Random Control, Seed 42

## Purpose

Estimate the effect of returning to the official speaker-overlapping WAXAL
split under the same training and epoch-checkpoint policy as C0-v2.

## Matched factors

- Pinned multilingual Whisper Base
- Seed 42 and full deterministic training
- No curriculum and no augmentation
- Three epochs and effective batch size six
- Learning rate, warmup, optimizer, decoding, and collation
- Epoch-boundary evaluation and saving
- Three retained checkpoints

## Changed factor

Only the data protocol changes from speaker-disjoint v2 to the original
normalized WAXAL train, validation, and test manifests. The eight-row difference
in training size changes epoch boundaries from 2,302/4,604/6,906 to
2,300/4,600/6,900.

The official test set is not speaker-independent: 1,700 of 1,711 test
utterances have speakers represented in training. This run is diagnostic and
cannot replace the speaker-disjoint control for generalization claims.

## Status

Launched on 2026-09-01 in tmux session
`rq1-c0-random-nondisjoint-seed42`.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_nondisjoint_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_nondisjoint_seed42/logs/train.log`

Results must not be added until metric and item-prediction artifacts are
present.