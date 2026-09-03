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

Completed on 2026-09-01.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_nondisjoint_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_nondisjoint_seed42/logs/train.log`
- Best checkpoint: epoch 2, step 4,600
- Validation WER: 29.2918%
- Test WER: 28.7307%
- Item-level validation WER/CER: 29.2995% / 6.6919%
- Item-level test WER/CER: 28.7232% / 6.2297%

Metric and hashed item-prediction artifacts are present.

## Comparison and interpretation

| Run | Validation WER | Test WER |
|---|---:|---:|
| Original official-split Base | 29.0647% | 28.4893% |
| Matched official-split diagnostic | 29.2918% | 28.7307% |
| Speaker-disjoint C0-v2 | 32.5262% | 31.3400% |

The matched official-split diagnostic differs from the original baseline by
only +0.2271 validation and +0.2414 test WER points, both below the
preregistered 0.5-point practical threshold. Relative to the matched diagnostic,
speaker-disjoint v2 is higher by 3.2343 validation and 2.6093 test points.

This strongly supports the speaker-disjoint protocol as the source of the
performance disparity rather than model, optimizer, or checkpointing drift.
The difference should be interpreted as the combined effect of evaluating
unseen speakers and changing the held-out speaker composition, not as a pure
causal estimate of speaker familiarity. The official-split run remains a
secondary in-domain benchmark; C0-v2 remains the confirmatory RQ1 control.