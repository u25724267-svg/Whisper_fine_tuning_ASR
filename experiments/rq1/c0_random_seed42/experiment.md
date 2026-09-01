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

Completed on 2026-09-01. This run is retained as a diagnostic pilot and is not
the confirmatory RQ1 control.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42/logs/train.log`
- Best checkpoint: step 6,000
- Validation WER: 30.1819%
- Test WER: 28.2441%
- Item-level validation WER/CER: 30.1940% / 6.0594%
- Item-level test WER/CER: 28.2587% / 5.9954%

## Diagnostic disposition

The run is not directly comparable with the original 28.4893% Base result
because it uses a new speaker-disjoint repartition. The new training set shares
11,022 rows with the old training set, while 2,777 old training rows moved to
held-out splits and 2,756 old validation/test rows moved into training.

The split also concentrated 51.1% of validation rows in one speaker and 35.1%
of test rows in one speaker. The dominant validation speaker contributed 58.9%
of validation errors, while the dominant test speaker had a relatively easy
22.78% WER. This makes the validation/test difference sensitive to a small
number of voices.

Finally, training reached step 6,891, but evaluation and saving occurred every
1,000 steps. `load_best_model_at_end` restored checkpoint 6,000, so the saved
model excludes the final 891 optimization updates. The replacement control must
use a less concentrated speaker split and epoch-boundary evaluation/saving.