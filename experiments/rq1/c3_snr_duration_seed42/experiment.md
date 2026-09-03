# RQ1-C3: Joint SNR-Duration Curriculum, Seed 42

## Purpose

Test whether a joint acoustic difficulty score improves Whisper Base relative
to C0-v2 and resolves limitations of the single-signal C1 and C2 orders.

## Treatment

For each training utterance,

$$
s_i=0.5q_i^{duration}+0.5(1-q_i^{SNR}).
$$

Duration and SNR percentiles are fitted on the training split only using
tie-aware average ranks. Short, high-SNR utterances receive lower difficulty;
long, low-SNR utterances receive higher difficulty.

- Order: ascending joint difficulty
- Ties: utterance ID, then dataset index
- Epochs: the same complete static order is applied in all three epochs
- Order artifacts: one item-level JSONL file per epoch

## Matched control

The control is `rq1-c0-random-seed42-v2`. Model revision, data manifests,
preprocessing, augmentation, optimizer, seed, batch size, epoch count, learning
rate schedule, checkpoint boundaries, decoding, validation, and test evaluation
are unchanged. Only training-example order differs.

## Motivation after C1 and C2

C1 improved test WER by 0.3407 points but did not cross the practical threshold.
C2 worsened test WER by 0.1783 points. C3 remains preregistered because duration
may provide complementary difficulty information and differentiates the 2,665
training rows tied at the 60 dB SNR ceiling.

## Required artifacts

- Epoch checkpoints at steps 2,302, 4,604, and 6,906
- Three complete joint-score order JSONL files
- Acoustic metadata hash in the run manifest
- Validation and test metrics
- Hashed item-level validation and test predictions

## Status

Completed on 2026-09-02.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c3_snr_duration_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c3_snr_duration_seed42/logs/train.log`
- Best checkpoint: epoch 3, step 6,906
- Trainer validation WER: 32.8288%
- Trainer test WER: 30.5522%
- Item-level validation WER/CER: 32.8368% / 7.1317%
- Item-level test WER/CER: 30.4564% / 6.4664%

All metric, curriculum-order, and hashed item-prediction artifacts are present.

## Comparison with C0-v2

| Metric | C0-v2 random | C3 joint | C3 change |
|---|---:|---:|---:|
| Item-level validation WER | 32.5182% | 32.8368% | +0.3186 |
| Item-level test WER | 31.3560% | 30.4564% | -0.8995 |
| Item-level test CER | 7.0103% | 6.4664% | -0.5439 |

A 10,000-replicate paired bootstrap clustered over the 24 test speakers gave a
95% interval of 0.3116--1.3958 WER points for the C3 improvement and a 0.9993
bootstrap probability of improvement. This supports a real seed-42 test effect,
but validation worsened by 0.3186 points. C3 is therefore retained as a
promising static candidate but is not promoted until the seed-42 static screen
is complete and the effect is replicated with seeds 43 and 44.