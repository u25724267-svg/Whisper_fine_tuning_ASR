# RQ1-C2: Strict SNR Curriculum, Seed 42

## Purpose

Test whether presenting acoustically cleaner, higher-SNR utterances before
lower-SNR utterances improves Whisper Base relative to C0-v2.

## Treatment

- SNR proxy: 75th-percentile active-frame RMS minus 20th-percentile all-frame
  RMS in dB, capped at 60 dB
- Ranking: tie-aware SNR percentile fitted on training only
- Difficulty: `1 - snr_percentile`
- Order: ascending difficulty, equivalent to high-to-low SNR
- Ties: utterance ID, then dataset index
- Epochs: the same complete static order is applied in all three epochs
- Order artifacts: one item-level JSONL file per epoch

## Matched control

The control is `rq1-c0-random-seed42-v2`. Model revision, data manifests,
preprocessing, augmentation, optimizer, seed, batch size, epoch count, learning
rate schedule, checkpoint boundaries, decoding, validation, and test evaluation
are unchanged. Only training-example order differs.

## Metadata audit

- 17,193 protocol rows decoded successfully
- 13,807 training IDs aligned exactly
- 16,663 unchanged-path PCM hashes verified
- 530 silence-trimmed path replacements recorded
- Same-path hash mismatches: zero
- Native audio: mono 48 kHz; training loader resamples to 16 kHz
- Training SNR range: 7.676--60.000 dB
- Rows capped at 60 dB: 2,665 (19.3%)

The 60 dB ceiling creates a large clean-speech tie group. Its deterministic ID
tie-break is retained and reported rather than tuned after observing results.

## Required artifacts

- Epoch checkpoints at steps 2,302, 4,604, and 6,906
- Three complete SNR-order JSONL files
- Acoustic metadata hash in the run manifest
- Validation and test metrics
- Hashed item-level validation and test predictions

## Status

Completed on 2026-09-02.

- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c2_snr_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c2_snr_seed42/logs/train.log`
- Best checkpoint: epoch 3, step 6,906
- Trainer validation WER: 33.1130%
- Trainer test WER: 31.5103%
- Item-level validation WER/CER: 33.1183% / 7.3623%
- Item-level test WER/CER: 31.5343% / 7.4551%

All metric, curriculum-order, and hashed item-prediction artifacts are present.

## Comparison with C0-v2

| Metric | C0-v2 random | C2 strict SNR | C2 change |
|---|---:|---:|---:|
| Item-level validation WER | 32.5182% | 33.1183% | +0.6001 |
| Item-level test WER | 31.3560% | 31.5343% | +0.1783 |
| Item-level test CER | 7.0103% | 7.4551% | +0.4448 |

C2 worsens all three matched metrics and is not promoted. The result does not
rule out the preregistered joint C3 score: duration may add complementary
difficulty information and resolves part of the 60 dB SNR ceiling tie.