# RQ1-C5: Dynamic S2S-Margin, Seed 42

## Purpose

Test whether model-dependent sequence difficulty improves speaker-disjoint Shona
ASR when the update budget and all non-ordering settings match C0-v2.

## Treatment

- Epoch 1: fixed seed-42 random permutation
- Score: summed teacher-forced token NLL divided by audio duration
- Score timing: online training forward, before that batch's optimizer update
- Epochs 2-3: ascending prior-epoch score
- Tie-breakers: duration, utterance ID, dataset index
- Every row appears exactly once per epoch
- No duration warm-up, replacement, pacing, or difficulty mixing

## Required artifacts

- Three item-level score files
- Three item-level presentation-order files
- Three epoch checkpoints
- Validation/test metrics and hashed item predictions

## Status

Dynamic state contract, full-data dry run, real Whisper loss reconstruction,
and a two-epoch Trainer transition test passed on 2026-09-02. Launched alongside
C4R in tmux session `rq1-c5-s2s-margin-seed42`; live score capture progressed
beyond optimizer step 200 without triggering alignment or reconstruction guards.

- W&B project: `whisper-shona-multilingual`
- W&B run: `sm6gwzi7`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed42/logs/train.log`

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully.

- Validation WER/CER: 33.8352 / 7.9178
- Test WER/CER: 31.3160 / 6.9650
- Test WER change versus C0-v2: 0.0400 points better, practically unresolved