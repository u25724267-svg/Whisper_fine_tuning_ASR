# RQ1-C7: Acoustic and S2S Hybrid, Seed 42

## Purpose

Test whether a fixed acoustic prior stabilizes dynamic model-based difficulty
under the same speaker-disjoint protocol and optimizer budget as C0-v2.

## Treatment

- Epoch 1: fixed seed-42 random permutation
- Acoustic component: C3's equal-weight duration and inverse-SNR percentile score
- Dynamic component: percentile rank of summed token NLL per audio second
- Hybrid score: 0.5 acoustic + 0.5 dynamic S2S
- Epochs 2-3: ascending prior-epoch hybrid score
- Tie-breakers: duration, utterance ID, dataset index
- Every row appears exactly once per epoch
- No duration warm-up, replacement, pacing, or difficulty mixing

## Required artifacts

- Three item-level component and hybrid score files
- Three item-level presentation-order files
- Acoustic metadata hash and runner/state hashes
- Three epoch checkpoints
- Validation/test metrics and hashed item predictions

## Status

On 2026-09-02, hybrid arithmetic and ordering, atomic audits, the 13,807-row
acoustic-alignment dry run, and a real Whisper loss-capture integration test
passed. The integration test verified that every hybrid audit score exactly
matched the fixed 0.5/0.5 component formula. C7 then launched successfully and
progressed beyond optimizer step 200 without triggering score or alignment
guards.

- W&B project: `whisper-shona-multilingual`
- W&B run: `99zxhpa9`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c7_acoustic_s2s_hybrid_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c7_acoustic_s2s_hybrid_seed42/logs/train.log`

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully.

- Validation WER/CER: 34.5229 / 8.4135
- Test WER/CER: 32.1916 / 7.5697
- Test WER change versus C0-v2: 0.8356 points worse

## Concurrency audit

C7 did not overlap C6 training. C6 completed at 16:41 UTC and C7's local W&B
run began at 20:50 UTC on 2026-09-02. C5, C6, and C7 have the identical epoch-1
order and checkpoint model hash
`58dbb9b96b317c3fd02502322857d747194582b0f1f1fae5ef91b06a9b0366cb`.
Their model hashes diverge after the first dynamic reordering, and C7's later
orders are distinct from C6's. The common improve-at-epoch-2, regress-at-epoch-3
trajectory reflects the dynamic curriculum family rather than concurrent GPU
execution.