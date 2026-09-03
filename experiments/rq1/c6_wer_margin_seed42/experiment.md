# RQ1-C6: Dynamic WER-Margin, Seed 42

## Purpose

Test whether model-dependent recognition error is a useful curriculum signal
under the same speaker-disjoint protocol and optimizer budget as C0-v2.

## Treatment

- Epoch 1: fixed seed-42 random permutation
- Score: per-item greedy WER captured before the training forward
- Epochs 2-3: ascending prior-epoch WER
- Tie-breakers: duration, utterance ID, dataset index
- Every row appears exactly once per epoch
- No duration warm-up, replacement, pacing, or difficulty mixing
- Generation: greedy, maximum length 225, Shona transcription

## Required artifacts

- Three item-level WER files with references, predictions, and S/D/I/H counts
- Three item-level presentation-order files
- Three epoch checkpoints
- Validation/test metrics and hashed item predictions
- Runtime and GPU-cost comparison against C0-v2

## Status

On 2026-09-02, the dynamic WER state contract, 13,807-row dry run, and real
Whisper pre-forward generation smoke test passed. Training and final Trainer
evaluation then completed successfully, followed by hashed item-level prediction
export.

- Validation WER/CER: 34.0316 / 7.6830
- Test WER/CER: 31.7711 / 6.8128
- Test WER change versus C0-v2: 0.4151 points worse
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c6_wer_margin_seed42`

## Concurrency audit

C6 and C7 did not train concurrently. C6's local W&B run began at 12:35 UTC
and its training log completed at 16:41 UTC on 2026-09-02. C7 began at 20:50
UTC. C5, C6, and C7 have the identical epoch-1 checkpoint model hash
`58dbb9b96b317c3fd02502322857d747194582b0f1f1fae5ef91b06a9b0366cb`,
as expected from their matched random first epoch. Their epoch-2 and epoch-3
orders and model hashes differ. GPU co-scheduling is therefore not a supported
explanation for C6's result.