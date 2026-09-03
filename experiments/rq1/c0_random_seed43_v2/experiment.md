# RQ1-C0: Random Control, Seed 43

## Purpose

Provide the shared seed-43 random-order control for C3 and C5 replication on
the fixed speaker-disjoint WAXAL v2 protocol.

## Contract

- Identical data, model revision, optimizer, update budget, and evaluation to C0 seed 42
- Training and data seed: 43
- Standard Transformers random sampler
- No curriculum or augmentation
- Epoch-boundary validation and checkpointing
- Hashed item-level validation/test predictions required

## Status

Launched on 2026-09-03 as the sole CUDA workload and progressed beyond
optimizer step 35 without runtime or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `i6on623a`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed43_v2`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed43_v2/logs/train.log`

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully.

- Validation WER/CER: 33.4422 / 7.5620
- Test WER/CER: 30.7891 / 6.6774