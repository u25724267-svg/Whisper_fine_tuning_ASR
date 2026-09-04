# RQ1-C0: Random Control, Seed 44

## Purpose

Provide the shared seed-44 random-order control for the final C5 replication on
the fixed speaker-disjoint WAXAL v2 protocol.

## Contract

- Identical data, model revision, optimizer, update budget, and evaluation to C0 seeds 42/43
- Training and data seed: 44
- Standard Transformers random sampler
- No curriculum or augmentation
- Epoch-boundary validation and checkpointing
- Hashed item-level validation/test predictions required

## Status

Validated and launched on 2026-09-03 after C5 seed-43 item evaluation completed.
The full-data dry run passed, and all behavior-controlling fields match C0 seed
43 except the shared seed and experiment identity. The run progressed beyond
optimizer step 35 without runtime or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `jaz5s4wo`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed44_v2`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed44_v2/logs/train.log`

Training, final Trainer evaluation, and hashed item-level prediction export
completed successfully.

- Validation WER/CER: 32.7572 / 7.3003
- Test WER/CER: 30.9887 / 6.7698