# RQ1-C5: Seed-42 Repeatability Rerun

## Purpose

Test exact repeatability of the original C5 result under the same seed, data,
model revision, optimizer, dynamic score, epoch orders, and checkpoint-selection
contract. This is not an independent-seed replication and does not increase the
inferential sample size.

## Comparison contract

- Original run: `rq1-c5-s2s-margin-seed42`
- Same seed: 42
- Same epoch-1 seeded random order
- Same online summed token NLL per audio second
- Same strict prior-epoch score ordering
- Same three-epoch update and evaluation budget
- New isolated output directory and W&B run

The runner and state file hashes differ from the original run because the same
files were subsequently extended with C6 and C7 branches. The C5 branch and its
score/sampler behavior are unchanged. Repeatability will be assessed using
epoch order hashes, checkpoint model hashes, validation/test metrics, and hashed
item-level predictions.

## Status

Launched on 2026-09-03 and progressed beyond optimizer step 35 without score,
alignment, or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `ydqxuavg`
- Output: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed42_rerun`
- Log: `/ext_data/casper/asr_experiment_outputs/rq1/c5_s2s_margin_seed42_rerun/logs/train.log`

Training and Trainer evaluation completed successfully. The final model and
all three checkpoint model files are byte-identical to the original C5 run;
validation WER 33.8245 and test WER 31.3480 also match exactly. Item-level
prediction export is still running. This confirms deterministic repeatability
but does not provide an independent replication.