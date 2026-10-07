# Medium RTX 4090 Memory Probe (2026-10-06)

## Outcome

Both clean C0 and LD-mask S4 probes passed using the unchanged queue settings:
train/evaluation batch 6, accumulation 1, FP16, non-reentrant gradient
checkpointing, AdamW, pinned Whisper Medium, and generation maximum length 225.

| Probe | Peak allocated training GiB | Peak reserved across probe GiB | Non-skipped updates |
|---|---:|---:|---:|
| Clean C0 | 14.235 | 14.994 | 2 |
| LD masks S4 | 14.234 | 14.986 | 2 |

The initial two-attempt clean probe was inconclusive: FP16 scaling skipped both
updates, leaving Adam state uninitialized. It is retained in
`memory_clean_batch6_20261006.json`; it must not be interpreted as a passed test.
The corrected clean result is `memory_clean_batch6_20261006_v2.json`.
LD results are in `memory_ld_batch6_20261006.json`.

Both corrected probes required six attempts: four initial scaling skips,
followed by two non-skipped optimizer steps with 946 optimizer-state entries.
No batch, precision, optimizer, or augmentation fallback was applied.

## Coverage

- Real WAXAL train examples selected by longest tokenized transcripts (184-190
  tokens including tokenizer special tokens), repeated for a bounded test.
- Six longest-tokenized validation examples for generation and loss evaluation.
- Production model loader, feature preparation, collator, augmentation hook,
  Seq2SeqTrainer, and training-argument construction.
- Normal generation evaluation followed by a diagnostic minimum-length override
  forcing generation to the configured maximum; optimizer states remain resident.
- CUDA allocation/reservation peaks and other visible process usage recorded.

The tensor widths reported by Trainer include output padding and must not be
interpreted as counts of generated non-padding tokens.

## Limits and safety

These are memory diagnostics, not experiment results or quality metrics. Only
two successful optimizer updates were required, with a cap of 16 attempts.
Automatic evaluation, saving, W&B reporting, and best-model loading were
disabled only for the probe. No model/checkpoint was saved. The experimental
configs and model output directories were not changed. No queue was launched.

The RTX 4090 had an unrelated Python process occupying approximately 540 MiB;
it was not stopped. After both tests the GPU returned to 566 MiB total usage.
PyTorch peaks are process-local and exclude other processes and some CUDA
overhead. Free-memory snapshots are not a continuously sampled minimum.

Results support testing the first full run without a batching accommodation.
They do not certify all batches, seeds, samplers, waveform inputs, checkpoint
I/O, dataset-wide evaluation buffering, or long-run fragmentation. Monitor
the first complete Medium run and stop on OOM rather than silently adapting.

## Reproduction

Use a new report filename each time:

```bash
HF_HOME=/ext_data/casper/huggingface_cache \
  .venv/bin/python asr.py test_medium_memory \
  --config experiments/medium/rq1/c0_random_seed42_v2/config.json \
  --report artifacts/experiment_outputs/medium/diagnostics/memory_clean_new.json
```