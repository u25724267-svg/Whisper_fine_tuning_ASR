# Whisper Medium Queue

Prepared only: no training was launched during setup.

`queue.json` lists 27 independent runs, with seeds 42, 43, and 44 for each cell:
C0 random, C1 SortaGrad, clean C3, waveform A2/A3, and SpecAugment S2/S3/S4/S5.
Each run starts from the pinned pretrained Whisper Medium checkpoint. LB/LD
exclude time warping; B-series and legacy overlapping-data pilots are excluded.

## Validate without training

```bash
.venv/bin/python asr.py prepare_medium_queue --check
.venv/bin/python asr.py prepare_medium_queue --dry-run-runners
./scripts/run_medium_rq1_rq2.sh --preflight-only
```

The original 27 runner dry-runs passed on 2026-10-03. On 2026-10-06, the
unstarted definitions were amended to inherit exact Base batching instead of
2 x 3 accumulation. Dry-runs validate processor, data, and ordering contracts,
not full Medium training memory feasibility.
All 27 corrected dry-runs passed on 2026-10-06. The historical audit also matched
all 27 Base scientific configs and recorded package versions, plus 87 input
hash comparisons. On 2026-10-06, bounded clean/LD batch-6 CUDA probes passed,
including two non-skipped AdamW updates and generation evaluation; maximum
PyTorch reservation was approximately 15 GiB. Full-run memory monitoring remains
required. Reports: `artifacts/experiment_outputs/medium/diagnostics/`.

## Launch later

Only when ready to train:

```bash
./scripts/run_medium_rq1_rq2.sh
```

This starts the detached `whisper-medium-rq1-rq2` controller, runs one job at a
time, exports WAXAL validation/test item predictions, and mirrors scientific
outputs into `artifacts/experiment_outputs/medium/` after each run. Model and
checkpoint outputs stay under `/ext_data/casper/asr_experiment_outputs/medium/`.
The controller log goes directly into the repository under `medium/logs/`.

The batch profile exactly matches Base: 6 examples/device, 1 accumulation step,
and evaluation batch 6. No automatic memory fallback is allowed. If this does
not fit, stop and review hardware or a matched-control protocol amendment.
Three checkpoints are retained. The launcher requires a conservative 864 GiB
external disk reserve for the complete queue and 20000 MiB free GPU memory.
Failure stops the queue; partial outputs are never automatically erased or
resumed. These checks require unused run destinations, so inspect partial runs
before attempting a restart. FLEURS evaluation is not part of this training queue.

Definitions are generated from unchanged same-seed Base configs, with their
paths and hashes recorded in `medium_extension`. Regeneration accepts identical
files but refuses drift. See
[the frozen protocol](../../documents/data/whisper_medium_rq1_rq2_protocol.md).