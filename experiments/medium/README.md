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

All 27 actual runner dry-runs passed on 2026-10-03. This validates processor,
data and ordering contracts, not a full Medium training memory measurement.

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

The batch profile is 2 examples/device with 3 accumulation steps (effective 6).
Three checkpoints are retained. The launcher requires a conservative 864 GiB
external disk reserve for the complete queue and 20000 MiB free GPU memory.
Failure stops the queue; partial outputs are never automatically erased or
resumed. These checks require unused run destinations, so inspect partial runs
before attempting a restart. FLEURS evaluation is not part of this training queue.

Definitions are generated from unchanged same-seed Base configs, with their
paths and hashes recorded in `medium_extension`. Regeneration accepts identical
files but refuses drift. See
[the frozen protocol](../../documents/data/whisper_medium_rq1_rq2_protocol.md).