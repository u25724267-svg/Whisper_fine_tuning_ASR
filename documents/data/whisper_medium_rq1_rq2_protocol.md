# Whisper Medium RQ1/RQ2 Extension

## Scope and status

Prepared on 2026-10-03, before Medium extension results. Training is not launched
by preparation. This is a model-size extension of the frozen Base protocols,
not a rerun of the legacy speaker-overlapping Medium pilots.

| Order | Cell | Ordering and treatment | Seeds |
|---|---|---|---|
| 1 | C0 / S0 / A0 | Clean, seeded random | 42, 43, 44 |
| 2 | C1 / S1 | Clean, SortaGrad | 42, 43, 44 |
| 3 | C3 / A1 | Clean, strict SNR-duration | 42, 43, 44 |
| 4 | A2 | Materialized mild waveform augmentation, random | 42, 43, 44 |
| 5 | A3 | Same waveform augmentation, C3 ordering | 42, 43, 44 |
| 6 | S2 | LB masks, random | 42, 43, 44 |
| 7 | S3 | LB masks, SortaGrad | 42, 43, 44 |
| 8 | S4 | LD masks, random | 42, 43, 44 |
| 9 | S5 | LD masks, SortaGrad | 42, 43, 44 |

Total: 27 runs. A0/A1 and S0/S1 reuse the Medium clean controls; no duplicate
clean runs are created. B-series and old Hugging Face mask-budget pilots are
excluded. FLEURS inference and statistical analysis are separate follow-ups,
not implicit training-queue stages.

## Frozen adaptation

- `openai/whisper-medium`, revision `abdf7c39ab9d0397620ccaea8974cc764cd0953e`.
- All runs start independently from the pretrained model, never a prior cell.
- Same speaker-disjoint WAXAL v2 splits and same seed-specific waveform data.
- Three epochs, learning rate 1e-5, 500 warm-up steps, epoch evaluation/saving,
  validation-WER best checkpoint, FP16, and gradient checkpointing.
- Medium microbatch 2, accumulation 3, effective batch 6, evaluation batch 2.
  This follows the existing Medium memory profile, not a tuned scientific claim.
  It does not imply bitwise equivalence to Base or identical mask RNG draws.
- Greedy generation, Shona transcription, maximum length 225.
- LB: F=27, mF=1, T=100, p=1, mT=1. LD doubles mF and mT to 2.
- No time warping. No combination of waveform and spectrogram augmentation.
- Same existing W&B project: `whisper-shona-multilingual`.
- Three retained checkpoints; conservative reserve 32 GiB/run, 864 GiB total.
  This is an estimate, with free space checked again before launch and each run.

Keeping effective batch size does not erase differences in model size,
microbatch numerical behavior, or stochastic augmentation draws. Base-to-Medium
comparisons must explicitly report this resource adaptation. For deterministic
strict C3 order, changing seeds may again yield identical results; repeated
artifacts would be repeatability evidence, not independent seed variation.

## Controls and provenance

Each generated config records the Base source path and SHA-256. Every declared
control is remapped to its Medium counterpart. All other data, curriculum,
augmentation, optimizer, and checkpoint-selection settings are retained.

Matched contrasts include C1-C0, C3-C0, A2-C0, A3-C3, A3-A2, S2-C0, S3-C1,
S4-S2, and S5-S3, with matched seeds. The three clean C3 controls support the
waveform augmentation contrast under C3. Selection must use WAXAL validation;
test and FLEURS must not tune these settings.

## Operation and storage

Prepare with `.venv/bin/python asr.py prepare_medium_queue`. Repeated preparation
accepts identical files and refuses differing files. `--check` validates frozen
derivation, input paths, waveform manifest hashes, and unused destinations.
`--dry-run-runners` validates all 27 real runner paths without training.

Run `./scripts/run_medium_rq1_rq2.sh --preflight-only` for launch preflight only.
Only `./scripts/run_medium_rq1_rq2.sh` without that flag starts the detached
controller `whisper-medium-rq1-rq2`. It uses one GPU job at a time. A failed
training or prediction stage stops the queue; partial outputs are not erased or
automatically resumed. Validate the first actual Medium run's memory footprint
before claiming the whole queue is GPU-tested; runner dry-runs do not train.

Definitions live in `experiments/medium/{rq1,rq2}/<cell>/`. Heavy outputs live
under `/ext_data/casper/asr_experiment_outputs/medium/{rq1,rq2}/<cell>/` (or the
explicit `ASR_OUTPUT_ROOT`). The existing sequence runner exports validation/test
predictions and hash-verifies a lightweight mirror into
`artifacts/experiment_outputs/medium/{rq1,rq2}/<cell>/` before advancing.
The controller log is written directly under that repository's `medium/logs/`.
Existing Base outputs and historical configs are never overwritten.

## Literature and interpretation

This extension transfers established policies; its three-epoch schedule,
microbatch adaptation, and model-size comparison are study choices rather than
exact replications of the cited experiments.

- Radford et al. (2023), Robust Speech Recognition via Large-Scale Weak
  Supervision: https://proceedings.mlr.press/v202/radford23a.html
- Amodei et al. (2016), Deep Speech 2 (SortaGrad):
  https://proceedings.mlr.press/v48/amodei16.html
- Park et al. (2019), SpecAugment (LB/LD policies):
  https://doi.org/10.21437/Interspeech.2019-2680

The exact waveform recipe and its adaptations remain in
`rq2_waveform_augmentation_policy.md`; this queue does not redesign that recipe.