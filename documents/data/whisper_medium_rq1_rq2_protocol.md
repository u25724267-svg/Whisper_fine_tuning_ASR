# Whisper Medium RQ1/RQ2 Extension

## Scope and status

Prepared on 2026-10-03, before Medium extension results. Training is not launched
by preparation. This is a model-size extension of the frozen Base protocols,
not a rerun of the legacy speaker-overlapping Medium pilots.

Amended on 2026-10-06 before any Medium runs: removed the initially proposed
2-example microbatch / 3-step accumulation accommodation. The intended changed
modeling factor is now only the pinned pretrained model variant. No training
or evaluation parameter may change implicitly to accommodate memory.

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
- Exact inherited Base batching: train batch 6, accumulation 1, evaluation batch 6.
  Learning-rate schedule, final-batch handling, checkpoint cadence, precision,
  optimizer settings, and gradient-checkpointing options are unchanged.
- Greedy generation, Shona transcription, maximum length 225.
- LB: F=27, mF=1, T=100, p=1, mT=1. LD doubles mF and mT to 2.
- No time warping. No combination of waveform and spectrogram augmentation.
- Same existing W&B project: `whisper-shona-multilingual`.
- Three retained checkpoints; conservative reserve 32 GiB/run, 864 GiB total.
  This is an estimate, with free space checked again before launch and each run.

Larger checkpoint storage and admission thresholds are operational differences,
not changes to optimization. Runtime and memory use will naturally differ.
The same seeds do not guarantee identical stochastic draws or predictions
across different architectures. For deterministic strict C3 order, repeated
seeds may again produce identical artifacts; this would be repeatability
evidence, not independent seed variation.

If batch 6 does not fit, stop before launching the full queue. Do not silently
reduce batch size, increase accumulation, enable a different optimizer, alter
precision, or truncate inputs. Either use sufficient-memory hardware or obtain
approval for a documented accommodation with matched Base controls. Equal
effective batch size alone is not proof of equivalent optimization or update
budgets. Full training memory feasibility remains unverified by dry-runs.
The subsequent bounded clean/LD probes described below provide runtime evidence,
but not a full-run memory guarantee.

## Controls and provenance

Each generated config records the Base source path and SHA-256. Every declared
control is remapped to its Medium counterpart. All other data, curriculum,
augmentation, optimizer, and checkpoint-selection settings are retained.

The full-config parity test allows only model ID/revision and operational
identity, paths, resources, W&B metadata, control references, and provenance
to differ. The launcher uses the same runner, sequence, item-prediction export,
and artifact-mirroring code as the Base family. Actual epoch/update counts,
data and order hashes, model generation settings, and software versions must
be checked against recorded Base run artifacts after execution. Source hashes
have changed through refactoring; config parity is not proof of byte-identical
historical runtime behavior.

Matched contrasts include C1-C0, C3-C0, A2-C0, A3-C3, A3-A2, S2-C0, S3-C1,
S4-S2, and S5-S3, with matched seeds. The three clean C3 controls support the
waveform augmentation contrast under C3. Selection must use WAXAL validation;
test and FLEURS must not tune these settings.

## Operation and storage

### Pre-run audit on 2026-10-06

- All 27 corrected runner dry-runs passed with exact Base batching.
- All 27 Base source configurations matched saved scientific blocks in the
  completed Base runs' `experiment_config.json` files.
- Package versions recorded in all 27 Base run manifests matched the current
  environment. This does not establish equivalence of all hardware or code.
- All 87 train/validation/test/acoustic-metadata hash comparisons passed across
  seven unique input files against the historical Base run manifests.
- Eight focused queue tests passed, including exact full-config scientific
  parity and protection against replacing differing definitions.
- No Medium training outputs existed and no training was launched during audit.
- Subsequent batch-6 runtime probes passed for clean C0 and LD S4 on the RTX
  4090: two non-skipped AdamW updates, normal generation evaluation, and
  full-length generation stress. Maximum allocated training memory was about
  14.235 GiB; peak reservation was about 14.994 GiB. Both probes needed four
  initial FP16-skipped attempts before successful updates. No checkpoints or
  W&B runs were created; no experiment queue was started.
- Evidence is under `artifacts/experiment_outputs/medium/diagnostics/`.
  This bounded sample does not certify long-run memory or every input batch.

Prepare with `.venv/bin/python asr.py prepare_medium_queue`. Repeated preparation
accepts identical files and refuses differing files. `--check` validates frozen
derivation, input paths, waveform manifest hashes, and unused destinations.
`--dry-run-runners` validates all 27 real runner paths without training.

Run `./scripts/run_medium_rq1_rq2.sh --preflight-only` for launch preflight only.
Only `./scripts/run_medium_rq1_rq2.sh` without that flag starts the detached
controllers `whisper-medium-rq1-rq2-gpu0` and `whisper-medium-rq1-rq2-gpu1`.
Operational amendment on 2026-10-10: alternating queue entries run in two
independent sequential lanes, pinned by UUID to the first two GPUs. Each run's
training and prediction export use only its assigned GPU; batching and frozen
configs are unchanged. Preflight requires 20,000 MiB free on each GPU. A failed
training or prediction stage stops its lane; the other lane may continue.
Partial outputs are not erased or automatically resumed. Global completion
order can differ from the listed queue order. Validate each GPU's actual Medium
memory footprint before claiming full-run feasibility; runner dry-runs do not
train.

Definitions live in `experiments/medium/{rq1,rq2}/<cell>/`. Heavy outputs live
under `/ext_data/casper/asr_experiment_outputs/medium/{rq1,rq2}/<cell>/` (or the
explicit `ASR_OUTPUT_ROOT`). The existing sequence runner exports validation/test
predictions and hash-verifies a lightweight mirror into
`artifacts/experiment_outputs/medium/{rq1,rq2}/<cell>/` before advancing.
Each lane's controller log is written under that repository's `medium/logs/`.
Existing Base outputs and historical configs are never overwritten.

### Bounded storage batches on 2026-10-10

After the six C0/C1 runs completed, the remaining 21 runs were divided into
two near-equal batches in frozen queue order. The original 27-run definition,
32 GiB per-run reserve, and scientific configs remain unchanged. Only the first
batch is launched using the existing sequence runner, with five runs per GPU.

| Batch | Queue entries | Runs | Local reserve | Conditions |
|---|---|---:|---:|---|
| 1 | 7--16 | 10 | 320 GiB | C3, A2, A3 seeds 42--44; S2 seed 42 |
| 2 | 17--27 | 11 | 352 GiB | S2 seeds 43--44; S3, S4, S5 seeds 42--44 |

Batch 1 GPU 0 runs C3 seed 42, C3 seed 44, A2 seed 43, A3 seed 42, and A3
seed 44. GPU 1 runs C3 seed 43, A2 seed 42, A2 seed 44, A3 seed 43, and S2
seed 42. Controllers are `medium-batch1-gpu0` and `medium-batch1-gpu1`;
training and prediction exports remain pinned to their lane's GPU UUID.

Completed C1 seeds 43/44 were uploaded to
`gs://casper/completed-runs/2026-10-10/medium/rq1/c1_sortagrad_seed43/` and
`gs://casper/completed-runs/2026-10-10/medium/rq1/c1_sortagrad_seed44/`.
Checksum-only comparisons passed for all 78 and 65 regular files respectively.
W&B convenience symlinks were excluded. With explicit user approval, only those
two local heavy output directories were removed; repository mirrors and the
original Qhivi/GCS backups were retained. About 366 GiB was free before batch 1.

Batch 2 is not automatically started. Before launching it, completed batch-1
outputs must be backed up and verified, any local cleanup must be authorized,
and the 352 GiB admission reserve must be checked again. Partial output
directories still stop their lane and are never automatically overwritten.

### Batch-two preparation on 2026-10-12

All ten batch-one runs completed with canonical prediction exports and verified
repository mirrors. Their full regular-file outputs were uploaded to
`gs://casper/completed-runs/2026-10-12/medium/` by
`scripts/backup_medium_batch1_gcs.sh`. The detached backup exited 0 with all ten
checksum comparisons clean. Verification logs and per-run markers are under
`artifacts/experiment_outputs/medium/logs/backup-batch1-20261012T205152Z/`.

With explicit user approval, the ten backed-up local heavy output directories
were removed after confirming that their source files had not changed since
verification. Repository mirrors, active data, and all GCS copies were retained.
Local free space was 365.87 GiB, above the 352 GiB batch-two reserve.

All eleven remaining configs passed frozen-derivation parity, input-path,
credential-presence, unused-destination, and actual runner dry-run checks.
The planned independent single-GPU sequences preserve queue entries 17--27:

| GPU | Ordered runs |
|---|---|
| 0 | S2 seed 43; S3 seeds 42/44; S4 seed 43; S5 seeds 42/44 |
| 1 | S2 seed 44; S3 seed 43; S4 seeds 42/44; S5 seed 43 |

Preparation did not launch training or create batch-two outputs. The existing
sequence runner will be used for these six-run/five-run lanes, with explicit
GPU UUID assignments. Recheck disk and GPU admission immediately before launch;
the full 27-run controller and its original reserve are unchanged.

## Literature and interpretation

This extension transfers established policies; its three-epoch schedule and
model-size comparison are study choices rather than
exact replications of the cited experiments.

The estimand is the effect of using the pretrained Whisper Medium variant
instead of Base under matched fine-tuning conditions. It is not a causal
isolation of parameter count from all differences in architecture and
pretrained weights. FLEURS comparisons must later use the same corrected v2
data, decoding, normalization, and prompt-clustered analysis as the Base runs.

- Radford et al. (2023), Robust Speech Recognition via Large-Scale Weak
  Supervision: https://proceedings.mlr.press/v202/radford23a.html
- Amodei et al. (2016), Deep Speech 2 (SortaGrad):
  https://proceedings.mlr.press/v48/amodei16.html
- Park et al. (2019), SpecAugment (LB/LD policies):
  https://doi.org/10.21437/Interspeech.2019-2680

The exact waveform recipe and its adaptations remain in
`rq2_waveform_augmentation_policy.md`; this queue does not redesign that recipe.