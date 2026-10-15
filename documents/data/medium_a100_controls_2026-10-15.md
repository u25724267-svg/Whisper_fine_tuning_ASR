# Medium A100 Hardware-Matched Control Replications

## Decision Before New Outcomes

The original 27-run Medium matrix mixed RTX 4090 training for C0 seeds 42--44
and C1 seeds 42/43 with later A100 training. C1 seed 43 finished all original
updates on RTX; its subsequent A100 checkpoint recovery was evaluation
completion, not fresh A100 training. The latest manifest's A100 device must
not be mistaken for its original training hardware.

Prepare five fresh replications on this machine's NVIDIA A100-SXM4-40GB GPUs:

| Condition | Seeds | Runner |
|---|---|---|
| C0 random | 42, 43, 44 | `train_full` |
| C1 SortaGrad | 42, 43 | `train_sortagrad` |

Existing C1 seed 44 already trained on this A100 machine and is reused rather
than retrained. Its archived weights remain at
`gs://casper/completed-runs/2026-10-10/medium/rq1/c1_sortagrad_seed44/`.
No restore is performed during queue preparation.

## Fixed Contract and Preservation

Each new config copies its original Medium config exactly except operational
IDs, output paths, control references, W&B metadata, GPU requirements, and
explicit replication provenance with the source SHA-256. IDs start with
`a100-`; configs and outputs use the separate `medium_a100` namespace.
The original 27 configs, outputs, predictions, and RTX results are not replaced.

Every run starts from pinned `openai/whisper-medium` revision
`abdf7c39ab9d0397620ccaea8974cc764cd0953e`, not an RTX checkpoint. No optimiser,
scheduler, or RNG state is resumed. Keep all original manifests, text/audio
processing, seed, ordering, augmentation, train/eval batch six, accumulation one,
three epochs, learning rate `1e-5`, 500 warm-up steps, FP16, determinism,
non-reentrant gradient checkpointing, validation-WER selection, three retained
checkpoints, and greedy Shona generation limit 225. Use the existing W&B project
`whisper-shona-multilingual` with fresh continuation-independent run IDs.

The queue uses two independent single-GPU lanes, not distributed training.
GPU 0 takes C0 seed 42, C0 seed 44, then C1 seed 43. GPU 1 takes C0 seed 43,
then C1 seed 42. Both the queue and individual launcher require the actual A100
device and no other active CUDA compute process on the assigned GPU. This
prevents overlap with the remaining FLEURS evaluation, even if memory is free.
An `.env` CUDA device override is refused rather than silently redirecting a run.

## Cohort Rule

The hardware-matched Medium analysis uses the new A100 C0 seeds 42--44 and new
A100 C1 seeds 42/43 plus existing A100 C1 seed 44. All other Medium conditions
retain their original A100 runs. C1 seed 44 is paired with the new A100 C0 seed
44 through the analysis mapping, without editing its historical config.

This rule is defined by hardware provenance before the new outcomes, not by
which execution has lower WER. Do not choose the better of RTX/A100 runs or
pool them as six independent seeds. Keep the historical mixed-hardware matrix
and paired same-seed RTX/A100 comparisons separate. This addresses the
within-Medium hardware-control mismatch; it does not by itself make the
RTX-trained Base versus A100-trained Medium comparison hardware-controlled,
create an untouched test population, or resolve source/version and numerical
differences. Record actual software, trainer hashes, device, and selected model.

No settings are selected on WAXAL test or FLEURS. Checkpoints remain
validation-selected, canonical item predictions remain authoritative, and
FLEURS evaluation is a later fixed-policy stage, not implicit retraining.

## Prepare and Launch Gates

```bash
.venv/bin/python asr.py prepare_medium_queue --a100-controls
.venv/bin/python asr.py prepare_medium_queue --a100-controls --check
.venv/bin/python asr.py prepare_medium_queue --a100-controls --dry-run-runners
bash scripts/run_medium_rq1_rq2.sh --a100-controls --preflight-only
```

The five-run admission reserve is **160 GiB**, unchanged at 32 GiB per run.
Only about 52 GiB was free during preparation, so training was not launched then.
Back up and verify current completed outputs, including their new FLEURS
exports, before any separately authorised local cleanup. No automatic deletion
or backup-triggered training is configured.

After sufficient space and both idle A100 GPUs are confirmed, launch explicitly:

```bash
bash scripts/run_medium_rq1_rq2.sh --a100-controls
```

Controllers are `whisper-medium-a100-controls-gpu0` and
`whisper-medium-a100-controls-gpu1`. Each sequence
performs training, canonical WAXAL validation/test export, and lightweight
mirroring before advancing. Partial outputs remain protected against overwrite.

## Authorised Cleanup and Launch

The batch-two backup completed with exit 0 and eleven exact verification
markers at `gs://casper/completed-runs/2026-10-15/medium/rq2/`.
Logs are under
`artifacts/experiment_outputs/medium/logs/backup-batch2-20261015T071130Z/`.
Before cleanup, each source was checked for changes after its verification
log, and FLEURS source/mirror prediction hashes were checked again.

With explicit user approval, only those eleven backed-up local run directories
were removed. Repository WAXAL/FLEURS mirrors, active data, and cloud copies
were retained. About 365 GiB was then free and the 160 GiB admission passed.

The five-run queue launched on 2026-10-15 at 08:12:21 UTC. Controllers
`whisper-medium-a100-controls-gpu0` and `whisper-medium-a100-controls-gpu1`
started fresh C0 seeds 42 and 43 on their separate GPU UUIDs. Both processes
authenticated to the existing W&B project and progressed through preprocessing.
Logs are in `artifacts/experiment_outputs/medium_a100/logs/` as
`sequence-gpu0-20261015T081221Z.log` and `sequence-gpu1-20261015T081221Z.log`.
All scientific settings, including generation length 225, remain unchanged.
The queue's immutable `training_launched` field describes preparation, not
current runtime; session status and completed artifacts are authoritative.

## Queued FLEURS Follow-up

At 08:29:49 UTC on 2026-10-15, FLEURS evaluation for the five fresh controls
was queued in detached session `medium-a100-fleurs-after-controls` using:

```bash
bash scripts/run_medium_fleurs_after_batch2.sh --a100-controls
```

The evaluator waits for both A100 training controllers to exit successfully,
then requires all five exported root best models, WAXAL prediction summaries,
and repository mirrors. A failed/disappearing training controller or partial
FLEURS output stops the follow-up instead of producing incomplete results.
Both A100 GPUs must be idle and meet the memory gate before inference starts.

The fixed corrected FLEURS validation/test manifests (393/925 rows), Shona
transcription, greedy generation length 225, FP16 and batch size six are reused.
Evaluation uses three jobs on GPU 0 and two on GPU 1, independently pinned by
UUID, with per-run exports and refreshed hash-verified mirrors. No additional
training or W&B run is created; completed FLEURS exports are skipped.

The controller log is
`artifacts/experiment_outputs/medium_a100/logs/medium-a100-fleurs-after-controls-20261015T082949Z.log`.
The reused original A100 C1 seed-44 model is archived and is not automatically
restored or evaluated by this five-model follow-up. It remains a separate
later coverage task, not a reason to pool incomplete three-seed external means.