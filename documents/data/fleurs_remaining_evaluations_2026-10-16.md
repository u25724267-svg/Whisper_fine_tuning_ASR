# Remaining Corrected FLEURS Evaluations

## Preparation Status

At initial preparation, no checkpoints were restored, evaluation launched,
data deleted, or training configuration changed. The later authorised launch
is recorded below. Immutable preparation inventory:
`experiments/fleurs_remaining_queue.json`.

The repository mirrors contain completed FLEURS summaries for 25 fine-tuned
Base runs, 11 original Medium augmentation runs, five new A100 Medium controls,
and the two pretrained zero-shot baselines. The five-control follow-up exited
successfully. Original archived models must not be mistaken for unfinished
training merely because their local weights were removed after verified backup.

## Primary Backlog

| Model | Condition | Missing Seeds | Runs |
|---|---|---|---|
| Medium | C1 SortaGrad | 44 | 1 |
| Medium | C3 SNR-duration | 42, 43, 44 | 3 |
| Medium | A2 waveform random | 42, 43, 44 | 3 |
| Medium | A3 waveform C3 | 42, 43, 44 | 3 |
| Medium | S2 SpecAugment LB random | 42 | 1 |
| Base | C2 SNR | 42, 43, 44 | 3 |
| Base | C3 SNR-duration | 43, 44 | 2 |
| Base | C4 acoustic cumulative | 42 | 1 |
| Base | C4r random pacing | 42 | 1 |
| Base | C5 S2S margin | 42, 43, 44 | 3 |
| Base | C6 WER margin | 42, 43, 44 | 3 |
| Base | C7 acoustic-S2S hybrid | 42 | 1 |

Total: **11 Medium + 14 Base = 25 primary evaluations**. Evaluate all listed
seeds, including deterministic repeats, but do not interpret identical runs as
independent accuracy replications. C4/C4r/C6/C7 are descriptive or exploratory
conditions; external evaluation does not promote them into primary causal claims.

The Medium cohort uses the five fresh A100 controls and original A100 C1 seed
44, as prespecified in `medium_a100_controls_2026-10-15.md`. The five historical
RTX C0/C1 models also lack FLEURS, but are separately listed rather than silently
pooled or substituted according to scores. Base's old non-disjoint/random
controls and exact C5 seed-42 rerun are also separate historical coverage tasks.
There are **eight additional historical/repeatability runs**, not 33 independent
primary experiments. The original C5 seed-42 run is retained in the main list;
its exact rerun is not an additional seed.

## Archive And Storage Gates

Known previously verified Medium archives:

- C1 seed 44: `gs://casper/completed-runs/2026-10-10/medium/rq1/c1_sortagrad_seed44/`.
- Other ten Medium runs: `gs://casper/completed-runs/2026-10-12/`, followed by the
  relative output paths in the queue.
- Base root-model objects are present under
  `gs://casper/qhivi-backup-2026-10-07/ext_data/casper/asr_experiment_outputs/`,
  followed by each relative output path and `/model.safetensors`. Repository
  artifact mirrors do not contain these weights.

The read-only cloud listing on 2026-10-16 failed because authentication needs
refreshing. After the user refreshed authentication, current read-only listings
confirmed all 11 Medium and 14 Base root-model objects at the exact queue paths.
Each Medium model is 3,055,544,304 bytes; each Base model is 290,403,936 bytes.
Total primary weights are 37,676,642,448 bytes, approximately **35.09 GiB**.
Representative Medium root listings include processor/tokenizer/generation and
provenance files, but all-run sidecar checks and selected-checkpoint identity
verification remain pending. Object presence/size is not downloaded checksum
verification or proof that the complete inference package is ready.

Approximately 223 GiB remained free after the refreshed inventory. Weight-only
payloads fit without deleting existing results; allow additional space for
sidecars, inference outputs, caches and temporary files. Restore only the
exported root best-model weights, processor/tokenizer/generation files and
required provenance, not three epoch checkpoints or optimizer/RNG state.
Measure actual cloud object bytes and check available space before admission;
do not use the old 32-GiB full-training-run reserve as an inference payload size.
Use two-model restore batches if necessary. Retain restored files unless cleanup
is separately approved; no automatic deletion or broad cloud sync is authorised.

Before inference, verify downloaded object checksums against cloud metadata,
saved experiment IDs/configs and selected-checkpoint provenance. Root exports
must be validation-WER-selected models, not automatically the last checkpoint.
Keep existing outputs untouched and stop for partial local restore or FLEURS
directories rather than overwriting them.

## Evaluation Contract And Launch Readiness

Use the existing `evaluate_predictions` command with `--model-dir` pointing to
each restored fine-tuned root export and its own original experiment config.
Pass only named `fleurs_validation` and `fleurs_test` manifests; omit `--splits`
to avoid accidentally adding WAXAL. Never use `--model-id` for these fine-tuned
models. The two zero-shot baselines are already complete and are not rerun.

Freeze corrected FLEURS validation/test at 393/925 recordings and the hashes in
the queue; confirm every audio path exists. Preserve batch six, FP16, Shona
transcription, greedy generation length 225, and cleared forced/suppressed
tokens. No FLEURS training split, checkpoint selection, tuning, or retraining.
Previously inspected FLEURS remains adaptive external evidence.

Use one independent model worker per idle physical GPU, UUID-pinned explicitly
inside detached processes. Require no active CUDA compute processes, not only
sufficient free memory. Evaluate validation and test, then verify prediction
hashes/row counts and refresh the lightweight repository mirror before advancing.
Existing complete validated results are skipped; partial results require review.

The existing batch-two launcher does not implement this archive queue: its
default mode covers the already-completed last eleven Medium runs, while
`--local-only` silently excludes archived weights. Do not launch either mode as
though it covers the backlog. After all-run package and provenance verification,
prepare a restore-aware launcher or explicit per-run commands against this queue,
then obtain the launch go-ahead. Preparation alone has not made cloud sources or
downloaded models ready for inference.

Expected inference-only time for Medium is roughly 1.5--2 hours across two GPUs,
based on the latest 14--16 minute runs and six jobs on the longer lane. Downloads,
checksum verification and Base evaluation are additional and not yet timed.

## Authorised Launch

After the user's explicit go-ahead, all 25 primary packages passed the live
archive preflight. For every run, required inference/provenance files are
present, scientific config blocks match the saved experiment, cloud trainer
state matches the repository record, and the exported root weight size and
CRC32C match the checkpoint selected by validation WER. Several Base C5/C6/C7
runs selected checkpoint 4604; these are not replaced by checkpoint 6906.
Total measured inference-package payload is **35.18 GiB**, including sidecars
and canonical WAXAL prediction provenance. The 20-GiB additional free-space
reserve, fixed FLEURS hashes/audio, two idle A100 GPUs and memory gates passed.

Launcher: `scripts/run_remaining_fleurs.py --launch`, using the project Python.
The detached controller started on 2026-10-16 at approximately 10:26 UTC:

```bash
tmux attach -t fleurs-remaining-20261016T102543Z
```

Run evidence is under
`artifacts/experiment_outputs/fleurs_remaining/20261016T102543Z/`:
`plan.json` pins every object generation, CRC32C, size, config hash and best
checkpoint; `controller.log` records restoration and per-lane transitions;
`status.json` records running/completed/failed state; `completed/` gains one
marker only after validated source predictions and refreshed mirror pass.
The preparation queue's unchanged `prepared_not_launched` status is a frozen
snapshot, not live runtime status; use these controller records instead.

Two independent UUID-pinned workers process 13/12 runs, starting with Medium
C1 seed 44 on GPU 0 and C3 seed 42 on GPU 1. Downloads go into exclusive
`.fleurs-restore` staging directories, are pinned to cloud object generations,
and receive local size/CRC32C and provenance checks before atomic promotion.
Each inference log is `logs/fleurs_corrected_v2.log` under the restored run.
Predictions must contain exactly the 393/925 corrected validation/test rows,
correct manifest/config hashes and valid JSONL hashes before mirror refresh.
Ten focused mocked tests and actual Python 3.10 SHA256/CRC32C/parser checks passed.

No epoch/optimizer checkpoints, retraining, historical eight-run expansion,
automatic cloud upload, or deletion is part of this launch. Restored packages
and partial failures are retained. Existing mirrors are copied forward by the
atomic refresh implementation; historical logs and curriculum artifacts remain.