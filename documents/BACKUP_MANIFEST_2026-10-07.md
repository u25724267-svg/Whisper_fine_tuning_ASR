# ASR Qhivi Incremental Backup — 2026-10-07

Destination: private SMB share `//qhivi.up.ac.za/home`, under `casper/ASR`.
This increment used additive `rclone copy` operations through the SMB backend.
No destination deletion was requested.
Verification completed at `2026-10-07T16:03:21Z`.

| Local source | Destination under `casper/ASR` | Regular files | Logical bytes | Verification |
|---|---|---:|---:|---|
| `/ext_data/casper/asr_experiment_outputs` | `ext_data/casper/asr_experiment_outputs` | 3,162 | 269,540,473,077 | Full content read back passed |
| `/ext_data/casper/whisper_runs` | `ext_data/casper/whisper_runs` | 103 | 24,547,891,816 | Full content read back passed |
| Current repository snapshot | `repository/Whisper_fine_tuning_ASR_2026-10-07` | 2,874 | 598,158,686 | Full content read back passed |

The experiment-output destination contained 1,588 files and 64,787,832,681
bytes before this increment. It now matches the local tree by file count and
size. A one-way `rclone check --download` compared every regular file's content
and exited successfully. The local tree contains 139 checkpoint directories,
including 72 created after the September 9 backup and 15 Medium checkpoints
created October 6–7.

The repository snapshot is from Git commit
`8eb4387650c5c970daa4a0b895b9f1db4816ae3f`. It contains source code,
Git history, documents, experiments, and research artifacts. The private
`.env`, virtual environment, historical `outputs/`, root `logs/` and `wandb/`,
Python caches, and bytecode were excluded. The historical `outputs/` files
were checked against the prior Qhivi archive and matched by path and size.
This manifest was written after the repository snapshot and is stored at the
backup root as `BACKUP_MANIFEST_2026-10-07.md`.
The `.env` remains backed up separately under
`secrets/Whisper_fine_tuning_ASR.env`; its SHA-256 still matches the local file.

The September-backed `/ext_data/casper/asr_data`,
`/ext_data/casper/whisper_data`, and
`/home/casper/Speech/data/waxal/sna_asr` trees passed fresh one-way path and
size checks. Their local regular-file modification times have not advanced
since September 9. The September manifest records full checksum checks for
these trees.

The SMB destination does not preserve Unix symlinks. The 190 W&B convenience
symlinks in the experiment-output tree were skipped; all regular files were
included. The regenerable Hugging Face cache and `.venv` remain excluded.

## Restore mapping

```text
ASR/ext_data/casper/asr_experiment_outputs -> /ext_data/casper/asr_experiment_outputs
ASR/ext_data/casper/whisper_runs           -> /ext_data/casper/whisper_runs
ASR/repository/Whisper_fine_tuning_ASR_2026-10-07
  -> /home/casper/asr_experiments/Whisper_fine_tuning_ASR
```

Restore the repository snapshot into a fresh directory or compare it with the
current Git checkout first. Its `.env` and virtual environment must be restored
or recreated separately.
