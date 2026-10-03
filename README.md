# Shona Whisper ASR Experiments

Reproducible Whisper fine-tuning, curriculum learning, augmentation, external
evaluation, and pseudo-labeling experiments for Shona ASR.

## Repository layout

```text
asr.py                  Single Python command entry point
asr_experiments/        Shared config, provenance, audio, training, and RQ4 code
  commands/             Training, evaluation, analysis, data, and RQ4 commands
configs/                Reusable and historical experiment configurations
experiments/            Immutable RQ1/RQ2 experiment definitions and records
experiment_core/        Curriculum samplers and dynamic curriculum state
scripts/                Guarded launchers and multi-stage workflows
tests/                  Standard-library unittest suite
documents/data/         Frozen scientific protocols and data provenance
documents/thesis/       Methodology and results drafts
artifacts/experiment_outputs/
                        Lightweight predictions, metrics, audits, and logs
```

Large models, checkpoints, optimizer state, prepared audio, and caches remain
under `/ext_data/casper`. Lightweight scientific outputs are mirrored into the
repository under `artifacts/experiment_outputs` with SHA-256 manifests.
Historical pre-RQ experiment outputs remain in the Git-ignored local
`outputs/` directory.

## Environment

The project uses Python 3.10 and the checked-in `.venv` convention:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
cp .env.example .env
```

Add the W&B API key only to the ignored `.env`. All tracked experiment configs
use the existing `whisper-shona-multilingual` project.

## Running experiments

Each training experiment lives under `experiments/rq1` or `experiments/rq2` and
contains `config.json`, `experiment.md`, and a compatibility `run.sh`.

Validate the runner named by a config before training:

```bash
.venv/bin/python asr.py train_full \
  --config experiments/rq1/c0_random_seed42_v2/config.json \
  --dry-run
```

Launch one experiment through the shared guarded launcher:

```bash
./scripts/run_experiment.sh experiments/rq1/c0_random_seed42_v2
```

Run implemented experiments sequentially, export item predictions, and mirror
their lightweight outputs:

```bash
./scripts/run_sequence.sh \
  experiments/rq2/s2_specaug_lb_random_seed42 \
  experiments/rq2/s3_specaug_lb_sortagrad_seed42
```

The supported config-selected runners are:

- `train_full.py`: standard shuffled training and SpecAugment;
- `train_sortagrad.py`: SortaGrad ordering;
- `train_snr_curriculum.py`: static acoustic curricula and pacing; and
- `train_s2s_curriculum.py`: dynamic loss, WER-margin, and hybrid curricula.

Launchers refuse unsafe output reuse, enforce configured disk/GPU thresholds,
and retain tmux panes and logs for diagnosis.

### Medium RQ1/RQ2 queue

The prepared 27-run Medium extension is listed in
[experiments/medium/queue.json](experiments/medium/queue.json). It covers seeds
42--44 for clean random, SortaGrad, and C3 controls, waveform A2/A3, and LB/LD
S2--S5 without time warping. Setup does not launch training.

Validate with `./scripts/run_medium_rq1_rq2.sh --preflight-only`. To explicitly
start the detached sequential queue later, run `./scripts/run_medium_rq1_rq2.sh`.
See [experiments/medium/README.md](experiments/medium/README.md) for storage,
resource requirements, and the validation contract.

## Evaluation and analysis

Export validation/test or named-manifest predictions:

```bash
.venv/bin/python asr.py evaluate_predictions \
  --config experiments/rq1/c0_random_seed42_v2/config.json \
  --model-dir /ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2 \
  --output-dir /tmp/predictions \
  --splits validation test
```

`asr.py compare_predictions_bootstrap` performs paired cluster bootstrap
comparisons. `asr.py compare_predictions_factorial_bootstrap` handles four-cell
factorials. FLEURS comparisons must use the corrected manifest as the cluster
manifest with `source_id` as the cluster field. Run `asr.py --help` to list all
registered commands.

## Output locations

```text
/ext_data/casper/asr_experiment_outputs/<rq>/<experiment>/
  model.safetensors, checkpoint-*/, optimizer state, tokenizer, W&B runtime

outputs/output_dir_*/
  historical local models and checkpoints retained outside Git

artifacts/experiment_outputs/<rq>/<experiment>/
  result JSON, item predictions, FLEURS predictions, curriculum audits,
  logs, run manifests, and artifact_manifest.json
```

Prepared protocols and materialized audio live under
`/ext_data/casper/asr_data`. Generated JSONL prediction rows and logs are stored
inside the repository tree but ignored by Git; hash-bearing summaries remain
available for version control.

## Tests and diagnostics

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
find scripts experiments -name '*.sh' -type f -exec bash -n {} \;
```

For a failed queued run:

1. Inspect `tmux list-sessions` and the retained pane.
2. Read `<external-run>/logs/train.log` or the workflow sequence log.
3. Verify the expected result and prediction summaries exist.
4. Run the exact config with `--dry-run` before creating a new immutable run.

Incomplete external output directories are never overwritten automatically.

## Legacy workflow

The `legacy/` directory retains `prepare_data.py`, `train.py`, `decode.py`,
`whisper_transcribe_WER.py`, `pilot_train.py`, and `slurm_run.sh` for historical
provenance. They predate the current WAXAL config-driven system, contain
machine-specific assumptions, and are not the supported workflow. In
particular, `legacy/decode.py` has a known undefined `output_file` reference.
They will not be deleted or repaired without an explicit archival decision.

See [documents/architecture.md](documents/architecture.md) for control flow and
module ownership, and [experiments/README.md](experiments/README.md) for the
confirmatory experiment contract.