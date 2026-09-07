# Confirmatory Experiments

Each experiment has a complete configuration, a one-command launcher, and an
experiment record. Shared launch behavior lives in `scripts/run_experiment.sh`.

All new outputs are stored under:

```text
/ext_data/casper/asr_experiment_outputs/<research-question>/<experiment-id>
```

Launch an experiment from its directory with `./run.sh`. The launcher refuses
to overwrite or resume an existing output directory. Historical outputs remain
in their original locations.

## GPU concurrency

Whisper Base workloads may run in pairs on the 24 GB RTX 4090. Launch them
sequentially so the second preflight observes the first process's allocation,
and require at least 10,000 MiB free before admitting the second workload. Do
not admit a third CUDA workload. On 2026-09-02, concurrent C4R training and C4
prediction export used 6,757 MiB combined at 99% GPU utilization, leaving
17,327 MiB free.

Treat this as a Base-specific measured policy. Re-measure before pairing Medium
or Large models, larger batches, or augmentation settings that increase memory.

## Sequential orchestration

`scripts/run_sequence.sh` runs only experiment directories that are already
implemented. Every requested directory must contain an executable `run.sh` and
a valid `config.json`. The script preflights the complete list before launching
the first stage, so a missing future experiment aborts with no launches.

Example:

```bash
./scripts/run_sequence.sh \
	experiments/rq2/a2_waveform_random_seed42 \
	experiments/rq2/a3_waveform_c3_seed42
```

For each implemented stage, the script:

1. Skips training only when the model and train/validation/test result files all
	 exist.
2. Otherwise launches or waits for the experiment's retained tmux pane.
3. Stops immediately on a nonzero pane status or incomplete output.
4. Exports validation/test item predictions when `summary.json` is absent.
5. Stops if the prediction export fails or omits its summary.
6. Continues to the next experiment only after the current stage is closed.

It does not implement missing runners, materialize augmented datasets, generate
configs, run bootstrap analysis, create corrupted evaluation sets, evaluate
FLEURS, or make scientific decisions. Those dependencies must be implemented,
validated, and represented as experiment directories before being added to a
sequence. Partial output directories are never overwritten automatically.

For the fully specified RQ2 primary matrix, use
`scripts/run_rq2_primary.sh`. It materializes seeds 43/44, generates hash-pinned
A2/A3 directories, dry-runs them, delegates sequential training and prediction
export to `run_sequence.sh`, writes matched bootstrap comparisons, and produces
a descriptive factorial summary. It then stops for scientific interpretation.
It never opens FLEURS or launches conditional SpecAug, Medium, RQ3, or RQ4 work.