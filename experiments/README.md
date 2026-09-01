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