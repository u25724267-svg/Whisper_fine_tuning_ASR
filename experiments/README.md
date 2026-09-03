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