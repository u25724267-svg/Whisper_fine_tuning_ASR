# Experiment Prediction Artifacts

This directory mirrors lightweight evaluation artifacts from
`/ext_data/casper/asr_experiment_outputs` while preserving each RQ1/RQ2 run's
relative path.

Included artifact families:

- `item_predictions/` validation/test predictions and summaries;
- `fleurs_corrected_v2/` corrected FLEURS predictions and summaries;
- curriculum order and score audits;
- root-level train/validation/test result, Trainer-state, experiment-config, and
	run-manifest JSON files;
- RQ1/RQ2 aggregate, comparison, and FLEURS aggregate reports; and
- centralized per-run logs.

Model weights, optimizer states, checkpoints, tokenizers, feature caches, and
training-argument binaries remain on external storage. Generated JSONL rows and
logs are ignored by Git because they are comparatively large; hash-bearing
JSON/Markdown summaries remain available for version control and analysis.

The external files remain authoritative. Re-copy artifacts only from immutable
completed output directories, and verify summary SHA-256 fields before analysis.