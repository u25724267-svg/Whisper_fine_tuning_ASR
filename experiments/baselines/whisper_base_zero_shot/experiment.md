# Whisper Base Zero-Shot WAXAL and FLEURS Baseline

## Purpose

Measure the pinned, pretrained `openai/whisper-base` checkpoint without any
fine-tuning on the speaker-disjoint WAXAL v2 validation/test sets and corrected
FLEURS v2 validation/test sets.

## Fixed inference contract

- Revision: `e37978b90ca9030d5170a5c07aadb050351a65bb`.
- Language/task prompt: Shona transcription.
- Greedy decoding with maximum generation length 225.
- Forced decoder IDs and suppress tokens cleared, matching controlled runs.
- WAXAL manifests: speaker-disjoint v2.
- FLEURS manifests: corrected immutable v2.
- No training, adaptation, checkpoint selection, or test-based tuning.

The four datasets are evaluated in one process so they share the exact model,
processor, and decoding configuration. Item-level predictions, aggregate
WER/CER/error counts, manifest hashes, prediction hashes, model ID, and model
revision are retained.

Scientific outputs are written directly to
`artifacts/experiment_outputs/baselines/whisper_base_zero_shot_waxal_fleurs_v1`.