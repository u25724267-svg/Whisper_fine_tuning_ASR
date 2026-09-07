# RQ4 Teacher Inference Pilot v1

## Decision

Batch size 4 is frozen for full C0 teacher inference. The pilot passed source
identity, deterministic transcript, token alignment, duration coverage, memory,
and resumability gates. No filtering threshold was selected.

## Final stratified pilot

- Output: `/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1_pilot32_final`.
- Rows: 32 across 22 speakers.
- Duration coverage: approximately 1.008--30.0 seconds.
- Empty outputs: 0.
- Maximum-length outputs: 0.
- Median geometric content-token confidence: 0.89347.
- Median raw no-speech-token probability: $2.13\times10^{-8}$.
- Prediction SHA-256: `42de43f484909492e7afda87cac772ff7c3aedf8b1dffb937c4d4fbb77f88a9e`.
- Peak CUDA memory reserved: 1,073,741,824 bytes.
- Reserved-memory headroom: 95.75% on the RTX 4090.

The raw no-speech score remains uncalibrated and is not a filtering threshold.

## Determinism and batching

Two batch-4 runs over the same 16 rows produced byte-identical prediction
files. Batch 8 produced identical decoded text but different post-EOS padding
lengths and floating-point scores. Serialization now trims at the first EOS.
Batch 4 is retained because it is exactly repeatable and has ample headroom;
throughput is secondary to stable pseudo-label evidence.

## Resume test

Only the final pilot's provenance, chunk JSONL, and completion marker were copied
to `/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1_resume_test`. A new
invocation verified the existing chunk and consolidated it without re-inference.
The resulting prediction SHA-256 exactly matched the source pilot.

## Superseded pilot directories

- `rq4_teacher_labels_c0_v1_pilot16`: stopped before inference because a named
  no-speech token was unavailable.
- `rq4_teacher_labels_c0_v1_pilot16_v2`: stopped on deterministic CuBLAS setup.
- Earlier completed pilots predate final EOS normalization or resource
  accounting and are retained only as implementation evidence.

Transformers 4.46.3 derives the no-speech token as
`no_timestamps_token_id - 1`; the production implementation follows that
library definition and labels the result as uncalibrated.