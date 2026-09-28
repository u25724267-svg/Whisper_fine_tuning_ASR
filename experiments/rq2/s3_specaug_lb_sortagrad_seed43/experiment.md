# RQ2-S3: LB Masks Without Time Warping, Seed 44

## Purpose

Test the Park et al. LibriSpeech Basic masking parameters, excluding time
warping, under sortagrad ordering on speaker-disjoint WAXAL v2.

## Treatment

- Ordering: SortaGrad (duration epoch 1, seeded random epochs 2--3).
- Frequency masks: one, width sampled from 0--27 Mel bins.
- Time masks: one, width sampled from 0--min(100, valid frames).
- Application: every training presentation.
- Validation/test masking: disabled.
- Waveform augmentation: disabled.

## Matched control

`rq1-c1-sortagrad-seed43`. Model revision, data, seed, optimizer, update budget, batch size,
checkpointing, and decoding are unchanged. Only SpecAugment masking differs.

## Status

Configured on 2026-09-28 under the frozen
`rq2_specaugment_lb_no_warp_protocol.md`. Training has not yet completed.
