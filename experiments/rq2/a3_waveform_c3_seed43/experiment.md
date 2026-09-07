# RQ2-A3: Waveform Augmentation with C3, Seed 43

## Purpose

Measure the C3-by-waveform interaction against the matched A2 condition.

## Contract

- Control: A2 seed 43
- Materialization seed: 43
- Frozen clean-data C3 order `86ff768edcc8`
- Frozen waveform policy `rq2-waveform-mild-v1`
- SpecAugment disabled
- Clean validation and test audio
- Training manifest SHA-256: `f69965ec6b870c17f1388f8deae62b5c28b3f32ebb33e9249c2310c0fcd3cbfe`
- Parameter audit SHA-256: `1349b617532a5d522f6b51c8bb20f7c0e221cf96548e292323466297f25617ff`
- Hashed item-level validation/test predictions required

## Status

Complete. Item-level validation/test WER is 33.3519/31.1005. The matched
validation improvement over A2 seed 43 is -0.0797 points with speaker-clustered
95% interval [-0.4864, 0.2471].
