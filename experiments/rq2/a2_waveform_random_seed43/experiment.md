# RQ2-A2: Waveform Augmentation, Random Training, Seed 43

## Purpose

Measure the waveform-augmentation main effect against the matched C0 control.

## Contract

- Control: C0 seed 43
- Materialization seed: 43
- Standard seeded random order
- Frozen waveform policy `rq2-waveform-mild-v1`
- SpecAugment disabled
- Clean validation and test audio
- Training manifest SHA-256: `f69965ec6b870c17f1388f8deae62b5c28b3f32ebb33e9249c2310c0fcd3cbfe`
- Parameter audit SHA-256: `1349b617532a5d522f6b51c8bb20f7c0e221cf96548e292323466297f25617ff`
- Hashed item-level validation/test predictions required

## Status

Complete. Item-level validation/test WER is 33.2723/31.1244. The matched
validation improvement over C0 seed 43 is +0.1699 points with speaker-clustered
95% interval [-1.3856, 2.0301].
