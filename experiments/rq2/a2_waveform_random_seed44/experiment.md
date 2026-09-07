# RQ2-A2: Waveform Augmentation, Random Training, Seed 44

## Purpose

Measure the waveform-augmentation main effect against the matched C0 control.

## Contract

- Control: C0 seed 44
- Materialization seed: 44
- Standard seeded random order
- Frozen waveform policy `rq2-waveform-mild-v1`
- SpecAugment disabled
- Clean validation and test audio
- Training manifest SHA-256: `b5873246e54e161fb9232073fd0a686d1c37b39cf48bff85e30fb0c341a35a46`
- Parameter audit SHA-256: `3166d6e54ec390bf1b37392a254ce5592e610aa82d1ac7040f1639fac83df5d0`
- Hashed item-level validation/test predictions required

## Status

Complete. Item-level validation/test WER is 32.7943/31.1484. The matched
validation improvement over C0 seed 44 is -0.0372 points with speaker-clustered
95% interval [-1.1437, 1.1340].
