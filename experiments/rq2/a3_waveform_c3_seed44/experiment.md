# RQ2-A3: Waveform Augmentation with C3, Seed 44

## Purpose

Measure the C3-by-waveform interaction against the matched A2 condition.

## Contract

- Control: A2 seed 44
- Materialization seed: 44
- Frozen clean-data C3 order `86ff768edcc8`
- Frozen waveform policy `rq2-waveform-mild-v1`
- SpecAugment disabled
- Clean validation and test audio
- Training manifest SHA-256: `b5873246e54e161fb9232073fd0a686d1c37b39cf48bff85e30fb0c341a35a46`
- Parameter audit SHA-256: `3166d6e54ec390bf1b37392a254ce5592e610aa82d1ac7040f1639fac83df5d0`
- Hashed item-level validation/test predictions required

## Status

Complete. Item-level validation/test WER is 32.6828/31.6460. The matched
validation improvement over A2 seed 44 is +0.1115 points with speaker-clustered
95% interval [-0.2616, 0.5727].
