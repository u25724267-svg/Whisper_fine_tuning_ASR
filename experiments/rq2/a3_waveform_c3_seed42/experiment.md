# RQ2-A3: Waveform Augmentation with C3, Seed 42

## Purpose

Measure the interaction between the selected C3 curriculum and the frozen mild
waveform policy. A2 seed 42 is the matched augmented random-order control.

## Treatment

- Same immutable seed-42 augmented training audio as A2
- Frozen C3 strict order derived from clean duration and SNR metadata
- Expected order hash prefix: `86ff768edcc8`
- SpecAugment disabled
- Clean validation and test audio
- Model, optimizer, update budget, and checkpoint selection matched to A2

Using augmented waveform durations to recompute C3 would change the order to
`a36d3ab9f7c1` and confound the factorial. This run explicitly sources duration
and SNR ranks from the clean acoustic metadata.

## Status

Validated and ready on 2026-09-04. The full 13,807-row augmented-data dry run
passed and reproduced C3's clean-order hash `86ff768edcc8`. Not launched;
waiting for A2 seed-42 item evaluation and GPU availability.