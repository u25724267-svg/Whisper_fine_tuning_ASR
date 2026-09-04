# RQ2 Waveform Augmentation Policy

## Frozen asset

- Source: OpenSLR SLR28, Room Impulse Response and Noise Database
- License: Apache 2.0
- Archive: `/ext_data/casper/asr_data/augmentation/slr28/rirs_noises.zip`
- Archive SHA-256: `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7`
- Extracted point-source noise WAV files: 843
- Extracted simulated RIR WAV files: 60,000
- Audio format: 16 kHz, 16-bit WAV in the source archive

Only SLR28 point-source noises and simulated RIRs are used. Real RIR subsets and
external MUSAN files are excluded from the primary policy.

## Frozen mild policy

- Additive noise probability: 0.5
- Target SNR: continuous uniform distribution from 10 to 25 dB
- Simulated RIR probability: 0.3
- Speed perturbation probability: 0.5
- Non-unit speed factors: 0.9 and 1.1 with equal probability
- Validation, test, and FLEURS audio: unmodified
- SpecAugment: disabled in every factorial cell

Transform decisions, asset paths, target SNR, speed factor, gain correction,
global seed, and derived per-utterance seed must be recorded for every training
row. Asset lists are sorted before seeded selection. The composition order is
speed perturbation, RIR convolution, additive noise, then anti-clipping gain.

## Materialization gate

Before A2 or A3 starts, a deterministic preprocessor must:

1. Produce an immutable seed-specific training manifest and audio directory.
2. Preserve utterance IDs and transcripts exactly.
3. Record actual output duration and every transform parameter.
4. Verify all audio is finite, mono, and 16 kHz with no clipping.
5. Reproduce byte-identical manifests and parameter records from the same seed.
6. Hash the source manifest, policy, archive, selected asset inventory, output
   manifest, and parameter record.

The first new RQ2 experiment is A2 seed 42: no curriculum with this waveform
policy. A0 reuses C0 and A1 reuses C3. A3 is not launched until the A2 pipeline
passes full-data audit.

## Seed-42 materialization

The full seed-42 data gate passed on 2026-09-03:

- Rows: 13,807
- Speed applied: 6,976
- Simulated RIR applied: 4,216
- Additive noise applied: 6,983
- No transform selected: 2,370
- Anti-clipping attenuation applied: 1,267
- Training manifest SHA-256: `3f0260f99540ffcb85705593ffbd294d522ed6c286a755022a61dbb4476687a3`
- Parameter audit SHA-256: `787e4bad25c5384f5a6c7d61a6e8caaf191161a6e3350d59b5905441d4e5aa21`

Every ID and transcript matches the clean source manifest. All output paths are
unique and readable, and all files are finite mono 16 kHz FLAC with exact
manifest durations. A staging-path defect found by the 64-row validation was
fixed before full materialization; failed test artifacts were retained under
`.broken_staging_paths` names and were not used for training.

The full manifest was then schema-repaired to remove the train-only
`augmentation_seed` column because Hugging Face JSON split loading requires
identical columns across train, validation, and test. Seed provenance remains in
every parameter row; no ID, transcript, duration, or audio path changed.