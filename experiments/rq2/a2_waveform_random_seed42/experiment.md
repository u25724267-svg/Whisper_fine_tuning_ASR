# RQ2-A2: Waveform Augmentation, Random Training, Seed 42

## Purpose

Measure the main effect of the frozen mild waveform policy without curriculum,
using C0 seed 42 as the matched clean control.

## Treatment

- Seed-specific immutable augmented WAXAL training audio
- Noise probability 0.5 at 10--25 dB target SNR
- Simulated RIR probability 0.3
- Speed perturbation probability 0.5 using 0.9x or 1.1x
- Standard random training order
- SpecAugment disabled
- Clean validation and test audio

## Required gates

- 13,807 augmented training rows
- Byte-repeatable transform decisions and output audio
- Mono 16 kHz finite audio with verified durations and peak limits
- Complete transform parameter and hash audit
- Baseline-matched model, optimizer, updates, and checkpoint selection

## Status

Full seed-42 materialization and audit completed on 2026-09-03. All 13,807 IDs
and transcripts are unchanged; audio is finite mono 16 kHz with exact recorded
durations and verified peak limits. The training manifest SHA-256 is
`3f0260f99540ffcb85705593ffbd294d522ed6c286a755022a61dbb4476687a3`,
and the parameter audit SHA-256 is
`787e4bad25c5384f5a6c7d61a6e8caaf191161a6e3350d59b5905441d4e5aa21`.
The full training dry run passed after the augmented manifest was made
cross-split schema-compatible. A2 launched as the sole CUDA workload and
progressed beyond optimizer step 200 without data, runtime, or CUDA errors.

- W&B project: `whisper-shona-multilingual`
- W&B run: `e4dwoxn2`
- Output: `/ext_data/casper/asr_experiment_outputs/rq2/a2_waveform_random_seed42`
- Log: `/ext_data/casper/asr_experiment_outputs/rq2/a2_waveform_random_seed42/logs/train.log`

Hashed item-level validation/test export completed. Compared with C0 seed 42,
the test WER change was -0.2635 points with a speaker-clustered 95% interval of
[-1.2848, 0.5120]; this is practically unresolved and does not support a clean
WAXAL benefit from waveform augmentation alone.

Training and final Trainer evaluation completed successfully on 2026-09-04.
Hashed item-level prediction export is pending GPU availability.

- Validation WER: 33.1687
- Test WER: 31.5875