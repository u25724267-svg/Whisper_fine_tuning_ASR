# WAXAL Speaker-Disjoint v2 Acoustic Metadata

## Purpose

Provide immutable, ID-aligned acoustic measurements for C2, C3, C4, and C7
without changing or filtering the speaker-disjoint v2 manifests.

## Method

Audio is decoded at its native 48 kHz sample rate in 100 ms frames. A frame is
active when its RMS is within 40 dB of the utterance peak-frame RMS. The proxy is

```text
speech RMS = 75th percentile RMS among active frames
noise RMS  = 20th percentile RMS among all frames
SNR proxy  = min(60, dBFS(speech RMS) - dBFS(noise RMS))
```

This reproduces the earlier WAXAL audit definition. Training independently
resamples audio to 16 kHz through the Hugging Face `Audio` feature.

## Outputs

- Metadata: `/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/acoustic_metadata.csv`
- Summary: `/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/acoustic_metadata_summary.json`
- Metadata SHA-256: `8e0e82cd604af6f6c899f64736082d2b36c44250397c85377d74f1b6e00cf5f5`

## Validation

- Protocol rows: 17,193
- Training rows: 13,807
- Native sample rate: 48 kHz for all rows
- Channels: mono for all rows
- Existing same-path PCM hashes verified: 16,663
- Same-path hash mismatches: 0
- Silence-trimmed path replacements with newly computed hashes: 530
- Maximum duration discrepancy: 2.75 ms
- Missing or non-finite SNR values: 0

## Training distribution

| Statistic | SNR proxy, dB | Speech ratio | Duration, s |
|---|---:|---:|---:|
| Minimum | 7.68 | 0.226 | 3.62 |
| 25th percentile | 20.22 | 0.708 | 17.50 |
| Median | 33.38 | 0.793 | 19.85 |
| 75th percentile | 53.91 | 0.858 | 23.38 |
| Maximum | 60.00 | 0.992 | 35.90 |

The SNR proxy reaches its 60 dB cap for 2,665 training utterances (19.3%). C2
retains this ceiling and resolves equal-score rows by utterance ID rather than
tuning the proxy after observing results.