# RQ1 Final Curriculum Analysis

## Decision

C3, the strict joint SNR-duration curriculum, is selected for the RQ2
curriculum factor. C5 is rejected. This decision was frozen after all planned
seed-42 screening and seed-43/44 replication artifacts were complete and before
the RQ2 waveform policy or FLEURS test results were opened.

## Aggregate results

| Condition | Validation WER, mean +/- SD | Test WER, mean +/- SD | Mean test change vs C0 |
|---|---:|---:|---:|
| C0 random | 32.9058 +/- 0.4796 | 31.0446 +/- 0.2875 | reference |
| C3 SNR-duration | 32.8368 +/- 0.0000 | 30.4564 +/- 0.0000 | -0.5882 |
| C5 S2S-margin | 33.9069 +/- 0.2611 | 31.7827 +/- 0.7268 | +0.7381 |

Lower WER is better. C3 is deterministic under the fixed strict order and
zero-dropout model, so its seed-42 and seed-43 models and predictions are
byte-identical. The C3 seed-44 artifact was therefore not retrained; the verified
C3 prediction artifact is compared against each stochastic C0 seed. C3 test
improvements over C0 were 0.8995, 0.3327, and 0.5323 WER points for seeds 42,
43, and 44 respectively. Validation changes were -0.3186, +0.6054, and -0.0797
points, where positive means improvement.

C5 test changes were +0.0399, -1.8310, and -0.4232 points across seeds, where
positive means improvement. Its mean test effect was -0.7381 points and its mean
validation effect was -1.0010 points. The exact seed-42 rerun reproduced every
checkpoint hash and metric but did not add an independent observation.

## Uncertainty

Speaker-clustered paired bootstrap analyses used 10,000 replicates per split and
seed. C3 test intervals were [0.3135, 1.3897], [0.1201, 0.5223], and [0.0209,
1.2611] WER points for seeds 42--44. C5 did not show a consistent positive test
effect and was consistently worse on validation.

## RQ2 reuse

- A0, no curriculum and no waveform augmentation: reuse C0 seeds 42--44.
- A1, C3 curriculum and no waveform augmentation: reuse the verified
  deterministic C3 artifact, disclosing its absent stochastic seed dimension.
- A2, no curriculum plus waveform augmentation: three new runs, seeds 42--44.
- A3, C3 curriculum plus waveform augmentation: three new runs, seeds 42--44.

The waveform policy and external noise/RIR assets must be frozen and hashed
before A2 or A3 starts. FLEURS test evaluation remains locked until then.