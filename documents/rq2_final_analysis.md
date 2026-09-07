# RQ2 Primary Factorial Analysis

## Decision

The primary A0--A3 matrix is complete. The frozen
`rq2-waveform-mild-v1` augmentation policy is not promoted. Clean C3 remains
the selected supervised condition for downstream transfer and teacher-selection
work. The optional waveform-plus-SpecAugment and Medium confirmation runs are
not justified by the primary result.

This decision was frozen before corrected FLEURS or corrupted-WAXAL results
were opened.

## Validation results

| Cell | Description | WER mean +/- SD | CER mean +/- SD |
|---|---|---:|---:|
| A0 | Clean, random | 32.9058 +/- 0.4796 | 7.3151 +/- 0.2399 |
| A1 | Clean, C3 | 32.8368 | 7.1317 |
| A2 | Augmented, random | 33.0607 +/- 0.2436 | 8.0084 +/- 0.4690 |
| A3 | Augmented, C3 | 33.0351 +/- 0.3360 | 7.7169 +/- 0.1321 |

Positive effects mean lower error. Augmentation without curriculum changed WER
by -0.1549 points and CER by -0.6933 points. Augmentation with C3 changed WER
by -0.1983 points and CER by -0.5852 points. The WER interaction was -0.0434
points, far below the preregistered 0.5-point practical threshold and
inconsistent across seeds.

No validation matched-bootstrap contrast had a 95% interval excluding zero.
A2 versus matched A0 produced WER improvements of -0.5974, +0.1699, and
-0.0372 points for seeds 42--44. A3 versus A2 produced +0.0451, -0.0797, and
+0.1115 points. A3 versus clean C3 produced -0.2337, -0.5151, and +0.1540
points. The evidence therefore does not support an augmentation main effect.
No beneficial C3-by-augmentation interaction was observed, but the reported
difference-in-differences is descriptive and was not formally
cluster-bootstrapped as an interaction contrast.

## Test results

Test results are reported after the validation-based decision and do not alter
selection. Mean WER was 31.0446 for A0, 30.4564 for A1, 31.2974 for A2, and
31.3701 for A3. Augmentation worsened test WER by 0.2528 points without
curriculum and 0.9137 points relative to clean C3. The test interaction was
-0.6609 points.

## Interpretation limits

A0, A2, and A3 contain independently trained seeds 42--44. A1 reuses the one
selected C3 seed-42 model for all three aggregate positions; its zero displayed
SD is therefore reuse, not evidence of zero training variance. Effects involving
A1 are descriptive and must not be treated as three independent C3
replications. The direct A2-versus-A0 contrast is fully seed-matched, and the
A3-versus-A2 contrast is independently trained and matched at every seed.

The primary matrix answers the promotion question on clean speaker-disjoint
WAXAL. Corrected FLEURS and corrupted-WAXAL remain separate frozen-policy OOD
diagnostics. They may characterize robustness, but they cannot retroactively
change the augmentation policy or primary promotion decision.

## Corrected FLEURS diagnostic

Corrected FLEURS inference was opened only after the primary decision was
frozen. All 10 distinct A0--A3 checkpoints completed validation and test
inference. Mean FLEURS test WER was 58.3099 for A0, 59.2294 for A1, 59.4879 for
A2, and 60.1009 for A3. Augmentation changed test WER by -1.1780 points without
curriculum and -0.8715 points relative to clean C3. The test interaction was
+0.3065 points with SD 1.0393. On FLEURS validation, augmentation without
curriculum was effectively neutral in WER (-0.0102 points), while its CER effect
was +0.8464 points. These OOD results do not rescue the augmentation policy.

The same A1 reuse limitation applies to this descriptive aggregate: the one C3
checkpoint is repeated for alignment and is not three independent runs.

## Artifacts

- Aggregate JSON: `/ext_data/casper/asr_experiment_outputs/rq2/aggregate/rq2_factorial_summary.json`
- Aggregate report: `/ext_data/casper/asr_experiment_outputs/rq2/aggregate/rq2_factorial_summary.md`
- Corrected FLEURS aggregate: `/ext_data/casper/asr_experiment_outputs/rq2/fleurs_aggregate/rq2_fleurs_factorial_summary.json`
- Corrected FLEURS report: `/ext_data/casper/asr_experiment_outputs/rq2/fleurs_aggregate/rq2_fleurs_factorial_summary.md`
- Matched bootstraps: `/ext_data/casper/asr_experiment_outputs/rq2/comparisons`
- Sequence log: `/ext_data/casper/asr_experiment_outputs/rq2/primary_sequence.log`

The aggregate records SHA-256 hashes for all 12 source summary positions.
All 18 matched-bootstrap files are present: three contrasts by three seeds by
two splits.