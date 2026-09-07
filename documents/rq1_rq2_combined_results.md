# Combined RQ1 and RQ2 Results

## Scope

This analysis covers the completed Shona curriculum screen and replications
(RQ1), the clean-by-waveform-augmentation factorial (RQ2), and corrected FLEURS
out-of-domain evaluation. Positive effects below mean lower WER for the named
treatment relative to its control. All reported changes are absolute WER
percentage points.

## Main findings

1. Most curriculum strategies did not improve validation WER. The unadjusted
  speaker-bootstrap intervals for C2 and C4--C7 excluded zero in the harmful
  direction on seed-42 validation; the corresponding C4 and C4R test intervals
  also excluded zero. These screen-level intervals are not multiplicity-adjusted.
2. C3, strict joint SNR-duration ordering, produced a consistent WAXAL test
  ranking advantage against three random-control shuffles, averaging +0.5882
  points. C3 is one deterministic model rather than three independent training
  replications. Its validation effect was small on average (+0.0690) and
  inconsistent by seed.
3. C3 did not transfer its WAXAL advantage to corrected FLEURS. Its FLEURS test
   point estimates versus matched C0 were -1.5200, -0.3440, and -0.8945 points;
   all prompt-clustered 95% intervals included zero.
4. The frozen waveform policy did not improve clean WAXAL validation WER, clean
  WAXAL test WER, or corrected FLEURS test WER. No beneficial
  curriculum-by-augmentation interaction was observed; the interaction was
  descriptive rather than formally cluster-bootstrapped.
5. Dynamic WER scoring was computationally expensive without an accuracy gain:
   C6 required 14,062.7 seconds (3.91 hours), approximately 6.4 times the C0
   seed-42 runtime of 2,200.2 seconds (36.7 minutes).

## RQ1 curriculum screen

The seed-42 paired speaker-bootstrap screen gave the following effects against
C0-v2.

| Condition | Validation effect [95% CI] | Test effect [95% CI] | Interpretation |
|---|---:|---:|---|
| C1 duration SortaGrad | -0.0690 [-0.6422, 0.2726] | +0.3407 [-0.5471, 0.9976] | Unresolved |
| C2 strict SNR | -0.6001 [-1.0738, -0.1934] | -0.1783 [-1.5829, 1.0127] | Validation degradation |
| C3 SNR-duration | -0.3186 [-0.7530, -0.0232] | +0.8995 [0.3135, 1.3897] | Split-dependent |
| C4 cumulative tiers | -1.9516 [-2.7561, -1.2665] | -1.2295 [-1.7750, -0.7388] | Harmful |
| C4R random pacing | -1.8746 [-2.7805, -1.1709] | -1.1045 [-1.6084, -0.6670] | Harmful |
| C5 S2S margin | -1.3170 [-2.1487, -0.5906] | +0.0399 [-0.9377, 0.9178] | Validation degradation |
| C6 WER margin | -1.5135 [-2.1838, -0.9526] | -0.4152 [-1.4054, 0.5231] | Costly; no gain |
| C7 acoustic-S2S hybrid | -2.0047 [-3.2909, -1.0806] | -0.8357 [-2.8082, 0.7074] | Validation degradation |

C4R is important for interpretation: random-subset pacing was nearly as harmful
as acoustic tier pacing. This suggests that restricting the eligible pool, not
only imperfect difficulty estimates, caused much of C4's degradation. The
dynamic and hybrid methods also failed despite using model-derived signals,
which argues against insufficient acoustic-score sophistication as the sole
explanation.

### Replication

C3 test improvements against C0 seeds 42--44 were +0.8995, +0.3327, and
+0.5323 points. Their respective intervals were [0.3135, 1.3897], [0.1201,
0.5223], and [0.0209, 1.2611]. Validation effects were -0.3186, +0.6054, and
-0.0797 points. C3 therefore shows a stable WAXAL test ranking but not stable
validation improvement.

C5 did not replicate: its test effects were +0.0399, -1.8310, and -0.4232
points, and its mean validation/test changes were -1.0010/-0.7381 points. The
exact C5 seed-42 rerun established implementation repeatability but was not an
independent statistical replicate.

## RQ2 waveform augmentation

| Cell | Validation WER mean +/- SD | Test WER mean +/- SD |
|---|---:|---:|
| A0 clean random | 32.9058 +/- 0.4796 | 31.0446 +/- 0.2875 |
| A1 clean C3 | 32.8368 | 30.4564 |
| A2 augmented random | 33.0607 +/- 0.2436 | 31.2974 +/- 0.2791 |
| A3 augmented C3 | 33.0351 +/- 0.3360 | 31.3701 +/- 0.2728 |

Augmentation without curriculum changed validation/test WER by
-0.1549/-0.2528 points. Relative to clean C3, augmentation changed
validation/test WER by -0.1983/-0.9137 points. The validation interaction was
-0.0434 points, and no validation matched-bootstrap interval excluded zero.
The waveform policy is therefore rejected for promotion.

On clean WAXAL test, error decomposition across independently trained seeds
shows that A2 increased insertions from 4.543% to 4.829% of reference words and
substitutions from 22.694% to 22.771%, while deletions decreased from 3.807% to
3.697%. The modest deletion reduction was outweighed by insertion and
substitution increases.

## Corrected FLEURS

| Cell | Validation WER mean +/- SD | Test WER mean +/- SD |
|---|---:|---:|
| A0 clean random | 56.9423 +/- 1.2466 | 58.3099 +/- 0.5884 |
| A1 clean C3 | 59.6219 | 59.2294 |
| A2 augmented random | 56.9526 +/- 0.7179 | 59.4879 +/- 1.4715 |
| A3 augmented C3 | 57.0397 +/- 0.9363 | 60.1009 +/- 2.0391 |

FLEURS test augmentation effects were -3.4716, +0.1188, and -0.1814 points
for A2 versus matched A0 seeds 42--44. Only seed 42 excluded zero, and it favored
the control: 95% interval [-7.1061, -0.2183]. A3-versus-A2 effects were -1.0446,
-1.1509, and +0.3565 points, all unresolved. The three C3-versus-C0 effects were
also negative but unresolved. Thus neither curriculum nor augmentation shows
evidence of improved natural-domain transfer.

FLEURS has no speaker IDs. Its 10,000-replicate paired analyses cluster by the
retained `source_id` sentence prompt (348 test clusters), preventing repeated
recordings of the same prompt from being treated as independent. This protects
against prompt dependence but cannot account for unknown speaker dependence.

## Interpretation

The results support a conservative conclusion: ordering can alter optimization
outcomes, but the investigated curricula do not provide robust validation and
cross-domain improvements. C3's WAXAL test advantage may reflect a useful
in-domain ordering effect, yet its failure on FLEURS argues against a general
robustness benefit. Pacing was consistently harmful, and more adaptive scoring
did not repair it.

The waveform policy may be mismatched to the target distortions. Its synthetic
noise, reverberation, and tempo perturbations did not improve natural FLEURS
speech and sometimes increased insertions. This is a negative result about the
tested mild policy, not evidence that all augmentation is ineffective.

## Methodological cautions

- C3 is deterministic under the frozen order and zero-dropout setup. Reusing
  it against three stochastic C0 controls quantifies control variability but
  does not provide three independent C3 trainings.
- WAXAL test performance contributed to selecting C3. Consequently, that split
  is not an untouched final confirmatory holdout for the C3 claim. The thesis
  should describe WAXAL test findings as selection-stage evidence and use the
  locked corrected FLEURS evaluation as the independent cross-domain check.
- The seed-42 screen includes multiple curriculum comparisons. The paired
  intervals are descriptive unless a multiplicity procedure is added.
- Corrupted-WAXAL evaluation remains unimplemented, so robustness conclusions
  currently cover clean WAXAL and natural OOD FLEURS only.
- A1's displayed zero SD reflects one reused deterministic checkpoint, not zero
  population variance.

## Decisions supported by the evidence

- Retain C3 as the prespecified downstream curriculum condition, but qualify
  its benefit as WAXAL-specific and not validation- or OOD-robust.
- Do not promote `rq2-waveform-mild-v1`.
- Do not run the conditional SpecAugment or Whisper Medium confirmation paths.
- Use clean C3 as the preregistered RQ3/RQ4 treatment, while retaining a clean
  random control so downstream claims do not depend on the uncertain C3 effect.