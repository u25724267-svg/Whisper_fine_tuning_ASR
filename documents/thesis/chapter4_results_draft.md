# Chapter 4: Results

> **Draft status.** This chapter reports results verified from saved metric,
> prediction, provenance, and bootstrap artifacts as of 2026-09-15. RQ2
> B-series training, RQ4 proxy qualification, manual audit, and RQ4 student
> comparisons remain incomplete and are not assigned outcomes here.

## 4.1 Reporting Conventions

Results are separated by experimental protocol because the archival,
SDE-normalized, and speaker-disjoint datasets are not interchangeable. WER and
CER are percentages unless otherwise stated. For treatment effects, a positive
value means that the treatment reduced error relative to its named control.
Speaker-clustered bootstrap intervals contain 10,000 paired replicates. These
intervals quantify evaluation-sample uncertainty conditional on fixed
checkpoints; they do not include training-seed, split-construction, or
model-selection uncertainty.

The speaker-disjoint WAXAL v2 protocol is primary for RQ1 and the completed RQ2
A-series. Results on the original speaker-overlapping splits are retained only
as exploratory evidence. Corrected FLEURS is natural external-domain evidence,
but its speakers are unavailable. FLEURS bootstrap analyses therefore cluster
recordings by source sentence prompt rather than speaker.

## 4.2 Data-Protocol Findings

The matched official-split control reproduced the historical Whisper Base
result within 0.25 WER points, whereas the speaker-disjoint protocol was more
difficult.

| Protocol/run | Validation WER | Test WER | Role |
|---|---:|---:|---|
| Original official split | 29.0647 | 28.4893 | Historical reference |
| Matched official split | 29.2918 | 28.7307 | Pipeline diagnostic |
| Speaker-disjoint v2, C0 seed 42 | 32.5262 | 31.3400 | Confirmatory control |

This table uses final Trainer evaluation metrics. Paired RQ1 effects below use
the standalone item-prediction evaluator, which gave 29.2995/28.7232 for the
matched official diagnostic and 32.5182/31.3560 for C0 seed 42. The small
differences reflect distinct metric-emission surfaces; each comparison uses one
surface consistently rather than mixing counts within a contrast.

The increase under speaker-disjoint evaluation is consistent with unseen-speaker
and split-composition difficulty rather than a training-pipeline regression.
Version 2 contained 113 training, 24 validation, and 24 test speakers with zero
speaker overlap. No held-out speaker contributed more than 20% of a split's rows
or duration.

## 4.3 Exploratory Model Capacity

Model-capacity experiments used the earlier SDE-normalized protocol and are not
directly pooled with speaker-disjoint results.

| Model | Validation WER | Test WER |
|---|---:|---:|
| Whisper Base | 29.0647 | 28.4893 |
| Whisper Medium | 22.9450 | 23.7739 |
| Whisper Large | 22.8735 | 22.7536 |

Medium reduced test WER by 4.72 points relative to Base. Large improved by a
further 1.02 points over Medium. Recorded training runtime was 9,205.9 seconds
(2.56 hours) for Medium and 55,273.1 seconds (15.35 hours) for Large; comparable
peak-memory measurements were not retained. These results motivated using
Whisper Base for controlled screening,
where more experimental conditions and seeds were feasible. They do not
establish that model size caused the same improvement under speaker-disjoint
evaluation because that comparison was not rerun on v2.

## 4.4 RQ1: Curriculum Learning

### 4.4.1 Seed-42 screen

The seed-42 screen compared every curriculum against the matched C0 random
control. Positive effects indicate lower treatment WER.

| Condition | Validation effect [95% CI] | Test effect [95% CI] | Result |
|---|---:|---:|---|
| C1 duration SortaGrad | -0.0690 [-0.6422, 0.2726] | +0.3407 [-0.5471, 0.9976] | Unresolved |
| C2 strict SNR | -0.6001 [-1.0738, -0.1934] | -0.1783 [-1.5829, 1.0127] | Validation degradation |
| C3 SNR-duration | -0.3186 [-0.7530, -0.0232] | +0.8995 [0.3135, 1.3897] | Split-dependent |
| C4 cumulative acoustic | -1.9516 [-2.7561, -1.2665] | -1.2295 [-1.7750, -0.7388] | Harmful |
| C4R random pacing | -1.8746 [-2.7805, -1.1709] | -1.1045 [-1.6084, -0.6670] | Harmful |
| C5 S2S-margin (duration-normalized NLL) | -1.3170 [-2.1487, -0.5906] | +0.0399 [-0.9377, 0.9178] | Validation degradation |
| C6 utterance WER | -1.5135 [-2.1838, -0.9526] | -0.4152 [-1.4054, 0.5231] | Costly; no gain |
| C7 acoustic-S2S | -2.0047 [-3.2909, -1.0806] | -0.8357 [-2.8082, 0.7074] | Validation degradation |

The intervals are unadjusted screen-level intervals. Because eight curriculum
conditions were examined, an interval excluding zero is not presented as
family-wise statistical significance. C3 was the only treatment with a
practically meaningful positive seed-42 test estimate and a wholly positive
test interval, but its validation result favored C0.

C4R was nearly as harmful as C4. This indicates that restricting the eligible
pool and repeating early examples contributed substantially to the degradation;
the acoustic ranking alone cannot explain C4. C5--C7 also failed despite using
model-derived difficulty, arguing against difficulty-score sophistication as a
sufficient remedy for strict curriculum ordering.

### 4.4.2 Selected repeated runs

| Condition | Validation WER, mean ± SD | Test WER, mean ± SD | Mean test change vs C0 |
|---|---:|---:|---:|
| C0 random | 32.9058 ± 0.4796 | 31.0446 ± 0.2875 | Reference |
| C3 SNR-duration | 32.8368 ± 0.0000 | 30.4564 ± 0.0000 | 0.5882 improvement |
| C5 S2S-margin (duration-normalized NLL) | 33.9069 ± 0.2611 | 31.7827 ± 0.7268 | 0.7381 degradation |

C3 improvements against C0 seeds 42--44 were 0.8995, 0.3327, and 0.5323 test
WER points. Validation effects were -0.3186, +0.6054, and -0.0797 points. C3 is
deterministic under strict ordering and zero dropout: its seed-42 and seed-43
models and predictions were byte-identical, and the same verified artifact was
used for the seed-44 comparison. The zero SD is therefore structural reuse, not
evidence of zero training variance or three independent replications.

C5 test effects were +0.0399, -1.8310, and -0.4232 points. Its exact seed-42
rerun reproduced every checkpoint hash and metric, establishing implementation
repeatability but not adding an independent statistical observation.

### 4.4.3 Computational cost

C6 required 14,062.7 seconds (3.91 hours), approximately 6.4 times the C0
seed-42 runtime of 2,200.2 seconds (36.7 minutes). Dynamic utterance-level WER
scoring therefore added substantial cost without an accuracy gain.

### 4.4.4 RQ1 answer

The investigated curricula did not produce a robust improvement across
validation, WAXAL test, and corrected FLEURS. C3 produced a modest
WAXAL-specific test ranking advantage, but validation was inconsistent and the
advantage did not transfer to FLEURS. Furthermore, WAXAL test performance
contributed to selecting C3 for the A-series, so that test result is
selection-stage evidence rather than an untouched confirmation. RQ1 therefore
supports a qualified negative conclusion for Whisper Base, this three-epoch
budget, and the exact curriculum definitions tested.

## 4.5 RQ2: Data Augmentation

### 4.5.1 Exploratory SpecAugment

The SpecAugment pilots used the earlier speaker-overlapping protocol and mostly
single seeds. They are reported as exploratory rather than confirmatory.

| Model/policy | Validation WER | Test WER | Improvement vs named control |
|---|---:|---:|---|
| Base clean | 29.0647 | 28.4893 | Reference |
| Base mild always-on | 29.0443 | 28.5689 | +0.0204 / -0.0796 vs clean |
| Base mild 50/50 | 29.3862 | 28.7232 | -0.3216 / -0.2339 vs clean |
| Medium clean | 22.9450 | 23.7739 | Reference |
| Medium mild always-on | 22.2636 | 22.7487 | +0.6814 / +1.0252 vs clean |
| Medium continuation stage 2 | 22.5214 | 22.7661 | -0.2578 / -0.0174 vs mild parent |

Positive changes consistently mean lower treatment WER. Base masking was
practically negligible or harmful. The one-seed Medium result was larger, but it was not independently
replicated and used the speaker-overlapping split. Continuing that checkpoint
for another 2,000 steps worsened validation and did not materially change test
WER. The configured Medium stage-3 continuation has no completed metric artifact
and is not assigned a result.

### 4.5.2 Completed A-series factorial

| Cell | Description | Validation WER, mean ± SD | Test WER, mean ± SD |
|---|---|---:|---:|
| A0 | Clean, random | 32.9058 ± 0.4796 | 31.0446 ± 0.2875 |
| A1 | Clean, C3 | 32.8368 | 30.4564 |
| A2 | Composite waveform, random | 33.0607 ± 0.2436 | 31.2974 ± 0.2791 |
| A3 | Composite waveform, C3 | 33.0351 ± 0.3360 | 31.3701 ± 0.2728 |

Waveform augmentation without curriculum produced validation/test improvements
of -0.1549/-0.2528 points, meaning WER increased by 0.1549/0.2528 points. With
C3, augmentation produced validation/test improvements of
-0.1983/-0.9137 improvement points. No validation matched-bootstrap interval
excluded zero. The validation difference-in-differences interaction was -0.0434
points and was not formally cluster-bootstrapped as an interaction contrast.

A2 increased insertions from 4.543% to 4.829% of reference words and
substitutions from 22.694% to 22.771%, while deletions decreased from 3.807% to
3.697%. The deletion reduction was outweighed by insertion and substitution
increases.

A1 reuses one deterministic C3 checkpoint in all aggregate positions. Its zero
SD and contrasts involving A1 are descriptive rather than three-run inference.
A2 versus A0 and A3 versus A2 are independently trained, seed-matched
comparisons.

### 4.5.3 Corrected FLEURS

| Cell | Validation WER, mean ± SD | Test WER, mean ± SD |
|---|---:|---:|
| A0 clean random | 56.9423 ± 1.2466 | 58.3099 ± 0.5884 |
| A1 clean C3 | 59.6219 | 59.2294 |
| A2 waveform random | 56.9526 ± 0.7179 | 59.4879 ± 1.4715 |
| A3 waveform C3 | 57.0397 ± 0.9363 | 60.1009 ± 2.0391 |

A2-versus-A0 FLEURS test effects were -3.4716, +0.1188, and -0.1814
improvement points across seeds 42--44. Only seed 42 excluded zero, and it
favored the clean control: 95% CI [-7.1061, -0.2183]. A3-versus-A2 effects were
-1.0446, -1.1509, and +0.3565 points, all unresolved. Prompt-clustered C3 versus
C0 effects were also unresolved. Neither curriculum nor the composite waveform
policy showed improved natural-domain transfer.

### 4.5.4 RQ2 answer and follow-up status

The frozen composite waveform policy was not promoted. This is a negative result
about its high augmented dose, one-view materialization, and combined
tempo/RIR/noise implementation; it is not evidence that augmentation is
universally ineffective. The mechanism-isolating B-series has leakage-safe
D0/DN/DR/DNR validation assets and a frozen analysis plan, but no B-series model
outcomes exist yet. RQ2's completed answer therefore applies to the A-series;
the B-series remains prospective.

## 4.6 RQ4: Pseudo-Labeling Progress

### 4.6.1 Admission and teacher generation

| Stage | Rows | Hours | Speakers |
|---|---:|---:|---:|
| Audited WAXAL unlabelled source | 85,384 | 475.65 | 219 |
| Admitted after model-independent gates | 60,677 | 320.83 | 177 |
| Completed C0 teacher labels | 60,677 | 320.83 | 177 |

Admission removed evaluation-speaker overlap, duration failures, exact decoded
audio overlap, and duplicates before teacher inference. The C0 seed-42 teacher
produced no empty outputs. It produced 328 maximum-length outputs, which were
excluded before shortlist ranking. Teacher generation completed in 238
checksummed chunks. The consolidated prediction SHA-256 was
`86c7decb9334c3c691f79059ae7b7d3b3665ad9b1f4f2e3524e3747e5e74e079`.

### 4.6.2 Labelled diagnostic calibration

The same teacher was evaluated on 13,807 labelled WAXAL training utterances from
113 speakers. Corpus WER/CER was 21.25%/4.78%. Utterance-level poor-label counts
were 1,430 above 40% WER, 606 above 50%, and 116 above 80%. There were no empty
outputs and 61 maximum-length outputs.

Five outer and four inner speaker-group folds compared confidence alone with a
logistic model using the frozen teacher diagnostics. $C=0.1$ was selected in
every outer fold and by the full inner analysis.

| Poor-label target | Score | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| WER >40% | Confidence | 0.7600 | 0.3827 |
| WER >40% | Diagnostic model | 0.7746 | 0.4044 |
| WER >50% | Confidence | 0.7738 | 0.2830 |
| WER >50% | Diagnostic model | **0.8086** | **0.3676** |
| WER >80% | Confidence | 0.8042 | 0.0967 |
| WER >80% | Diagnostic model | 0.8705 | 0.5330 |

For the primary WER-above-50% target, the diagnostic model passed the local
0.70 usefulness gate and produced higher cross-fitted ROC-AUC and PR-AUC point
estimates than confidence alone. No paired uncertainty interval for the AUC
difference has yet been computed. Canonical Whisper thresholds did not
form complete correctness gates: compression ratio above 2.4 had precision
0.866 but recall 0.096, while mean content-token log probability below -1.0
identified no positives. Max-length status had precision 0.918 but recall 0.092.
These results support multivariate calibration rather than direct transfer of
long-form Whisper defaults.

### 4.6.3 Proxy shortlist

The initial 2% dual speaker cap was structurally unable to produce 80 hours,
with maximum label-blind capacity of 79.12 hours. Before shortlist creation, the
cap was changed to the smallest evaluated feasible value, 2.5%, with capacity
91.58 hours.

| Component | Rows | Hours | Speakers |
|---|---:|---:|---:|
| Confidence top 80 h | 15,016 | 79.9999 | 154 |
| Diagnostic top 80 h | 15,832 | 79.9996 | 155 |
| Overlap | 11,441 | 62.3074 | 149 |
| Union shortlist | 19,407 | 97.6921 | 160 |

The union's maximum speaker duration and row shares were 3.55% and 3.43%. The
independent 2.5% component caps imply a maximum 5% union bound, so these
realized shares satisfy the frozen union constraint. The shortlist SHA-256 was
`d6e24b0d3b963e6d357bbb0f7003348775f98e827cadf485d9880a2ba9da372f`.

### 4.6.4 SeamlessM4T-v2 proxy status

The pinned zero-shot SeamlessM4T-v2 model completed labelled-training inference
over 13,807 utterances (79.76 hours, 113 speakers) in 216 chunks. It produced no
empty or maximum-length outputs. Median normalized character disagreement with
the Whisper teacher was 0.3958, with p05/p95 values of 0.0956/0.8038. The proxy
prediction SHA-256 was
`8846bcbff2b81fb6248db601fd5a1f6ff21d92879db3faf502c3c2060e37ebb0`.

Proxy qualification ROC-AUC/PR-AUC against true teacher error has not yet been
computed. The Seamless proxy therefore remains unqualified and has not been run
over the unlabelled shortlist. No manual-audit or student-training result exists.

### 4.6.5 Interim RQ4 answer

The 320.83-hour admitted pool exceeds the largest planned 80-hour student pool,
and teacher generation is technically feasible. Multivariate Whisper diagnostics
produced higher cross-fitted AUC point estimates than confidence alone on
labelled training speakers. It
is not yet possible to conclude that filtered pseudo-labels improve ASR because
proxy qualification, manual audit, and matched student comparisons are pending.

## 4.7 RQ3 Status

No suitable licensed Tshivenda corpus has been admitted for the confirmatory
transfer experiment. RQ3 therefore has no confirmatory result. Legacy
multilingual Shona/Tshivenda/isiZulu runs remain exploratory because they did not
hold language prompts, data, and training budgets constant.

## 4.8 Integrated Findings

1. Speaker-disjoint evaluation produced materially higher WER than the original
  overlapping split, consistent with the combined effects of unseen speakers
  and changed split composition.
2. Strict curriculum ordering did not improve Whisper Base consistently.
3. Cumulative pacing harmed both validation and test performance even when
   priority was randomized.
4. Dynamic S2S and WER difficulty did not improve strict ordering, and WER
   scoring imposed substantial computational cost.
5. The tested composite waveform augmentation did not improve clean WAXAL or
   corrected FLEURS performance.
6. Off-the-shelf Whisper thresholds were insufficient pseudo-label quality
   gates, while grouped multivariate calibration improved error discrimination.
7. Downstream pseudo-label and mechanism-isolating augmentation conclusions must
   wait for their frozen remaining stages.

## 4.9 Limitations

- Most RQ1 conditions were screened with one seed, and the seed-42 interval
  family was not multiplicity-adjusted.
- C3 is deterministic and reused, so its apparent zero seed variance is not an
  estimate of training stability.
- WAXAL test contributed to C3 selection; C3 test performance is not an
  independent confirmation.
- FLEURS prompt clustering cannot account for unknown speaker dependence.
- Exploratory model-size and SpecAugment results use speaker-overlapping splits.
- The A-series interaction is descriptive rather than directly bootstrapped.
- Synthetic corruption evaluates controlled robustness rather than natural OOD
  transfer.
- RQ4 diagnostic calibration uses utterances on which the C0 teacher was itself
  trained. Speaker-grouped cross-fitting separates diagnostic-model folds but
  cannot remove possible optimism from teacher-training exposure or distribution
  differences between labelled and unlabelled WAXAL.
- Proxy and student qualification remain incomplete.
- Results are specific to the selected Whisper revisions, Shona datasets,
  three-epoch budget, preprocessing contract, and operational method definitions.

## 4.10 Chapter Summary

RQ1 found no curriculum with consistent validation and cross-domain benefit.
C3 showed a modest WAXAL-specific ranking advantage, while cumulative and
dynamic strategies were ineffective or harmful. The completed RQ2 A-series did
not support promotion of its composite waveform policy, including on corrected
FLEURS. RQ4 established a large admitted unlabelled pool, completed teacher and
labelled-proxy inference, and produced higher cross-fitted AUC point estimates
for grouped multivariate diagnostics than for confidence alone. RQ2 B-series
training, proxy qualification,
manual audit, RQ4 student comparisons, and RQ3 transfer remain pending and must
be added only after their immutable analyses complete.

## Author Notes Before Final Submission

- Replace the RQ4 proxy-status paragraph with qualification metrics when the
  speaker-grouped analysis is complete.
- Add B-series and student results only after their frozen promotion decisions.
- Reconcile the stale Base mixed and Medium stage-3 run reports with their
  machine-readable artifacts before citing them outside this draft.
- Generate final figures from immutable JSON/JSONL artifacts rather than manual
  transcription.
- Convert citations and artifact references to the university's required style.