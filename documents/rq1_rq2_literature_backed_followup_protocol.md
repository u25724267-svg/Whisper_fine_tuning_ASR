# Literature-Backed RQ1/RQ2 Follow-up Protocol

## Status and scope

This is a prospective protocol for `rq2-followup-v1`. It does not alter the
completed RQ1 conditions C0--C7/C4R or RQ2 conditions A0--A3, and it authorizes
no experiment launch. New conditions use a B-series namespace so completed
evidence cannot be silently reclassified.

## Evidence diagnosis

The completed waveform policy transformed 82.5% of training rows, materialized
one fixed view per row and seed, and combined noise, RIR, and pitch-preserving
tempo perturbation. It therefore could not identify which mechanism caused the
negative result. Selection chiefly used clean WAXAL validation, and the policy
did not improve corrected FLEURS. This motivates isolated mechanisms, preserved
clean exposure, and target-aligned robustness validation. It does not imply
that augmentation must improve ASR or that every literature-backed transform
will transfer to Shona Whisper.

## RQ1 decision

Broad RQ1 screening is closed. Strict score sweeps, competence/pacing grids,
and C3 crossings are not justified by the completed validation and OOD results.
The only optional diagnostic is `C6-Mix`: utterance-WER ranking plus the uniform
difficulty mixing reported by Karakasidis et al. (2022). It would be a
Whisper adaptation, not an exact replication, and would use one seed-42
validation-only gate. It is not required for RQ2 and cannot select an RQ2
condition.

## RQ2 conditions

| Condition | Training condition | Role |
|---|---|---|
| B00 | Clean data, conventional random sampling | Matched control |
| BN | Additive non-speech noise only | Isolated mechanism |
| BR | Simulated RIR only | Isolated mechanism |
| BT | Resampling-based speed factors 0.9, 1.0, and 1.1 | Closest transfer of Ko et al. (2015) |
| BNR | Only independently successful noise/RIR factors | Conditional primary treatment |

BN and BR retain 50% clean and 50% augmented presentations. This is a
conservative study-specific multi-condition choice, not a probability prescribed
by Ko et al. BT assigns one third of presentations to each 0.9/1.0/1.1
resampling factor, including the unmodified 1.0 view, as the closest algorithmic
transfer of Ko et al. (2015). Resampling-based speed changes duration and pitch;
the earlier pitch-preserving tempo transform is not equivalent.

SpecAugment, codecs, Whisper Medium, stronger doses, and curriculum crossings
are excluded from the primary study. Each would require a separately frozen,
literature-supported promotion decision.

## Asset governance and validation battery

Before training, SLR28 noise and RIR files must be assigned to train,
validation, and test inventories by decoded file-content SHA-256. Identical
content must always receive the same partition. Inventory files, hashes,
partition salt, and allocation proportions are immutable once generated.

The speaker-disjoint WAXAL validation battery contains paired views of the same
utterances:

| View | Definition |
|---|---|
| D0 | Clean audio |
| DN | Held-out non-speech noise at fixed 5, 10, and 15 dB SNR |
| DR | Held-out simulated RIR |
| DNR | Held-out noise plus simulated RIR |

The exact severity grid is a study-specific diagnostic range informed by
noise-robust ASR literature; it is not a universal optimum. These are controlled
synthetic robustness views, not natural OOD data. Validation assets may define
the training policy; test assets and manifests remain locked until promotion.
The completed A2 policy used the full asset pool, so any same-pool corrupted A2
result is exploratory and cannot serve as leakage-free confirmation.

Noise segments must be selected from files at least as long as the target
waveform and cropped at a seeded offset. Short clips must not be tiled because
periodic repetition creates an avoidable synthetic cue. This duration-compatible
selection is a study-specific artifact-control rule.

## Endpoints and promotion

Improvement is always defined as baseline WER minus candidate WER. The
robustness endpoint is the equal-weight macro-average of corruption-family WERs,
not WER after pooling all corrupted utterances. Clean D0 is a non-inferiority
constraint.

BN, BR, and BT are mechanism-selection comparisons. BNR versus B00 becomes the
primary contrast only after the successful mechanisms and composition are
frozen. Promotion requires all of the following:

1. Robustness macro-average improvement of at least 0.5 absolute WER points.
2. Paired 95% confidence-interval lower bound greater than zero.
3. Clean improvement confidence-interval lower bound greater than -0.5 points.

The 0.5-point thresholds are local practical margins, not literature constants.

## Seeds and external evaluation

Matched seeds 42--46 are fixed before execution and all five must complete.
Five is a study-specific compute/uncertainty compromise motivated by literature
showing that single-seed scores are unreliable; it is not a prescribed standard.
Existing B00-compatible controls for seeds 42--44 may be reused only if model,
revision, data, optimizer, update budget, decoding, and checkpoint selection are
identical. Otherwise they must be rerun.

FLEURS has already been inspected. It must never select a transform, severity,
checkpoint, seed, or stopping decision. After the complete WAXAL decision is
hashed and frozen, all completed conditions may be evaluated on FLEURS once and
reported as **adaptive external evaluation**. A confirmatory natural-OOD claim
requires a new untouched Shona corpus.

## Statistical analysis

- Use paired speaker-cluster bootstrap resampling for WAXAL, conditional on each
  fixed checkpoint.
- For a four-cell noise/RIR analysis, sample speakers once per replicate and
  reuse the same sample for every cell when computing main effects and
  interaction.
- Report every seed and its interval. Speaker bootstrap estimates evaluation
  sampling uncertainty, not retraining uncertainty.
- Add a two-stage seed-plus-speaker sensitivity analysis if all required cells
  have independent matched seeds.
- Apply Holm adjustment to the noise, RIR, and noise-by-RIR interaction family.
  CER, severities, speakers, and acoustic slices are secondary or descriptive.

## Frozen stopping rules

- Do not tune after observing seed 42; complete seeds 42--46 unchanged.
- Combine only isolated mechanisms that independently satisfy promotion.
- Do not add dose levels, SpecAugment, codecs, C3, or Medium unless a separate
  rule was frozen before inspecting candidate outcomes.
- Rerun only documented technical failures, with the same seed and configuration.
- Stop waveform follow-up if no isolated mechanism passes.

## Decision ledger

| Decision | Evidence | Classification | Frozen implication |
|---|---|---|---|
| Close broad RQ1; optional C6-Mix only | Karakasidis et al. (2022); completed C5--C7 | Literature-motivated adaptation | No new score or pacing sweep |
| B00 random clean control | Standard matched experimental control | Study-specific control | All treatment contrasts remain matched |
| BN/BR isolate mechanisms | Ko et al. (2017) | Literature-motivated adaptation | No composite policy during selection |
| BN/BR use 50% clean replay | Multi-condition training and prior high-dose failure | Study-specific operational choice | Clean exposure is preserved at a frozen rate |
| BT uses 0.9/1.0/1.1 resampling | Ko et al. (2015) | Closest algorithmic transfer | Do not substitute tempo stretching |
| BNR is conditional | Mechanism-first ablation logic | Study-specific control | Composition freezes only after isolated results |
| Content-hash asset partitions | Leakage prevention and SLR28 provenance | Study-specific control | No asset content crosses partitions |
| Duration-compatible noise selection | Avoid periodic artifacts from tiled clips | Study-specific control | Crop long noise; never repeat short noise |
| D0/DN/DR/DNR validation | Ko et al. (2017); Braun et al. (2017) | Literature-motivated adaptation | Synthetic robustness, not natural OOD |
| Five matched seeds | Reimers and Gurevych (2017) | Study-specific operational choice | Report all seeds; no optional stopping |
| 0.5-point margins | Existing thesis decision threshold | Study-specific operational choice | Practical, not statistical or universal |
| Paired bootstrap and Holm | Bisani and Ney (2004); Holm (1979) | Literature-backed analysis | Preserve pairing and control the stated family |
| FLEURS restricted to one final pass | Prospective leakage control | Study-specific control | Report as adaptive external evaluation |
| Exclude SpecAugment/codecs/C3/Medium | Factor isolation and completed evidence | Study-specific control | Separate preregistration required |

## Primary references

- Ko et al. (2015), [Audio Augmentation for Speech Recognition](https://doi.org/10.21437/Interspeech.2015-711).
- Ko et al. (2017), [A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition](https://doi.org/10.1109/ICASSP.2017.7953152).
- Braun et al. (2017), [A Curriculum Learning Method for Improved Noise Robustness in Automatic Speech Recognition](https://doi.org/10.23919/EUSIPCO.2017.8081267).
- Park et al. (2019), [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://doi.org/10.21437/Interspeech.2019-2680).
- Karakasidis et al. (2022), [Comparison and Analysis of New Curriculum Criteria for End-to-End ASR](https://doi.org/10.21437/Interspeech.2022-10046).
- Reimers and Gurevych (2017), [Reporting Score Distributions Makes a Difference](https://doi.org/10.18653/v1/D17-1035).
- Bisani and Ney (2004), [Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation](https://doi.org/10.1109/ICASSP.2004.1326000).
- Holm (1979), [A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733).
- [OpenSLR SLR28: Room Impulse Response and Noise Database](https://www.openslr.org/28/).

## Execution order

1. Generate and audit content-hash-disjoint asset inventories.
2. Generate one-row fixtures, then immutable D0/DN/DR/DNR validation manifests.
3. Freeze manifests, endpoints, seeds, implementation hashes, and stopping rules.
4. Run B00/BN/BR/BT for seeds 42--46 without policy tuning.
5. Evaluate the locked validation battery and identify independent passers.
6. Freeze BNR composition and its BNR-versus-B00 primary contrast, if eligible.
7. Run all five BNR seeds, if eligible, and freeze the WAXAL decision.
8. Open locked WAXAL test assets/manifests once.
9. Run the one-time FLEURS adaptive external evaluation.

This document does not authorize steps 4--9 to start automatically.