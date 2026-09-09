# Shona ASR Confirmatory Experiments Roadmap

Last updated: 2026-09-08

## Scope and priorities

The confirmatory study is organized around four research questions:

1. effects of static and dynamic acoustic data-level curriculum learning;
2. effects of controlled waveform augmentation on synthetic robustness and
  external generalization;
3. transfer of a Shona-selected curriculum to Tshivenda; and
4. improvement from validation-calibrated proxy/hybrid-filtered pseudo-labeling.

RQ1 and the original RQ2 A-series are complete. Current effort is restricted to
the mechanism-isolating RQ2 B-series and RQ4 proxy-filtered pseudo-labeling. RQ3
remains blocked on a suitable licensed Tshivenda corpus. Every non-obvious new
choice must be traced to a primary source, selected by a frozen validation-only
sweep, or justified as a measured engineering constraint.

## Fixed experiment contract

- Primary screen model: pinned multilingual `openai/whisper-base`.
- Confirmation model: Whisper Medium only after a frozen Base promotion rule.
- Shona protocol: `waxal_shona_speaker_disjoint_v2`.
- Split sizes: 13,807 train, 1,715 validation, and 1,671 test utterances.
- Speaker counts: 113 train, 24 validation, and 24 test, with zero overlap.
- Maximum held-out speaker share: 20% of rows and hours.
- Evaluation and saving: epoch boundaries; all three checkpoints retained.
- RQ1/A-series/RQ4 student seeds: 42--44; B-series confirmation seeds: 42--46.
- Existing W&B project: `whisper-shona-multilingual`.
- Output root: `/ext_data/casper/asr_experiment_outputs`.
- Final test data is not used for scheduler or hyperparameter selection.
- FLEURS has already been inspected and is adaptive external evaluation for any
  new B-series or RQ4 claim; it cannot select parameters.
- Every analyzed run requires item-level predictions and artifact hashes.

The established `train_full.py` baseline behavior remains frozen. Curriculum
treatments use a separate runner and shared tested sampler modules.

The decision evidence is maintained in
`documents/rq1_rq2_literature_defensibility.md`,
`documents/rq1_rq2_literature_backed_followup_protocol.md`,
`documents/data/rq4_pseudolabel_method_review.md`, and
`documents/data/rq4_pseudolabel_protocol_v1.md`. A parameter not directly
specified by a primary source must be selected by a frozen validation-only
sweep or documented as a measured engineering constraint before use.

## Current status

| Item | Status | Evidence or blocker |
|---|---|---|
| Speaker-disjoint Shona protocol v2 | Complete | Maximum speaker row share is 19.5% validation and 17.2% test; zero overlap |
| RQ1 C0--C7/C4R | Complete | Screen, selected repeats, item predictions, and paired speaker-bootstrap analyses complete |
| RQ1 conclusion | Complete | No curriculum improved consistently across validation, WAXAL test, and corrected FLEURS; optional C6-Mix is diagnostic only |
| RQ2 A0--A3 | Complete | Seeds 42--44 and corrected FLEURS complete; composite waveform policy not promoted |
| RQ2 follow-up assets | Complete | Content-hash-disjoint SLR28 partitions and paired D0/DN/DR/DNR validation battery audited |
| RQ2 B-series | Prospective | Parameter-sweep amendment must freeze clean ratio/SNR before B00/BN/BR/BT training |
| Corrected FLEURS OOD protocol | Complete; closed to selection | Later use is adaptive external evaluation, not pristine confirmation |
| RQ3 Tshivenda transfer | Blocked | No suitable licensed corpus is currently available |
| RQ4 admission | Complete | 60,677 eligible utterances, 320.83 hours, 177 speakers after overlap/duration gates |
| RQ4 C0 teacher inference | Complete | 60,677 labels in 238 chunks; prediction SHA-256 `86c7decb9334...e74e079` |
| RQ4 diagnostic calibration | Complete | Nested speaker-grouped primary ROC-AUC 0.8086 versus confidence 0.7738 |
| RQ4 proxy shortlist | Complete | 97.69-hour confidence/diagnostic union, 19,407 rows, 160 speakers |
| RQ4 proxy/student study | License-gated/prospective | Seamless labelled qualification, manual audit, and matched students remain |

## Immediate engineering sequence

1. Freeze a versioned sweep amendment before inspecting any new B-series or RQ4
  student outcome.
2. Calibrate RQ4 hallucination/confidence gates on speaker-grouped labelled
  training data.
3. Build the speaker-capped 160-hour RQ4 shortlist and verify SeamlessM4T-v2
  license suitability before proxy inference.
4. Run the limited RQ2 clean-ratio/SNR screen, freeze selected parameters, then
  run B00/BN/BR/BT confirmation without test/FLEURS selection.
5. Complete proxy qualification, manual audit, pool-size/loss-weight selection,
  and matched RQ4 student training.

## Baseline protocol finding

| Run | Validation WER | Test WER | Role |
|---|---:|---:|---|
| Original official split | 29.0647% | 28.4893% | Historical reference |
| Matched official split | 29.2918% | 28.7307% | Diagnostic bridge |
| Speaker-disjoint v2 | 32.5262% | 31.3400% | Confirmatory RQ1 control |

The matched official split reproduces the historical result within 0.25 WER
points. The higher speaker-disjoint result is therefore consistent with the
expected unseen-speaker and held-out-composition difficulty rather than training
pipeline drift. Curriculum treatments continue on v2; the official split is
reported only as a secondary in-domain benchmark.

## RQ1 curriculum screen

All seed-42 conditions use identical data, model revision, optimizer, update
budget, batch size, decoding, and evaluation. Only ordering or pacing changes.

| ID | Condition | Family | Additional dependency |
|---|---|---|---|
| C0 | Seeded random shuffle | Control | Complete |
| C1 | Duration SortaGrad | Static ordering | Complete: test WER 31.0153%, a 0.3407-point improvement over C0-v2 |
| C2 | Strict SNR ordering | Static acoustic | Complete: test WER 31.5343%, 0.1783 points worse than C0-v2 |
| C3 | Strict joint SNR-duration ordering | Static acoustic | Complete: test WER 30.4564%, 0.8995 points better than C0-v2; validation 0.3186 worse |
| C4 | Cumulative acoustic tiers | Static pacing | Complete: validation WER 34.4698%, test WER 32.5855%; item predictions exported |
| C4R | Random-subset pacing | Mandatory pacing control | Complete: item-level validation WER 34.3928%, test WER 32.4604% |
| C5 | Random epoch 1, then S2S-margin | Dynamic model-based | Complete: item-level validation WER 33.8352%, test WER 31.3160% |
| C5 repeatability | Exact seed-42 C5 rerun | Repeatability check | Complete; checkpoint, Trainer metric, and item-prediction hashes match C5 |
| C6 | Random epoch 1, then WER-margin | Dynamic model-based | Complete: item-level validation WER 34.0316%, test WER 31.7711% |
| C7 | Random epoch 1, then 0.5 acoustic + 0.5 S2S-margin | Hybrid | Complete: item-level validation WER 34.5229%, test WER 32.1916% |

The fixed 10,000-replicate speaker-clustered bootstrap screen is complete for
C1-C7 against C0-v2. C3 is the only condition with a practically meaningful
positive test effect and a wholly positive 95% interval: improvement 0.8995
WER points, 95% CI 0.3135--1.3897, and probability 0.9083 of exceeding the
0.5-point threshold. Its validation effect is negative (-0.3186 points).

C5 is the strongest dynamic or hybrid condition but remains unresolved on test
(improvement 0.0399, 95% CI -0.9377--0.9178) and is worse on validation
(-1.3170, 95% CI [-2.1487, -0.5906]). Following the prespecified replication
policy, seeds 43 and 44 advance for C0, C3, and C5. C5 advances to establish
whether the dynamic result is consistently null, not because seed 42 supports
improvement. C1, C2, C4, C4R, C6, and C7 do not advance.

Replication execution is sequential. C0 seed 43 is complete with item-level
validation/test WER 33.4422/30.7891. C3 seed 43 is complete with item-level
validation/test WER 32.8368/30.4564; all model and prediction hashes exactly
match C3 seed 42 because strict ordering and zero dropout make its seed inactive.
C5 seed 43 is complete with item-level validation/test WER 33.6891/32.6201,
including a 1.8310-point test degradation versus C0 seed 43. C0 seed 44 is
complete with item-level validation/test WER 32.7572/30.9887. C5 seed 44, the
final planned RQ1 training replication, is complete with item-level
validation/test WER 34.1963/31.4118.

RQ1 aggregate analysis is complete. C3 is selected for the RQ2 curriculum
factor: mean test improvement 0.5882 WER points across matched C0 seeds, with a
mean validation improvement of 0.0690 points. C5 is rejected after mean test and
validation degradations of 0.7381 and 1.0010 points. The frozen analysis is in
`documents/rq1_final_analysis.md`.

The completed design retained C4R as the matched pacing control, began dynamic
C5--C7 from C0's random epoch-one order, and disclosed that deterministic C3
has no independent seed dimension. These controls remain part of the final RQ1
method rather than future work. The only permitted RQ1 follow-up is an optional
seed-42 C6-Mix diagnostic transferring the uniform WER-difficulty mixing of
Karakasidis et al. (2022); it cannot select an RQ2 or RQ4 condition.

### Null-result path

The null-result path was reached: the investigated curricula did not improve
consistently across validation, WAXAL test, and corrected FLEURS. All conditions
and negative results are retained. C3's WAXAL-specific ranking advantage is
reported with its test-selection and deterministic-reuse limitations; it is not
treated as a general robustness gain.

## RQ2 A-series: curriculum and composite augmentation

Synthetic augmentation means waveform modification: controlled additive noise,
room impulse response convolution, and mild pitch-preserving tempo perturbation. SpecAugment is
disabled or fixed identically across the factorial.

| ID | Curriculum | Waveform augmentation |
|---|---:|---:|
| A0 | No | No |
| A1 | Yes | No |
| A2 | No | Yes |
| A3 | Yes | Yes |

A0 and A1 reused matched RQ1 artifacts where every relevant field and manifest
hash agreed. The four-cell factorial and all six augmented runs completed over
seeds 42--44.

Evaluation included clean speaker-disjoint WAXAL and corrected FLEURS as natural
OOD speech. FLEURS was opened only after the A-series selection decision froze.

The RQ2 policy is frozen as `rq2-waveform-mild-v1` using Apache-2.0 SLR28
point-source noises and simulated RIRs. Seed-42 materialization passed the full
13,807-row audit. A2 seed 42 ran on W&B run `e4dwoxn2`; SpecAugment was disabled
and clean validation/test audio was unchanged.

All six newly trained A2/A3 runs for seeds 42--44 are complete with hashed
item-level predictions. All 18 matched speaker-bootstrap comparisons and the
descriptive factorial aggregate are complete. The primary decision is frozen:
`rq2-waveform-mild-v1` is not promoted, and clean C3 remains the selected
supervised condition. Mean validation WER was 32.9058 (A0), 32.8368 (A1),
33.0607 (A2), and 33.0351 (A3). Augmentation changed validation WER by -0.1549
points without curriculum and -0.1983 points with C3; the interaction was
-0.0434 points. See `documents/rq2_final_analysis.md` for uncertainty results
and the A1 reuse limitation. After this decision was frozen, corrected FLEURS
validation/test inference completed for all 10 distinct A0--A3 checkpoints.
Mean FLEURS test WER was 58.3099 (A0), 59.2294 (A1), 59.4879 (A2), and 60.1009
(A3); these OOD results do not rescue augmentation. The conclusion is limited
to this high-dose composite policy.

## RQ2 B-series: mechanism-isolating follow-up

Ko et al. (2015, 2017) motivate isolated resampling-speed, noise, and simulated
RIR conditions. The B-series contains B00 clean, BN noise-only, BR RIR-only, BT
0.9/1.0/1.1 resampling, and conditional BNR using only independently successful
mechanisms. SpecAugment, codecs, C3, stronger doses, and Medium are excluded
unless a separate pre-outcome amendment justifies them.

Content-hash-disjoint SLR28 train/validation/test inventories and the paired
1,715-row D0/DN/DR/DNR validation battery are complete. Test assets remain
locked. Synthetic-corruption performance is not presented as natural OOD
generalization, and later FLEURS use is adaptive external evaluation.

Before B-series training, a versioned sweep amendment freezes:

1. clean-presentation candidates $1/3$, $1/2$, and $2/3$;
2. training SNR candidates bounded by the fixed 5/10/15 dB validation range;
3. seed-42 selection using macro corruption-family WER and clean
  non-inferiority; and
4. confirmation on seeds 43--46 without further tuning.

The ratio grid is a local sensitivity study around Ko's one-third clean share,
not a published optimum. Promotion requires at least 0.5 WER-point robustness
improvement, a paired interval lower bound above zero, and clean-improvement
lower bound above -0.5. Holm adjustment covers noise, RIR, and interaction.
BNR is allowed only after its composition is frozen from isolated results.

## RQ3 Shona-to-Tshivenda curriculum transfer

The curriculum is selected using Shona only and frozen before Tshivenda runs.
No licensed Tshivenda corpus is present in the local data roots, and Google
FLEURS does not provide Tshivenda. RQ3 training remains blocked on corpus
selection and preparation; no substitute language is used silently.

| ID | Tshivenda condition |
|---|---|
| T0 | Random control |
| T1 | Frozen Shona-derived curriculum |
| T2 | Tshivenda-derived native curriculum |

All three conditions use the same Tshivenda data, updates, and language-neutral
Whisper decoder prefix. Transfer efficiency compares the gain from T1 with the
gain from T2. The legacy Shona+Tshivenda and Shona+Tshivenda+isiZulu pilots are
context only because they mixed languages, budgets, and prompt policies.

## RQ4 pseudo-labeling

The model-independent gate is complete. Of 85,384 WAXAL unlabelled rows (475.65
hours), 60,677 rows, 320.83 hours, and 177 speakers remained after decoded-PCM
deduplication, WAXAL/FLEURS overlap removal, validation/test-speaker exclusion,
mono validation, and the 1--30 second duration gate.

The C0 seed-42 teacher is selected by validation and neutrality; C3 is excluded
because its promotion used test evidence and did not transfer to FLEURS.
Deterministic teacher inference is complete for all admitted rows in 238
checksummed chunks. There are zero empty outputs and 328 maximum-length outputs
to calibrate. Prediction SHA-256 is
`86c7decb9334c3c691f79059ae7b7d3b3665ad9b1f4f2e3524e3747e5e74e079`.

Waheed et al. (2025) motivate independent proxy agreement over confidence-only
filtering; Rangappa et al. (2025) motivate character-consensus fixed-budget
selection. The remaining sequence is:

1. calibrate confidence, max-length, repetition, compression, speaking-rate,
  no-speech, and acoustic-consistency diagnostics with nested speaker-grouped
  labelled-training predictions;
2. create a speaker-capped 160-hour shortlist using frozen gates;
3. confirm SeamlessM4T-v2 CC-BY-NC-4.0 suitability, then run Shona proxy
  inference only on the shortlist;
4. compute normalized character disagreement and retain the proxy only if
  held-out AUC is at least 0.70 and exceeds confidence-only AUC;
5. select among 20/40/80-hour pools on grouped development evidence, with fewer
  than 20 passing hours defined as feasibility failure;
6. conduct a blinded probability-weighted audit of 200 accepted and 50
  boundary-rejected items without retuning; and
7. compare gold-only, random-pseudo, confidence-only, and proxy/hybrid students
  with matched quantity, duration, speaker caps, gold exposure, updates, and
  seeds 42--44.

The 160-hour shortlist is the union of independently capped top-80-hour
confidence and diagnostic-model rankings. Each component permits at most 2.0
hours and 378 rows per speaker; the union therefore permits at most 4.0 hours
and 756 rows per speaker. The 2.5% value is the smallest label-blind feasible
candidate; the initial 2% proposal could supply at most 79.12 hours.

Pseudo-label loss weights 0.25/0.50/1.00 form one validation-only sensitivity
grid and are frozen before confirmation. The AUC threshold, pool sizes, audit
sample, and weight grid are study-specific gates or sweeps, not paper-prescribed
constants. Exactly one pseudo-label generation is primary.

## Analysis and decision rules

- Primary metric: corpus WER.
- Secondary metrics: CER, substitutions, deletions, and insertions.
- Curriculum diagnostics: convergence per update/audio hour and WER/CER by SNR
  and duration quartile.
- RQ2 diagnostics: clean WAXAL, paired synthetic corruption, and adaptive
  external FLEURS performance.
- RQ3 diagnostic: transfer efficiency relative to random and native curricula.
- RQ4 diagnostics: retained hours, proxy/confidence AUC, weighted audit error,
  speaker concentration, max-length outputs, repetitions, and downstream WER.
- Effects below 0.5 absolute WER points remain unresolved without consistent
  replication and paired confidence intervals.
- Bootstrap analysis is clustered by speaker where speaker IDs are available;
  Holm adjustment applies to the prespecified B-series factorial family.

## Remaining milestone order

| Order | Work |
|---:|---|
| 1 | Freeze B-series and RQ4 sweep amendments before outcome inspection |
| 2 | Build RQ4 grouped calibration data and 160-hour shortlist |
| 3 | Run and qualify SeamlessM4T-v2 proxy; perform manual audit |
| 4 | Run RQ2 seed-42 parameter screen and freeze B-series settings |
| 5 | Complete B00/BN/BR/BT seeds 43--46; run BNR only if eligible |
| 6 | Select RQ4 pool/weight and complete matched student seeds 42--44 |
| 7 | Open locked WAXAL test only after each decision freezes; report FLEURS adaptively |
| 8 | Resume RQ3 only after a licensed Tshivenda corpus is secured |
| 9 | Consolidate uncertainty analyses, negative results, bibliography, and thesis chapters |

The sequence prevents test-driven branching. GPU time is not the only
constraint: proxy licensing, manual audit, grouped calibration, and complete
reporting are explicit gates. Observed manifests supersede planning estimates.