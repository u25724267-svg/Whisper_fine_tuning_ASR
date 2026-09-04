# Shona ASR Confirmatory Experiments Roadmap

Last updated: 2026-09-01

## Scope and priorities

The confirmatory study is organized around four research questions:

1. effects of static and dynamic acoustic data-level curriculum learning;
2. interaction between curriculum and synthetic waveform augmentation for OOD
   generalization;
3. transfer of a Shona-selected curriculum to Tshivenda; and
4. improvement from confidence-filtered pseudo-labeling.

Effort is allocated approximately 65% to curriculum, 20% to augmentation, and
15% to pseudo-labeling. Curriculum is the primary contribution. Augmentation is
limited to one preregistered factorial, while pseudo-labeling proceeds only if
its data-quality gate passes.

## Fixed experiment contract

- Primary screen model: pinned multilingual `openai/whisper-base`.
- Confirmation model: Whisper Medium only after a replicated Base effect.
- Shona protocol: `waxal_shona_speaker_disjoint_v2`.
- Split sizes: 13,807 train, 1,715 validation, and 1,671 test utterances.
- Speaker counts: 113 train, 24 validation, and 24 test, with zero overlap.
- Maximum held-out speaker share: 20% of rows and hours.
- Evaluation and saving: epoch boundaries; all three checkpoints retained.
- Confirmatory seeds: 42, 43, and 44.
- Existing W&B project: `whisper-shona-multilingual`.
- Output root: `/ext_data/casper/asr_experiment_outputs`.
- Final test data is not used for scheduler or hyperparameter selection.
- Every analyzed run requires item-level predictions and artifact hashes.

The established `train_full.py` baseline behavior remains frozen. Curriculum
treatments use a separate runner and shared tested sampler modules.

## Current status

| Item | Status | Evidence or blocker |
|---|---|---|
| Speaker-disjoint Shona protocol v1 | Diagnostic only | Zero overlap, but one speaker supplies 51.1% of validation and one supplies 35.1% of test |
| RQ1-C0 random control, seed 42 | Diagnostic pilot complete | Validation 30.1819%, test 28.2441%; checkpoint 6,000 restored although training reached 6,891 |
| Item-level prediction exporter | Complete | C0 validation/test predictions and hashes are stored with the run |
| Speaker-disjoint Shona protocol v2 | Complete | Maximum speaker row share is 19.5% validation and 17.2% test; zero overlap |
| RQ1-C0 v2 random control, seed 42 | Complete | Validation 32.5262%, test 31.3400%; epoch 3 selected and item predictions exported |
| Matched non-disjoint diagnostic | Complete | Validation 29.2918%, test 28.7307%; reproduces the original baseline within 0.25 points |
| Acoustic metadata for new splits | Complete | All 17,193 v2 rows audited; metadata hash `8e0e82cd604a`; no same-path PCM mismatches |
| Static curriculum runner | Complete | C1-C4/C4R implemented with audited epoch orders |
| Dynamic curriculum runner | S2S and WER implemented | C5 running; C6 validated and ready |
| Corrected FLEURS OOD protocol | Complete and locked | All 3,781 rows retained; numbers restored, IDs unique, and zero WAXAL audio overlap |
| Licensed unlabelled Shona pool | Not selected | Blocks RQ4 pseudo-label training |

## Immediate engineering sequence

1. Complete C4R and C5, then export their item-level predictions.
2. Implement and validate per-example greedy WER capture for C6.
3. Reuse the audited S2S state with the fixed acoustic prior for C7.
4. Compare C0-C7 with speaker-clustered paired bootstrap intervals.
5. Select conditions for seed-43 and seed-44 replication.

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
| C5 repeatability | Exact seed-42 C5 rerun | Repeatability check | Training complete; all checkpoint hashes and Trainer metrics exactly match C5; item export running |
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

After C5 seed-43 item evaluation, run C0 seed 44 and C5 seed 44. Do not spend a
third full run on C3 seed 44 under the unchanged deterministic definition; reuse
the verified C3 artifact when comparing against the seed-44 control and disclose
that C3 has no stochastic seed dimension.

C4R is always run, not only when C4 appears promising. It distinguishes the
effect of difficulty-based selection from the generic effect of a restricted
eligible pool.

The C7 coefficient is fixed before results are observed. If C7 wins, one
0.3-acoustic/0.7-S2S sensitivity run is permitted. No broader weight search is
performed.

The confirmatory acoustic score uses only percentile-ranked duration and SNR
with equal weights. The legacy four-signal score remains archival context and is
not used by C2-C4 or C7. Dynamic C5-C7 begin with C0's seeded random order so
their effects are not confounded with C1's SortaGrad warm-up. The legacy 20%
difficulty-mixing regularizer is not carried forward.

After the seed-42 screen, repeat C0, the strongest static method, and the
strongest dynamic or hybrid method with seeds 43 and 44. Promote a method only
when its validation behavior is consistent, its test effect is practically
meaningful, and paired uncertainty does not support a negligible difference.

### Null-result path

If no curriculum reliably improves upon C0, the RQ1 conclusion is that the
investigated curricula do not materially improve Whisper Base under the
speaker-disjoint low-resource protocol. All conditions and negative results are
retained. RQ2 tests augmentation independently and in interaction with either
the selected curriculum or, if none is selected, the strongest prespecified
candidate. RQ3 remains a transferability test rather than a rescue experiment.

## RQ2 curriculum and synthetic augmentation

Synthetic augmentation means waveform modification: controlled additive noise,
room impulse response convolution, and mild speed perturbation. SpecAugment is
disabled or fixed identically across the factorial.

| ID | Curriculum | Waveform augmentation |
|---|---:|---:|
| A0 | No | No |
| A1 | Yes | No |
| A2 | No | Yes |
| A3 | Yes | Yes |

A0 and A1 reuse matched RQ1 artifacts when every relevant field and manifest
hash agrees. The complete confirmatory factorial contains four conditions over
three seeds; with valid reuse, six newly trained augmented runs remain.

Evaluation includes speaker-disjoint WAXAL, fixed corrupted WAXAL variants, and
corrected FLEURS as natural OOD speech. FLEURS test is opened only after the
augmentation policy and curriculum choice are frozen.

The RQ2 policy is frozen as `rq2-waveform-mild-v1` using Apache-2.0 SLR28
point-source noises and simulated RIRs. Seed-42 materialization passed the full
13,807-row audit. A2 seed 42 ran on W&B run `e4dwoxn2`; SpecAugment was disabled
and clean validation/test audio was unchanged.

A2 seed-42 training completed with Trainer validation/test WER
33.1687/31.5875; item-level export is pending because another user's process
currently occupies the GPU. A3 seed 42 is fully validated and reproduces the
frozen C3 order hash `86ff768edcc8`; it launches only after A2 is fully closed.

## RQ3 Shona-to-Tshivenda curriculum transfer

The curriculum is selected using Shona only and frozen before Tshivenda runs.

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

Pseudo-labeling starts only after the following gate passes:

1. a licensed unlabelled Shona corpus is identified;
2. overlap with all gold splits is removed;
3. at least 20-40 useful hours remain after segmentation and filtering;
4. confidence is calibrated against labelled development speech; and
5. a stratified manual audit of approximately 200 labels is acceptable.

The first comparison contains gold-only, gold plus unfiltered pseudo-labels,
and gold plus confidence-filtered pseudo-labels. Confidence-paced training is
added only if filtering already improves the student. Gold exposure and total
optimizer updates are matched across conditions. If the gate fails, RQ4 is
reported as a feasibility study or removed before final submission.

## Analysis and decision rules

- Primary metric: corpus WER.
- Secondary metrics: CER, substitutions, deletions, and insertions.
- Curriculum diagnostics: convergence per update/audio hour and WER/CER by SNR
  and duration quartile.
- RQ2 diagnostics: clean, corrupted, and FLEURS OOD performance.
- RQ3 diagnostic: transfer efficiency relative to random and native curricula.
- RQ4 diagnostics: retained hours, estimated label error, empty outputs,
  repetitions, and confidence calibration.
- Effects below 0.5 absolute WER points remain unresolved without consistent
  replication and paired confidence intervals.
- Bootstrap analysis is clustered by speaker where speaker IDs are available.

## Indicative schedule

| Period | Work |
|---|---|
| Week 1 | Finish C0; prediction exporter; acoustic metadata; static sampler; C1-C3 |
| Week 2 | C4, C4R, C5-C7 seed-42 screen; select candidates |
| Week 3 | Replicate C0 and selected curricula; perform uncertainty analysis |
| Week 4 | RQ2 factorial and corrected FLEURS OOD evaluation |
| Week 5 | Analysis buffer; Medium confirmation if justified |
| Week 6 | Tshivenda transfer study |
| Week 7+ | Pseudo-label data gate and pilot if a suitable corpus exists |

The schedule prioritizes implementation validation and analysis between runs.
GPU time alone is not treated as the limiting factor. Planning estimates are
approximately 0.6--1.0 GPU hours per Base condition, 2.5--3.0 hours per Medium
condition, and 40--60 GPU hours for the confirmatory program before
pseudo-label generation. Observed run manifests supersede these estimates.