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
- Shona protocol: `waxal_shona_speaker_disjoint_v1`.
- Split sizes: 13,778 train, 1,696 validation, and 1,719 test utterances.
- Speaker counts: 129 train, 16 validation, and 16 test, with zero overlap.
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
| Speaker-disjoint Shona protocol | Complete | Split summary and manifests under `/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v1` |
| RQ1-C0 random control, seed 42 | Running | Central output contains checkpoint 3,000; final metrics are not yet available |
| Item-level prediction exporter | Not started | Required before C0 is analysis-complete |
| Acoustic metadata for new splits | Not started | Required before C2, C3, C4, and C7 |
| Static curriculum runner | Not started | Required before C1-C4/C4R |
| Dynamic curriculum runner | Not started | Required before C5-C7 |
| Corrected FLEURS OOD protocol | Not started | Digits must be retained; current FLEURS runs remain pilots |
| Licensed unlabelled Shona pool | Not selected | Blocks RQ4 pseudo-label training |

## Immediate engineering sequence

1. Allow C0 to finish and preserve its output unchanged.
2. Build a standalone evaluator that saves validation/test IDs, speakers,
   references, predictions, and word/character error counts.
3. Run the evaluator on C0 and hash its prediction artifacts.
4. Compute or verify duration, SNR proxy, active-speech ratio, and audio hashes
   for every speaker-disjoint row.
5. Implement and test the shared curriculum sampler interface.
6. Validate exact coverage, deterministic ordering, score-to-ID alignment,
   epoch transitions, and unchanged validation/test behavior.
7. Launch C1 only after these checks pass.

## RQ1 curriculum screen

All seed-42 conditions use identical data, model revision, optimizer, update
budget, batch size, decoding, and evaluation. Only ordering or pacing changes.

| ID | Condition | Family | Additional dependency |
|---|---|---|---|
| C0 | Seeded random shuffle | Control | Running |
| C1 | Duration SortaGrad | Static ordering | Static sampler |
| C2 | Strict SNR ordering | Static acoustic | SNR metadata |
| C3 | Strict joint SNR-duration ordering | Static acoustic | SNR metadata |
| C4 | Cumulative acoustic tiers | Static pacing | SNR metadata and pacing |
| C4R | Random-subset pacing | Mandatory pacing control | Same stage sizes and exposure as C4 |
| C5 | S2S-margin | Dynamic model-based | Per-example loss capture |
| C6 | WER-margin | Dynamic model-based | Per-example decoding and WER capture |
| C7 | 0.5 acoustic + 0.5 S2S-margin | Hybrid | Static metadata and dynamic scoring |

C4R is always run, not only when C4 appears promising. It distinguishes the
effect of difficulty-based selection from the generic effect of a restricted
eligible pool.

The C7 coefficient is fixed before results are observed. If C7 wins, one
0.3-acoustic/0.7-S2S sensitivity run is permitted. No broader weight search is
performed.

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
GPU time alone is not treated as the limiting factor.