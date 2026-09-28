# RQ2 SpecAugment LB-Masks Factorial Protocol

## Status and purpose

Frozen before training or validation results are inspected. This is a post hoc,
literature-derived follow-up to the completed RQ1 curriculum study and is kept
separate from the completed waveform A-series.

The experiment tests whether the LibriSpeech Basic (LB) masking policy from
Park et al. transfers to low-resource Whisper Base, and whether its effect
depends on random versus SortaGrad ordering.

## Augmentation policy

The original LB policy is $W=80$, $F=27$, $m_F=1$, $T=100$, $p=1.0$, and
$m_T=1$. This study omits time warping $W$ but transfers the masking parameters:

```json
{
  "type": "specaugment_paper_masks",
  "enabled": true,
  "application_probability": 1.0,
  "frequency_mask_count": 1,
  "frequency_mask_max_width": 27,
  "time_mask_count": 1,
  "time_mask_max_width": 100,
  "time_mask_max_proportion": 1.0
}
```

The resulting policy is called **LB masks without time warping**, not an exact
LB replication. Park et al. found time warping less influential and more
expensive than masking and recommended dropping it first under computational
constraints. The original paper used 960 hours of LibriSpeech, LAS models, and
much longer schedules; transfer to approximately 80 hours of Shona Whisper is
therefore an empirical question.

## Factorial conditions

| ID | Ordering | LB masks |
|---|---|---|
| S0 | Seeded random | No |
| S1 | SortaGrad | No |
| S2 | Seeded random | Yes |
| S3 | SortaGrad | Yes |

S0 reuses C0 seeds 42--44. S1 reuses C1 seeds 42--44. S2 and S3 are newly
trained with seeds 42--44. Waveform augmentation and the Hugging Face
mask-budget SpecAugment policy are disabled in all four cells.

The direct effects are:

- S2 versus S0: LB masks under random ordering;
- S3 versus S1: LB masks under SortaGrad;
- S1 versus S0: SortaGrad without masks;
- S3 versus S2: SortaGrad with masks; and
- $(S3-S1)-(S2-S0)$: descriptive interaction, followed by a joint
  speaker-cluster bootstrap before any inferential interaction claim.

## Fixed training contract

- Model: pinned `openai/whisper-base` revision
  `e37978b90ca9030d5170a5c07aadb050351a65bb`.
- Data: `waxal_shona_speaker_disjoint_v2`.
- Epochs: 3.
- Seeds: 42, 43, and 44, matched across cells.
- Batch size: 6; gradient accumulation: 1.
- Learning rate: $10^{-5}$; warm-up: 500 steps.
- FP16 and gradient checkpointing enabled.
- Epoch-boundary evaluation and checkpointing.
- Checkpoint selection: validation WER only.
- Validation/test audio is never masked.
- Existing W&B project: `whisper-shona-multilingual`; no new project.

## Outcomes and decision rules

Primary selection uses speaker-disjoint validation WER. CER and
substitution/deletion/insertion rates are secondary. Report every seed, mean,
SD, range, runtime, and available memory diagnostics. Use matched paired
speaker-cluster bootstraps for S2-S0 and S3-S1. Test data cannot select the
policy or seed.

The local practical-promotion rule requires:

1. at least 0.5 absolute validation-WER-point improvement in either matched
   augmentation contrast;
2. a paired 95% interval with lower bound above zero; and
3. no material insertion/repetition degradation.

Failure to promote is reported as evidence about this LB-masks transfer under
the fixed three-epoch Whisper Base protocol, not evidence that SpecAugment is
universally ineffective.

## Primary reference

Park et al. (2019),
[SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://doi.org/10.21437/Interspeech.2019-2680).