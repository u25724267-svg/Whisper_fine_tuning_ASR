# RQ2 SpecAugment LD-Masks Extension Protocol

## Status and purpose

Frozen before LD training or validation results are inspected. This post hoc,
literature-derived extension follows the completed LB-masks experiment and
tests a stronger SpecAugment masking dose under the same fixed training budget.

## Augmentation policy

Park et al.'s LibriSpeech Double (LD) policy is $W=80$, $F=27$, $m_F=2$,
$T=100$, $p=1.0$, and $m_T=2$. This study omits time warping $W$ and transfers
only the masking parameters:

```json
{
  "type": "specaugment_paper_masks",
  "enabled": true,
  "application_probability": 1.0,
  "frequency_mask_count": 2,
  "frequency_mask_max_width": 27,
  "time_mask_count": 2,
  "time_mask_max_width": 100,
  "time_mask_max_proportion": 1.0
}
```

Mask widths are sampled independently for each mask. Masks may overlap. Time
masks are bounded by valid, unpadded frames. Validation, test, and FLEURS audio
remain unmasked. The treatment is called **LD masks without time warping**, not
an exact LD replication.

Relative to LB masks without time warping, LD preserves both maximum widths and
doubles the frequency- and time-mask counts from one to two. Therefore the
primary new contrast isolates masking dose rather than mask width.

## Extension conditions

| ID | Ordering | Policy | Seeds |
|---|---|---|---|
| S2 | Seeded random | LB masks, no warping | 42--44, reused |
| S3 | SortaGrad | LB masks, no warping | 42--44, reused |
| S4 | Seeded random | LD masks, no warping | 42--44, new |
| S5 | SortaGrad | LD masks, no warping | 42--44, new |

The primary LD-dose contrasts are S4-S2 and S5-S3. Secondary contextual
contrasts are S4-S0 and S5-S1 against the existing clean controls, plus S5-S4
for ordering under LD. Any interaction claim requires a joint
speaker-cluster bootstrap; seed means alone are descriptive.

## Fixed training contract

- Pinned `openai/whisper-base` revision
  `e37978b90ca9030d5170a5c07aadb050351a65bb`.
- Speaker-disjoint WAXAL Shona v2 manifests.
- Three epochs; seeds 42, 43, and 44.
- Effective batch size 6.
- Learning rate $10^{-5}$ and 500 warm-up steps.
- FP16, gradient checkpointing, and full determinism.
- Epoch-boundary validation/checkpointing and validation-WER selection.
- Greedy generation with maximum length 225.
- No waveform augmentation and no Hugging Face mask-budget augmentation.
- Existing W&B project `whisper-shona-multilingual` only.

Each S4 config must match its same-seed S2 config except experiment identity,
control metadata, output path, W&B metadata, and the two mask counts. Each S5
config has the analogous requirement against S3.

## Outcomes and decision rules

Primary comparison uses speaker-disjoint validation WER. Report every seed,
mean, SD, and range, followed by matched paired speaker-cluster bootstrap
comparisons for S4-S2 and S5-S3. CER and substitution/deletion/insertion rates
are secondary safety outcomes. WAXAL test and FLEURS cannot select the policy.

LD is practically preferred over LB only if a matched ordering contrast shows:

1. at least 0.5 absolute validation-WER-point improvement;
2. a paired 95% interval whose lower improvement bound exceeds zero; and
3. no material insertion or repetition degradation.

## Primary reference

Park et al. (2019),
[SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://doi.org/10.21437/Interspeech.2019-2680).