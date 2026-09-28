# Paper-Style SpecAugment Masks Without Time Warping

## Status

Implemented and validated, but no experiment policy or result is authorized by
this document. Exact mask counts and bounds must be frozen in a versioned
experiment protocol before training.

## Implementation

Configuration type: `specaugment_paper_masks`.

Required fields:

```json
{
  "type": "specaugment_paper_masks",
  "enabled": true,
  "application_probability": 1.0,
  "frequency_mask_count": 2,
  "frequency_mask_max_width": 8,
  "time_mask_count": 2,
  "time_mask_max_width": 20,
  "time_mask_max_proportion": 0.2
}
```

The values above demonstrate the schema and are not a frozen scientific policy.

For each selected training example:

1. apply a fixed number of frequency masks;
2. sample each frequency width uniformly from zero to its configured maximum;
3. apply a fixed number of time masks;
4. sample each time width uniformly from zero to
   `min(time_mask_max_width, time_mask_max_proportion * valid_frames)`; and
5. sample mask start positions uniformly within the valid dimension.

Time masks use the feature attention mask and cannot enter padded frames.
Frequency masks operate across all time frames. Masked values are zero. The
operation runs only in model training mode; validation and test features remain
unchanged. `application_probability` is a separate clean-versus-masked
presentation decision and is not the paper's proportional time bound.

## Relationship to Park et al.

The implementation reproduces fixed mask counts, randomly sampled widths, and
the proportional time-width bound described by Park et al. (2019). It excludes
time warping and therefore must be called a paper-style masking implementation,
not a full exact SpecAugment replication.

The existing `specaugment` type remains unchanged and continues to use the Hugging
Face Whisper mask-budget API with fixed widths.

## Validation

- Unit tests cover valid-frame time bounds, full-axis frequency masks,
  training-only behavior, and invalid configuration rejection.
- Random and SortaGrad speaker-disjoint dry-runs pass the new schema.
- The existing Hugging Face mild policy still passes its dry-run.
- A real Whisper Base hook test confirmed paper-mask dispatch, training-only
  behavior, and no masking beyond valid time frames.

Primary source: Park et al. (2019),
[SpecAugment](https://doi.org/10.21437/Interspeech.2019-2680).