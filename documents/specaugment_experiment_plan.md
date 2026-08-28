# SpecAugment experiment plan

## Evidence used

- The original SpecAugment paper reports that time and frequency masking drive
  most of the gain; time warping adds little and is omitted here.
- The closest attached Whisper study (about 60 hours per language) reports a
  best policy with time probability 0.05, time length 8, minimum 1 mask,
  frequency probability 0.15, frequency length 15, and minimum 3 masks.
- In Transformers 4.46.3, `mask_feature_min_masks=3` forces three 15-bin spans
  on Whisper's 80-bin input. This is treated as a strong replication policy,
  not the default.
- Other attached Whisper papers either omit exact SpecAugment parameters or
  confound augmentation with data, optimizer, batch, or alignment changes.

## Controlled runs

All runs use the same pinned Whisper Base revision, cleaned WAXAL manifests,
seed 42, three epochs, batches, optimizer, scheduler, decoding, and evaluation.
Only the augmentation block changes.

1. `whisper-base-shona-specaug-control-seed42.json`: no augmentation.
2. `whisper-base-shona-specaug-mild-seed42.json`: 5% time masking with length
   8 and one 8-bin frequency mask.
3. `whisper-base-shona-specaug-paper9-seed42.json`: attached paper's strong
   three-by-15-bin frequency policy.
4. `whisper-base-shona-specaug-mixed50-seed42.json`: each training presentation
  has a deterministic 50% chance of remaining clean and a 50% chance of using
  the mild masks. This is the preferred clean-plus-augmented policy.

Validation and test features are never augmented because Whisper applies these
masks only while `model.training` is true.

## Interpretation

Compare validation and test WER against the deterministic control. Treat changes
below 0.5 WER points cautiously and repeat promising policies with additional
seeds. The current WAXAL v1 test has major speaker overlap, so final claims need
speaker-disjoint v2 training and evaluation.