# RQ2 SpecAugment FLEURS Evaluation Protocol

## Status and purpose

Frozen before LB/LD FLEURS results are computed. This protocol evaluates the
completed paper-style SpecAugment family on corrected Google FLEURS Shona as
external-domain evidence. FLEURS cannot select checkpoints, seeds, masking
policies, or training hyperparameters.

## Evaluation corpus

- Dataset: `google/fleurs`.
- Configuration: `sn_zw`.
- Revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
- Local protocol: `fleurs_shona_corrected_v2`.
- Validation: 393 recordings.
- Test: 925 recordings.
- Training split: not evaluated and not used for adaptation.

The corrected protocol retains numbers, uses unique recording IDs, preserves
prompt-level `source_id`, and has no decoded-audio overlap with WAXAL.

## Checkpoint family

The new queue contains 15 checkpoints:

- clean SortaGrad controls C1, seeds 42--44;
- LB masks without time warping S2 random, seeds 42--44;
- LB masks without time warping S3 SortaGrad, seeds 42--44;
- LD masks without time warping S4 random, seeds 42--44; and
- LD masks without time warping S5 SortaGrad, seeds 42--44.

Clean random controls C0 seeds 42--44 already have corrected FLEURS v2
predictions and are reused without inference. C1 seed 42 uses its
validation-selected `checkpoint-6906`; C1 seeds 43--44 and all S2--S5 runs use
their exported root best model.

Legacy Hugging Face mask-budget experiments are excluded from this confirmatory
family because they used speaker-overlapping WAXAL data and incompatible
masking semantics. They may be evaluated later as a separately labelled
exploratory appendix.

## Fixed inference contract

- Shona transcription language/task prompt.
- Greedy generation with maximum length 225.
- Forced decoder IDs and suppress tokens cleared, matching training evaluation.
- Per-device evaluation batch size 6 and FP16.
- Identical corrected validation/test manifests for every checkpoint.
- Item-level references, hypotheses, WER/CER counts, manifest hashes, and
  prediction hashes retained.
- No new W&B project or training run.

External run directories remain authoritative. After each checkpoint completes,
its FLEURS output and log are hash-verified into the matching repository path
under `artifacts/experiment_outputs`.

## Planned analysis

Report every seed plus mean, sample SD, and range for validation/test WER and
CER. Primary paired contrasts are S2-S0, S3-S1, S4-S2, and S5-S3. Secondary
contextual contrasts are S4-S0, S5-S1, S3-S2, and S5-S4. Factorial interaction
claims require a joint paired bootstrap.

Because FLEURS contains repeated recordings of the same prompt, bootstrap
uncertainty must use the corrected manifest as `--cluster-manifest` and
`source_id` as the cluster field. Holm adjustment applies within the frozen
contrast family. Positive reported effects mean lower error for the candidate.