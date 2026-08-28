# Whisper Medium Shona Mild SpecAugment Stage 3

## Run identity

| Field | Value |
|---|---|
| Status | Running; preprocessing features in RAM |
| Start time (UTC) | 2026-08-20 18:03:43 |
| tmux session | `whisper-medium-shona-specaug-mild-stage3-6000-seed42` |
| Trainer PID | `2214005` |
| Output | `/ext_data/casper/whisper_runs/output_dir_whisper_medium_shona_specaug_mild_stage3_6000_seed42` |
| Log | `logs/whisper-medium-shona-specaug-mild-stage3-6000-seed42.log` |
| W&B project | `whisper-shona-multilingual` |

## Purpose

Run 6,000 additional optimizer steps from the completed stage-2 model to test
whether very low-rate continued adaptation recovers improvement or confirms
overfitting/saturation.

## Parent result

| Metric | Stage 1 best | Stage 2 parent |
|---|---:|---:|
| Validation WER | 22.2636% | 22.5214% |
| Test WER | 22.7487% | 22.7661% |

Stage 2 did not improve Stage 1. Stage 3 is exploratory and is not motivated by
an improving validation trend.

## Optimization

| Parameter | Value |
|---|---:|
| Additional steps | 6,000 |
| Fresh optimizer | AdamW Torch |
| Learning rate | 5e-7 |
| Warm-up | 100 steps |
| Schedule | Linear to zero over 6,000 steps |
| Batch / accumulation | 2 / 3 |
| Effective batch | 6 |
| Augmentation | Always-on mild SpecAugment |
| Evaluation/logging/checkpoints | Every 1,000 steps |
| Checkpoint retention | 2, including optimizer/scheduler state |

This is another fresh optimizer phase because Stage 2 saved model weights only.
Unlike Stage 2, Stage 3 uses `/ext_data/casper/whisper_runs`, where sufficient
space exists to save optimizer checkpoints and resume safely after interruption.

## Reproducibility

Configuration:
`configs/whisper-medium-shona-specaug-mild-stage3-6000-seed42.json`.

At launch, the run manifest hashes the Stage-2 parent weights and configs, this
trainer/config/dependency set, and every data manifest. Runtime package and GPU
metadata are also captured. W&B uses the approved
`whisper-shona-multilingual` project.

### Captured hashes

| Artifact | SHA-256 |
|---|---|
| Stage-2 parent weights | `8aa08c7e8d67204c31bb6c7fc8b853dd8a8ad068cc6301ed8f70965359d3e1d6` |
| Parent model config | `0eb8319efc33682be177f2cbb3a6b4b3483ce1680aa2c8117ef9d00f099a2861` |
| Parent generation config | `7ea7312efa5a615c1075505fb724a00d63383866e8adda360618fc0b9b02e75f` |
| Stage-3 trainer | `fca58a2bf1db6df3875ff289971ef0012372b413db748338f9017f41ba13ec67` |
| Stage-3 config | `4a0ef181b441148042151cb9b3a47770701f9ddabe31ae164bcb0349284feb06` |
| Dependency pins | `1eefe5d5d2fd57e5bb4b2cd0d709cf0934779fc3410ed988975eba3ab4c7e873` |

## Results

**Pending completion.** The run is active; no result should be reported until
saved metric files exist.

| Metric | Stage 2 | Stage 3 |
|---|---:|---:|
| Validation WER | 22.5214% | Pending |
| Test WER | 22.7661% | Pending |

The run is useful only if test WER remains stable while validation improves.
Continued degradation is evidence to stop extending this augmentation trajectory.