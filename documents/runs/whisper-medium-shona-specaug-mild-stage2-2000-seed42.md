# Whisper Medium Shona Mild SpecAugment Stage 2

## Run identity

| Field | Value |
|---|---|
| Status | Completed |
| Start time (UTC) | 2026-08-20 15:42:30 |
| tmux session | `whisper-medium-shona-specaug-mild-stage2-2000-seed42` |
| Trainer PID | `2183672` |
| Output | `output_dir_whisper_medium_shona_specaug_mild_stage2_2000_seed42` |
| Log | `logs/whisper-medium-shona-specaug-mild-stage2-2000-seed42.log` |
| W&B project | `whisper-shona-multilingual` |

## Purpose

Extend the selected three-epoch Medium mild-SpecAugment model by exactly 2,000
optimizer steps and measure whether additional low-rate adaptation improves or
overfits clean validation/test WER.

## Parent model

| Field | Value |
|---|---|
| Parent experiment | `whisper-medium-shona-specaug-mild-seed42` |
| Selected checkpoint | Step 6,000 |
| Parent validation WER | 22.2636% |
| Parent test WER | 22.7487% |
| Local model source | `output_dir_whisper_medium_shona_specaug_mild_seed42` |

The parent checkpoint is model-only. Optimizer and scheduler state do not exist,
so this is not an exact resume. It is a new stage initialized from the selected
weights.

## Stage-2 optimization

| Parameter | Value |
|---|---:|
| Additional optimizer steps | 2,000 |
| Learning rate | 2e-6 |
| Warm-up steps | 100 |
| Scheduler | Linear to zero over 2,000 steps |
| Batch / accumulation | 2 / 3 |
| Effective batch | 6 |
| Precision | FP16 |
| Gradient checkpointing | Enabled |
| SpecAugment | Same always-on mild policy |
| Evaluation | Steps 1,000 and 2,000, plus final validation/test |
| Checkpointing | No intermediate model checkpoint; final model only |

The reduced learning rate is intentionally conservative because the parent model
is already near convergence. A fresh 1e-5 schedule would not reproduce the prior
optimizer trajectory and could destabilize the selected model.

## Reproducibility

At launch, `run_manifest.json` hashes the 3.05 GB parent `model.safetensors`, its
model/generation configs, stage-2 trainer/config/dependencies, and all data
manifests. The same package and hardware metadata used by earlier runs is also
captured.

### Captured hashes

| Artifact | SHA-256 |
|---|---|
| Parent model weights | `8985e28990d3b083fb42953f64211a770726050e54aa107f94a5495b79b55054` |
| Parent model config | `539345790120dad9fd7a8a243db27aeddbc7ddb6ec88e2c7bd0396ca8a7da88f` |
| Parent generation config | `7ea7312efa5a615c1075505fb724a00d63383866e8adda360618fc0b9b02e75f` |
| Stage-2 trainer | `fca58a2bf1db6df3875ff289971ef0012372b413db748338f9017f41ba13ec67` |
| Stage-2 config | `d525a6ba185040d6c19b28cfe7da495735bbf254e826f0d208a0b02acd84d39c` |
| Dependency pins | `1eefe5d5d2fd57e5bb4b2cd0d709cf0934779fc3410ed988975eba3ab4c7e873` |

Configuration:
`configs/whisper-medium-shona-specaug-mild-stage2-2000-seed42.json`.

W&B project: `whisper-shona-multilingual`.

## Results

| Metric | Parent | Stage 2 |
|---|---:|---:|
| Validation loss | 0.23761 | 0.24213 |
| Validation WER | 22.2636% | 22.5214% |
| Test loss | 0.23611 | 0.24019 |
| Test WER | 22.7487% | 22.7661% |

Stage 2 worsened validation WER by 0.2578 points and test WER by 0.0174
points. The test change is negligible, while validation moved in the wrong
direction. This stage did not improve the selected parent model.

Training loss was 0.09998 over 2,000 steps, with a 2,840.07-second training
runtime. Validation WER was 22.6107% at step 1,000 and 22.5214% at step 2,000.

W&B: `https://wandb.ai/dsfsi/whisper-shona-multilingual/runs/dwe3zbav`.

The stage is successful only if validation improvement is accompanied by stable
or improved test WER. A validation-only gain with test degradation is treated as
overfitting.

## Stage-3 decision

A further 6,000-step stage is being run only as an explicit long-adaptation
experiment requested after this non-improving stage. It uses a lower 5e-7
learning rate and is documented separately. Improvement is not expected a priori.