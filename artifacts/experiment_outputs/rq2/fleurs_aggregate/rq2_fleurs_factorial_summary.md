# RQ2 Factorial Summary

Positive effects mean lower error. This report is descriptive and does not make a promotion decision.

## Fleurs_Validation

### Cell means

| Cell | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| a0_clean_random | 56.9423 +/- 1.2466 | 20.1865 +/- 1.6587 |
| a1_clean_c3 | 59.6219 +/- 0.0000 | 21.3928 +/- 0.0000 |
| a2_waveform_random | 56.9526 +/- 0.7179 | 19.3401 +/- 1.3252 |
| a3_waveform_c3 | 57.0397 +/- 0.9363 | 19.5306 +/- 0.9638 |

### Effects

| Effect | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| augmentation_without_curriculum | -0.0102 +/- 1.7746 | +0.8464 +/- 0.6624 |
| curriculum_without_augmentation | -2.6796 +/- 1.2466 | -1.2063 +/- 1.6587 |
| curriculum_with_augmentation | -0.0871 +/- 1.3244 | -0.1905 +/- 1.2059 |
| augmentation_with_curriculum | +2.5822 +/- 0.9363 | +1.8622 +/- 0.9638 |
| interaction | +2.5925 +/- 2.5612 | +1.0158 +/- 0.7083 |

## Fleurs_Test

### Cell means

| Cell | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| a0_clean_random | 58.3099 +/- 0.5884 | 18.9298 +/- 0.2582 |
| a1_clean_c3 | 59.2294 +/- 0.0000 | 20.0770 +/- 0.0000 |
| a2_waveform_random | 59.4879 +/- 1.4715 | 20.0350 +/- 0.4211 |
| a3_waveform_c3 | 60.1009 +/- 2.0391 | 20.5580 +/- 1.5958 |

### Effects

| Effect | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| augmentation_without_curriculum | -1.1780 +/- 1.9919 | -1.1052 +/- 0.6205 |
| curriculum_without_augmentation | -0.9195 +/- 0.5884 | -1.1472 +/- 0.2582 |
| curriculum_with_augmentation | -0.6130 +/- 0.8413 | -0.5230 +/- 1.7754 |
| augmentation_with_curriculum | -0.8715 +/- 2.0391 | -0.4810 +/- 1.5958 |
| interaction | +0.3065 +/- 1.0393 | +0.6242 +/- 1.7134 |

