# RQ2 Factorial Summary

Positive effects mean lower error. This report is descriptive and does not make a promotion decision.

## Validation

### Cell means

| Cell | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| a0_clean_random | 32.9058 +/- 0.4796 | 7.3151 +/- 0.2399 |
| a1_clean_c3 | 32.8368 +/- 0.0000 | 7.1317 +/- 0.0000 |
| a2_waveform_random | 33.0607 +/- 0.2436 | 8.0084 +/- 0.4690 |
| a3_waveform_c3 | 33.0351 +/- 0.3360 | 7.7169 +/- 0.1321 |

### Effects

| Effect | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| augmentation_without_curriculum | -0.1549 +/- 0.3970 | -0.6933 +/- 0.6687 |
| curriculum_without_augmentation | +0.0690 +/- 0.4796 | +0.1834 +/- 0.2399 |
| curriculum_with_augmentation | +0.0257 +/- 0.0971 | +0.2915 +/- 0.4555 |
| augmentation_with_curriculum | -0.1983 +/- 0.3360 | -0.5852 +/- 0.1321 |
| interaction | -0.0434 +/- 0.5624 | +0.1081 +/- 0.6809 |

## Test

### Cell means

| Cell | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| a0_clean_random | 31.0446 +/- 0.2875 | 6.8192 +/- 0.1719 |
| a1_clean_c3 | 30.4564 +/- 0.0000 | 6.4664 +/- 0.0000 |
| a2_waveform_random | 31.2974 +/- 0.2791 | 7.1224 +/- 0.2161 |
| a3_waveform_c3 | 31.3701 +/- 0.2728 | 7.1030 +/- 0.1487 |

### Effects

| Effect | WER mean +/- SD | CER mean +/- SD |
|---|---:|---:|
| augmentation_without_curriculum | -0.2528 +/- 0.0883 | -0.3033 +/- 0.1143 |
| curriculum_without_augmentation | +0.5882 +/- 0.2875 | +0.3528 +/- 0.1719 |
| curriculum_with_augmentation | -0.0727 +/- 0.3858 | +0.0194 +/- 0.2064 |
| augmentation_with_curriculum | -0.9137 +/- 0.2728 | -0.6367 +/- 0.1487 |
| interaction | -0.6609 +/- 0.3609 | -0.3334 +/- 0.2264 |

