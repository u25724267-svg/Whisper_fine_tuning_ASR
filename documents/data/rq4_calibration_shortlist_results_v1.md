# RQ4 Calibration and Proxy Shortlist Results v1

## Labelled calibration corpus

- Rows: 13,807.
- Speakers: 113.
- Hours: 79.762.
- Corpus WER/CER: 21.25% / 4.78%.
- Utterances above 40/50/80% WER: 1,430 / 606 / 116.
- Empty/max-length teacher outputs: 0 / 61.
- Prediction SHA-256: `8a7d5bd44f3f9310a24f1e48886f79ba2f0c01f2ad4d485958875101e1e2817a`.

## Nested speaker-grouped calibration

Five outer and four inner speaker-group folds were used. Logistic regularization
$C=0.1$ was selected in every outer fold and by the full inner analysis.

| Poor-label target | Score | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| WER >40% | Confidence only | 0.7600 | 0.3827 |
| WER >40% | Diagnostic model | 0.7746 | 0.4044 |
| WER >50% | Confidence only | 0.7738 | 0.2830 |
| WER >50% | Diagnostic model | **0.8086** | **0.3676** |
| WER >80% | Confidence only | 0.8042 | 0.0967 |
| WER >80% | Diagnostic model | 0.8705 | 0.5330 |

The primary diagnostic model exceeds both the local 0.70 usefulness gate and
confidence-only performance. This qualifies it for shortlist coverage; it does
not qualify SeamlessM4T-v2, whose disagreement must be evaluated separately on
labelled speakers.

Canonical Whisper breakpoints did not transfer as complete correctness gates.
Compression ratio above 2.4 had high precision (0.866) but low recall (0.096)
for WER >50%. Content mean log probability below -1.0 and its no-speech
combination identified no positives. Max-length status had precision 0.918 and
recall 0.092. These observations support multivariate calibration and prohibit
blind adoption of the original long-form defaults.

## Feasibility correction

The original 2% dual speaker cap could supply at most 79.12 hours and therefore
could not produce either 80-hour component. This was established from admitted
speaker durations and counts. Before any shortlist was produced, the cap was
changed to the smallest evaluated feasible value, 2.5%, whose label-blind
capacity is 91.58 hours. The correction is recorded in the v2 amendment.

## Proxy shortlist

The shortlist is the union of confidence-top-80h and diagnostic-top-80h under
independent 2.5% dual caps.

| Component | Rows | Hours | Speakers |
|---|---:|---:|---:|
| Confidence | 15,016 | 79.9999 | 154 |
| Diagnostic | 15,832 | 79.9996 | 155 |
| Overlap | 11,441 | 62.3074 | 149 |
| Union | 19,407 | 97.6921 | 160 |

The union maximum speaker duration/row shares are 3.55%/3.43%, below the 5%
union bounds implied by two independent 2.5% components. Empty and max-length
outputs are excluded; 328 unlabeled max-length rows were rejected before
ranking.

- Diagnostic model SHA-256: `0cd9f80aee1e1f72faaba55734615581b7b31e05c4069baea867ba730961d5a0`.
- OOF predictions SHA-256: `c39069cc22d02290e0f5255da55cbd2bc2cf6efb9ad91a4324cfc0bc5cff7604`.
- Shortlist SHA-256: `d6e24b0d3b963e6d357bbb0f7003348775f98e827cadf485d9880a2ba9da372f`.

## Next gate

Run SeamlessM4T-v2 first on labelled WAXAL training data, then qualify its
teacher-disagreement score against the same WER >50% target and outer speaker
logic. Only a proxy that reaches ROC-AUC at least 0.70 and exceeds confidence
may be run over the 97.69-hour shortlist. Model download and inference remain
conditional on CC BY-NC 4.0 noncommercial institutional approval.