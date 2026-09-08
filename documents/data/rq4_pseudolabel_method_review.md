# RQ4 Pseudo-Label Method Review and Recommendation

## Recommendation

Use validation-calibrated proxy-agreement filtering. The frozen C0 seed-42
Whisper Base checkpoint supplies training labels. SeamlessM4T-v2, which supports
Shona speech input and text output, is used only as an independent proxy. The
primary quality signal is normalized character disagreement between their
transcripts; Whisper token confidence and acoustic-consistency disagreement are
secondary signals.

This combines the strongest applicable evidence from uDistil-Whisper and the
fixed-budget CER-consensus study while remaining feasible on one RTX 4090.

## Method review

| Study | Method | Evidence | Decision for RQ4 |
|---|---|---|---|
| Waheed et al., uDistil-Whisper, NAACL 2025 | Proxy WER against SeamlessM4T, SONAR similarity, confidence, entropy, NLL, or PESQ | For the true-WER-above-80% target, proxy AUC 0.82 and SONAR 0.77 versus confidence 0.68 | Adopt proxy agreement; do not transfer the 80% target or a proxy threshold without Shona calibration |
| Rangappa et al., Interspeech 2025 | Average pairwise CER among Whisper, Zipformer, and Parakeet; retain CER below 5% | A selected 100 hours matched or exceeded full 1,800--7,500 hour training | Adopt fixed-budget character agreement, but calibrate a Shona threshold |
| Khurana et al., DUST, ICASSP 2021 | One deterministic and three dropout hypotheses; retain when maximum normalized edit distance is below 0.3 | Recovered 60--80% of the supervised-domain gap | Do not reproduce strict DUST: the frozen Whisper teacher has zero dropout |
| Park et al., Improved Noisy Student, Interspeech 2020 | Iterative, development-calibrated confidence filtering plus stronger student augmentation | Strong LibriSpeech gains | Reuse calibration principle only; full multi-generation TPU recipe is infeasible and RQ2 augmentation was negative |
| Likhomanenko et al., slimIPL, Interspeech 2021 | Dynamic pseudo-label cache and online CTC relabeling | Strong 10/100-hour LibriSpeech results | Reject for this study: CTC-specific and expensive in absolute compute |
| Gandhi et al., Distil-Whisper, 2023 | Filter pseudo-labels by WER against available ground truth | A 10% threshold improved distilled training | Not applicable to genuinely unlabeled WAXAL |
| Yang et al., GigaSpeech 2, ACL 2025 | Whisper labels, forced alignment, multidimensional filtering, iterative refinement | Large gains for three low-resource Asian languages | Reuse hard quality gates; do not copy the large iterative pipeline |
| African-language studies | Filtered or iterative semi-supervision in Kinyarwanda, Somali, and South African code-switched ASR | Gains are possible but domain- and iteration-dependent | Use one round first; treat later rounds as conditional |

## Frozen proposed procedure

1. Complete model-independent admission: decoded-PCM deduplication, removal of
   gold/FLEURS overlap and evaluation speakers, mono audio, and 1--30 second
   duration.
2. Use C0 seed 42 as the label teacher because it has the best eligible WAXAL
   validation WER. C3 is not used as teacher because its promotion depended
   partly on test results and it did not improve corrected FLEURS.
3. Generate deterministic greedy Shona transcripts from unaugmented audio and
   record token probabilities, no-speech probability, output length, repetition,
   compression ratio, and speaking rate.
4. Apply calibrated hallucination gates, then create a speaker-capped 160-hour
   shortlist using teacher confidence.
5. Run SeamlessM4T-v2 Shona ASR only on the shortlist. Its CC-BY-NC-4.0 model
   license must be recorded and institutional noncommercial use confirmed.
6. Compute normalized character disagreement after one frozen normalization:

$$
d_i = \frac{\operatorname{EditDistance}(N(\hat y_i^W),N(\hat y_i^S))}
{\max(1,|N(\hat y_i^W)|,|N(\hat y_i^S)|)}.
$$

7. Record Whisper geometric token confidence

$$
c_i = \exp\left(\frac{1}{T_i}\sum_{t=1}^{T_i}\log p_{it}\right)
$$

   and maximum character disagreement under two frozen, label-preserving
   waveform perturbations. This is acoustic consistency, not DUST uncertainty.
8. Calibrate the selector using speaker-cross-fitted predictions from labelled
   training data. The primary poor-label target is utterance WER above 50%, with
   40% and 80% sensitivity targets. Use nested speaker-grouped resampling when
   choosing thresholds and estimating quality. Do not use WAXAL test or FLEURS.
9. Retain Seamless proxy agreement only if it predicts poor teacher labels with
   held-out AUC at least 0.70 and outperforms confidence alone. The 0.70 gate is
   study-specific, not a uDistil constant. Otherwise fall
   back to the calibrated confidence/consistency selector and report the proxy
   qualification failure.
10. Select among final speaker-capped 20-, 40-, and 80-hour pools on grouped
    development evidence. If fewer than 20 hours pass the frozen gates, report a
    feasibility failure rather than relaxing them after seeing results.
11. Conduct a blinded, probability-based manual audit of 200 accepted examples
    plus 50 boundary-rejected examples. Estimate accepted-pool WER with sampling
    weights. The audit is go/no-go only and cannot retune thresholds.

## Student comparison

Use the full gold training set in every condition:

- gold only;
- gold plus a random pseudo-labelled pool from the same eligible source;
- gold plus the confidence-only top fixed-budget pool;
- gold plus the proxy/hybrid-filtered fixed-budget pool.

Match speaker caps and duration distributions across pseudo-label conditions.
Run student seeds 42--44. Hold gold draws and optimizer updates fixed; pseudo
conditions add an auxiliary loss with initial weight 0.5. This estimates the
effect of added pseudo supervision at fixed gold exposure, not equal compute.
The primary contrast is hybrid-filtered versus random pseudo-labels.

Run exactly one pseudo-label iteration. Relabeling, confidence pacing, and loss
weights 0.25/1.0 are permitted only if the filtered condition improves frozen
validation metrics without increasing insertions or repetitions.

## Evaluation caveat

The RQ4 protocol must be versioned and frozen before pseudo-label generation.
WAXAL and FLEURS test results from earlier research questions have already been
inspected, so RQ4 results on them are secondary rather than a pristine
study-level confirmation. No RQ4 threshold, model, or iteration decision may use
either test set.

## Primary references

- [uDistil-Whisper, NAACL 2025](https://doi.org/10.18653/v1/2025.naacl-long.296)
- [Efficient Data Selection, Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-2580)
- [DUST, ICASSP 2021](https://doi.org/10.1109/ICASSP39728.2021.9414299)
- [Improved Noisy Student, Interspeech 2020](https://doi.org/10.21437/Interspeech.2020-1470)
- [slimIPL, Interspeech 2021](https://doi.org/10.21437/Interspeech.2021-740)
- [Distil-Whisper](https://arxiv.org/abs/2311.00430)
- [GigaSpeech 2, ACL 2025](https://doi.org/10.18653/v1/2025.acl-long.135)
- [Kinyarwanda semi-supervised ASR, ICASSP 2024](https://doi.org/10.1109/ICASSP48485.2024.10447447)
- [Five-lingual South African ASR, Interspeech 2019](https://doi.org/10.21437/Interspeech.2019-1325)