# RQ2/RQ4 Parameter-Sweep Amendment v2

## Status

Frozen before B-series training, RQ4 gate calibration, proxy inference, manual
audit, or student training. Completed RQ1/RQ2 results and completed RQ4 teacher
labels are not reinterpreted by this amendment. WAXAL test and FLEURS cannot
select any value below.

## Why a sweep is permitted

Primary literature supports the method families but does not provide universal
values for Shona Whisper. A small, declared validation-only grid is therefore
used where direct transfer would be unjustified. Every candidate and selection
metric is fixed here; unsuccessful candidates remain reportable.

## RQ2 B-series sweep

- Clean-presentation proportions: $1/3$, $1/2$, and $2/3$.
- Noise severities: 5, 10, and 15 dB, matching the frozen synthetic validation
  range.
- Selection run: seed 42 only.
- Selection endpoint: equal-weight macro-average of DN, DR, and DNR family WER.
- Constraint: clean D0 improvement interval lower bound must exceed -0.5 WER
  points.
- Confirmation: freeze the selected policy and run seeds 43--46 unchanged.

One-third preserves the clean share in Ko et al.'s 0.9/1.0/1.1 speed-expansion
recipe. One-half and two-thirds are symmetric local sensitivity values around
equal clean/corrupt exposure. Ko et al. do not prescribe these probabilities
for noise or RIR. The 5/10/15 dB values are a controlled severity grid, not a
claim of universal optimality.

## RQ4 calibration target

The primary poor-label target is utterance teacher WER greater than 50% on
labelled WAXAL training predictions. WER greater than 40% and 80% are sensitivity
targets. Rangappa et al. use a greater-than-50% high-WER definition; uDistil
reports 20/40/80% targets and its headline AUC comparison at greater than 80%.
The 50% primary target avoids treating this approximately 32%-WER teacher's
ordinary errors as catastrophic.

Five outer speaker-group folds estimate ROC-AUC, PR-AUC, false-acceptance rate,
and accepted-pool micro-WER. Four inner speaker-group folds select diagnostic
models and operating thresholds. The fold counts are study-specific: 113
labelled training speakers leave approximately 22--23 speakers per outer test
fold while preserving enough inner groups for threshold selection.

The primary model comparison is confidence-only against logistic regression on
the frozen teacher diagnostics: mean/minimum/p10 content-token probability,
raw no-speech score, compression ratio, words/characters per second, unique-word
ratio, maximum identical-word run, repeated bigram/trigram fractions,
non-letter fraction, duration, content-token count, and max-length status.
Logistic inverse-regularization candidates are 0.1, 1, and 10, a study-specific
logarithmic sensitivity grid. Inner grouped PR-AUC for the greater-than-50% WER
target selects the value; outer folds estimate performance. Missing diagnostics
are median-imputed within training folds only, and numeric features are
standardized within training folds only.

OpenAI Whisper's compression-ratio 2.4, average-log-probability -1.0, and
no-speech 0.6 defaults are evaluated as candidate diagnostic breakpoints, not
adopted as pseudo-label correctness gates. Their original role is long-form
temperature fallback/silence handling, whereas this study uses deterministic
short-form Hugging Face generation and content-token confidence.

## RQ4 shortlist and speaker caps

The proxy shortlist is capped at 160 hours, twice the largest final-pool
candidate. This leaves rejection capacity after proxy scoring; the earlier
100-hour proposal was incompatible with an 80-hour final candidate. The
160-hour value is a local compute/design constraint, not a Rangappa replication.

The shortlist is the union of two independently capped selections: the best 80
hours by confidence-only score and the best 80 hours by the calibrated
diagnostic model. Each component applies the 80-hour cap below. Their union is
therefore at most 160 hours, 3.2 hours per speaker, and 604 rows per speaker.
Overlap reduces proxy compute without removing the top-80-hour support of either
selector. This union rule is frozen before full calibration results are read.

Each target pool uses a 2% dual per-speaker cap on duration and expected row
count:

| Pool | Duration cap per speaker | Expected rows | Row cap per speaker |
|---:|---:|---:|---:|
| 20 h | 0.4 h | 3,782 | 75 |
| 40 h | 0.8 h | 7,565 | 151 |
| 80 h | 1.6 h | 15,130 | 302 |
| 160 h shortlist | 3.2 h | 30,260 | 605 |

Expected rows use the admitted-pool mean duration of 19.035 seconds. The cap is
a label-blind diversity control guaranteeing at least 50 contributing speakers
by either measure. Random, confidence-only, and hybrid pools must match the
observed speaker-duration distribution.

## RQ4 proxy qualification and student sweeps

- Compare proxy disagreement and confidence on identical outer speaker folds.
- Retain the proxy only if held-out ROC-AUC is at least 0.70 and exceeds
  confidence-only AUC; 0.70 is a local minimum-usefulness gate.
- Final-pool candidates: 20, 40, and 80 hours.
- First select pool size using one seed-42 hybrid-student screen with all other
  settings fixed.
- At the selected pool size, test pseudo-label loss weights 0.25, 0.50, and
  1.00 using seed 42; the centre is equal weighting and the outer values are a
  symmetric multiplicative sensitivity check.
- Freeze pool size and loss weight, then run matched random, confidence-only,
  hybrid, and gold controls with seeds 43 and 44. Seed-42 results remain
  selection-stage evidence.
- Apply Holm adjustment within each three-value sweep family.

If fewer than 20 hours pass frozen quality gates and speaker caps, RQ4 is a
feasibility failure. Values must not be relaxed after inspecting outcomes.

## Evidence corrections

uDistil's reported proxy/SONAR/confidence AUC comparison of 0.82/0.77/0.68
concerns detection of teacher WER greater than 80%; it is not a universal Shona
threshold. Rangappa et al.'s less-than-5% three-system pairwise CER gate and
100-hour experiments are direct facts, but neither value transfers directly to
this two-system Shona setting.

## Primary sources

- Ko et al. (2015), [Audio Augmentation for Speech Recognition](https://doi.org/10.21437/Interspeech.2015-711).
- Ko et al. (2017), [A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition](https://doi.org/10.1109/ICASSP.2017.7953152).
- Waheed et al. (2025), [uDistil-Whisper](https://doi.org/10.18653/v1/2025.naacl-long.296).
- Rangappa et al. (2025), [Efficient Data Selection](https://doi.org/10.21437/Interspeech.2025-2580).
- Gandhi et al. (2023), [Distil-Whisper](https://doi.org/10.48550/arXiv.2311.00430).
- OpenAI Whisper canonical decoding at commit [`90fdc511`](https://github.com/openai/whisper/blob/90fdc51112dc59242201d038a48add97ffd24b5d/whisper/transcribe.py).