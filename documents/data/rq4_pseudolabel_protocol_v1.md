# RQ4 Pseudo-Label Protocol v1

## Frozen inputs

- Eligible corpus: 60,677 rows, 320.8307 hours.
- Eligible manifest: `/ext_data/casper/asr_data/waxal_shona_unlabeled_admission_v1/eligible.jsonl`.
- Eligible manifest SHA-256: `2305614f4215463e5a93cead01c9e1e58a6cd7dc3efd0fd70c934895968af92a`.
- Teacher: RQ1 C0 random seed 42 v2.
- Teacher directory: `/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2`.
- Teacher weights SHA-256: `a3e4df1999147a9170078cdde23d2c340f85b67f13f650aff7210d47fcec7e9c`.
- Base model revision: `e37978b90ca9030d5170a5c07aadb050351a65bb`.

C0 is selected because validation, rather than test performance, supports its
use as the neutral teacher. C3 is excluded because its promotion partly used
WAXAL test evidence and it did not improve corrected FLEURS.

## Stage 1: deterministic teacher inference

The C0 teacher produces one greedy Shona transcript per admitted utterance from
unmodified audio. Inference uses no sampling, one beam, no timestamps, and the
checkpoint's existing suppression/decoder settings. Each source row is read
from its pinned Parquet coordinate and its decoded PCM SHA-256 must match the
admission record before inference.

For every item, retain:

- raw and BasicTextNormalizer text;
- generated token IDs and selected-token log probabilities;
- geometric mean content-token confidence;
- minimum and tenth-percentile content-token probability;
- EOS generation and maximum-length status;
- raw no-speech-token probability at the start-of-transcription decoder state;
- compression, speaking-rate, and repeated-ngram diagnostics;
- source coordinates, speaker, duration, and PCM hash.

The no-speech value is an uncalibrated model diagnostic, not a probability of
label correctness. Repetition and compression are hallucination proxies, not
proof of hallucination. Confidence is not used alone to declare label quality.

Outputs use immutable 256-row chunks with checksummed completion markers.
Resume accepts only chunks whose row bounds, count, and SHA-256 validate.

## Pilot gates

Before full inference:

1. Verify source IDs, coordinates, duration, and PCM hashes on a stratified pilot.
2. Require identical token IDs for repeated runs and batch-size comparison.
3. Verify special-token exclusion and EOS alignment manually on sample rows.
4. Characterize raw no-speech scores on speech, silence, and noise without
   selecting a threshold.
5. Freeze a batch size with at least 20% GPU-memory headroom.
6. Validate interruption/resume and consolidated hashes.

## Stage 2: shortlist and proxy agreement

After Stage 1 is complete, calibrated hallucination gates create a speaker-capped
160-hour shortlist. SeamlessM4T-v2 is used only as an independent Shona proxy;
its CC-BY-NC-4.0 license must be approved for institutional noncommercial use.
Normalized character disagreement is the primary proxy signal. Thresholds are
calibrated with speaker-cross-fitted predictions on labelled WAXAL training,
never WAXAL test or FLEURS.

The shortlist is the union of independently speaker-capped top-80-hour rankings
from confidence and the calibrated diagnostic model. This guarantees coverage
for the largest candidate pool from each selector while bounding proxy compute
at 160 hours. The exact model features, regularization grid, dual 2% speaker
caps, and selection rules are frozen in the v2 parameter-sweep amendment.

The primary poor-label target is utterance WER above 50%, with 40% and 80%
sensitivity targets. The proxy is retained only if held-out AUC is at least 0.70
and exceeds confidence alone; 0.70 is a local minimum-usefulness gate. Final
speaker-capped pool candidates are 20, 40, and 80 hours. Fewer than 20 passing
hours is a feasibility failure; thresholds must not be relaxed after observing
the outcome. Exact sweep rules are frozen in
`documents/data/rq2_rq4_parameter_sweep_amendment_v2.md`.

## Stage 3: student comparison

Compare gold only, random pseudo labels, confidence-only pseudo labels, and
proxy/hybrid-filtered pseudo labels using matched speaker caps, duration
distributions, gold exposure, optimizer updates, and seeds 42--44. Run one
pseudo-label generation only. The primary contrast is hybrid-filtered versus
random pseudo labels.

## Literature basis

- Waheed et al. (2025), [uDistil-Whisper](https://doi.org/10.18653/v1/2025.naacl-long.296): proxy agreement outperformed confidence-only filtering.
- Rangappa et al. (2025), [Efficient Data Selection](https://doi.org/10.21437/Interspeech.2025-2580): character-consensus filtering under fixed budgets.
- Khurana et al. (2021), [DUST](https://doi.org/10.1109/ICASSP39728.2021.9414299): uncertainty filtering; not directly reproduced because this teacher has zero dropout.
- Park et al. (2020), [Improved Noisy Student](https://doi.org/10.21437/Interspeech.2020-1470): development-calibrated filtering and student perturbation.
- Likhomanenko et al. (2021), [slimIPL](https://doi.org/10.21437/Interspeech.2021-740): dynamic CTC pseudo-labeling, excluded as architecture- and compute-mismatched.

No WAXAL test or FLEURS result may select an RQ4 label, gate, threshold, proxy,
student, loss weight, iteration, or stopping decision.