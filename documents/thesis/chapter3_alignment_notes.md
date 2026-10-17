# Aligning Chapter 3 with Chapter 4

`chapter4_final.tex` reports C0, C1, C2, C3, C5, C6, A2, A3 and S2–S5. It excludes
C4, C4R and C7, which Chapter 3 does not describe.

The following changes to `chapter3_final.tex` are needed so that Chapter 3
describes exactly what Chapter 4 reports. They are listed in order of importance.
Configuration values were checked against
`artifacts/experiment_outputs/*/experiment_config.json`.

## 1. SpecAugment section is wrong for S2–S5 (must fix)

`chapter3_final.tex` §Feature-level augmentation describes several things the
reported runs did not use:
- time/frequency masks with probability 0.05 and length 8;
- clean, always-augmented and mixed (0.5) conditions (Table `tab:augmentation_probabilities`).

That describes an exploratory configuration (`configs/whisper-base-shona-specaug-mixed50-*`).
The reported runs used the following (`rq2/s*_specaug_*/experiment_config.json`):

| Policy | Frequency masks | Max width | Time masks | Max width | Applied to |
|---|---|---|---|---|---|
| LB (light), S2/S3 | 1 | 27 mel channels | 1 | 100 frames | every training example (p = 1.0) |
| LD (strong), S4/S5 | 2 | 27 mel channels | 2 | 100 frames | every training example (p = 1.0) |

Neither policy uses time warping, and the maximum time-mask proportion is 1.0.
S2 and S4 use random order; S3 and S5 use SortaGrad.

Suggested replacement for the paragraph and table:

```latex
We apply SpecAugment \citep{Park2019} to Whisper's log-Mel input during training
only. Following the LB and LD policies of \citet{Park2019}, without time warping,
the light policy applies one frequency mask of up to 27 Mel channels and one time
mask of up to 100 frames to every training example, and the strong policy applies
two masks of each type with the same maximum widths. Masked values are set to zero
and masking is disabled for validation, testing and FLEURS. To study the
combination with curriculum learning (RQ2), each policy is trained with random
order (S2: light, S4: strong) and with SortaGrad (S3: light, S5: strong).
```

If you keep the mixed-0.5 / p = 0.05 work, label it explicitly as exploratory
phase only.

## 2. C3 (SNR + duration ordering) is not described (must fix)

C3 is the "best performing curriculum" combined with waveform augmentation (A3), but
Chapter 3 never defines it. Add after §Strict acoustic signal-to-noise ratio:

```latex
\subsubsection{Combined SNR and duration ordering}
This static curriculum combines the two acoustic signals, treating short and clean
recordings as easiest:
\begin{equation}
    s_i^{\text{C3}} = 0.5\,P(d_i) + 0.5\,\bigl(1 - P(\widehat{\text{SNR}}_i)\bigr),
\end{equation}
where $P(\cdot)$ is the percentile rank within the training split. Utterances are
presented in ascending order of $s_i^{\text{C3}}$ in all three epochs, with ties
broken by utterance identifier. Like C2, this ordering is deterministic, so runs
with different seeds are identical.
```

Also name C3 in §Synthetic waveform augmentation. For example: "The curriculum
combined with waveform augmentation was C3, selected as the best-performing
curriculum in the Whisper Base screening." This matches the selection limitation
stated in Chapter 4.

## 3. Hybrid curriculum

Chapter 3 introduces a "hybrid" category (§Curriculum Learning Framework, and the
"static, dynamic and hybrid" bullet in §Experimental Sequence). The hybrid run (C7)
is excluded from Chapter 4, so delete those mentions or reduce them to "static and
dynamic".

## 4. Planned but not run

These items are described in Chapter 3 but have no results. Chapter 4 now lists
them as future work in its Limitations section.

- §Augmentation continued (B00, BN, BR, BT, BNR): there are no runs. Remove the
  subsection and its table, or rewrite it as "planned".
- §Uniform difficulty mixing: the C5/C6 configs record
  `"difficulty_mixing": false`. Remove the subsection or state that mixing was not
  used in the reported runs. The "356 medium and 533 hard examples" figures must not
  be presented as applied.

## 5. Statistical analysis is missing

Chapter 4 uses paired speaker- or prompt-cluster bootstrap intervals, but Chapter 3
never describes them. Add a short subsection under §Evaluation Metrics:

```latex
\subsubsection{Statistical comparison}
Differences between conditions are reported as control minus treatment WER in
percentage points, averaged over the three seed-matched pairs. Uncertainty due to
the evaluation sample is estimated with a paired cluster bootstrap (100{,}000
resamples): WAXAL test speakers (24) or FLEURS sentence prompts (348) are resampled
with replacement, all recordings of a resampled cluster are kept together, and the
same resample is applied to control and treatment before corpus WER is recomputed
from summed edit counts. Percentile 95\% intervals are reported, together with
99.75\% intervals as a Bonferroni sensitivity check for the 20 primary comparisons.
These intervals condition on the trained models and do not include variation
between training runs.
```

## 6. Smaller corrections

- **Experimental Sequence, last bullet.** It says "Evaluating the best performing
  models on an out of domain dataset", but every condition was evaluated on FLEURS.
  Correct it to "all trained models".
- **RQ wording.** Chapter 4 uses a lightly corrected wording ("acoustic data-level
  curriculum learning on the fine-tuning of ASR foundation models on limited Shona
  data"). Use the identical text in both chapters. RQ1 says "acoustic", while C5 and
  C6 are model-based, so consider "acoustic and model-based curriculum learning".
- **Waveform augmentation.** The policy (`configs/rq2-waveform-mild.json`) applies
  noise with probability 0.5 at 10–25 dB SNR, RIR with probability 0.3, and speed
  0.9/1.1 with probability 0.5, in the order speed, RIR, noise. Chapter 3 omits the
  noise probability. Chapter 3 says "pitch-preserving tempo"; the config key is
  `speed`, so check the implementation before keeping "pitch-preserving".
- **Equation label.** `\label{split-objective}` is referenced as
  `\eqref{eq:split-objective}`, so the reference will not resolve.
- **CER equation.** It uses `&=` inside `equation`, which will not compile. Use `=`.
- **Speaker-disjoint split.** "Of the 97 validation speakers..." describes the
  original splits. That is fine, but make sure the 13,807 / 1,715 / 1,671 counts are
  the ones quoted wherever "controlled experiments" are mentioned.
