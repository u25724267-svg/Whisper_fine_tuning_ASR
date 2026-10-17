# Critique of Chapter 4 (`chapter4_codex.tex`, snapshot 16 Oct 2026)

**Overall verdict.** The numbers are carefully audited and the inferential caution is
largely correct. As a *Master's results chapter*, however, it currently reads as an
audit report written for a reviewer, not as an argument written for an examiner.
Three things need to change:

1. The findings are hard to find. The chapter has about 82 instances of "not", 41 of
   "does not / cannot / must not", and no stated headline result.
2. The chapter is organised around the history of the experiments (hardware migration,
   "earlier" claims, "newly available" results) rather than around RQ1 and RQ2.
3. It leaves out the most interesting things the data show: the epoch-wise learning
   dynamics, the SortaGrad epoch-1 collapse, and the decomposition of training time.

Sections 1–3 below cover science, structure, and writing. Section 4 covers figures
(three new ones are already generated). Section 5 gives a proposed chapter outline.
Section 6 lists blocking issues before submission.

---

## 1. Scientific and analytical issues

### 1.1 Missed finding: SortaGrad collapses after epoch 1, then fully recovers *(high priority)*

The evidence JSON (`epoch_evaluations`) shows the following seed-mean WAXAL validation
WER (%). C1, S3, and S5 are the SortaGrad conditions.

| | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Medium C0 (random) | 27.69 | 26.03 | 25.66 |
| **Medium C1** | **72.52** | 26.53 | 25.91 |
| **Medium S3** | **46.04** | 25.37 | 24.45 |
| **Medium S5** | **40.37** | 25.02 | 24.04 |
| Base C0 | 37.44 | 33.53 | 32.90 |
| **Base C1 / S3 / S5** | **43.55 / 45.07 / 47.70** | ≈33.2–33.8 | ≈32.2–32.6 |

A model that has seen only short utterances generalises very badly to the full-length
validation set. It recovers completely once shuffled training begins.

Epoch-1 validation generation also takes far longer: 92–95 min against 53 min for C0
on Medium A100. That points to over-long, repetitive hypotheses running to the
225-token limit, not to truncation. (This is an inference: epoch-1 predictions were
not saved.) It has three consequences:

- **It is the mechanistic answer to RQ1.** SortaGrad's effect is a transient
  length-generalisation failure that the later shuffled epochs erase. That explains why
  the final-checkpoint differences are about 0.04–0.09 pp. The chapter currently reports
  only the null final result.
- **It explains Table `tab:time`.** C1, S3, and S5 take 22–38 min longer on Medium.
  The difference is almost entirely epoch-1 *validation* time, not training cost
  (see Fig. 3 below). The present text ("Adding SortaGrad raises that time to 339.30
  minutes") implies the wrong cause.
- **Masking dampens the collapse.** On Medium, epoch-1 WER falls from 72.5 (C1) to
  46.0 (S3) and 40.4 (S5), so SpecAugment partly protects against short-first ordering.
  This is a real, reportable curriculum × augmentation interaction, and it is directly
  relevant to RQ2. On Base the pattern does not hold (S5 is worse than C1), so it should
  be reported as model-dependent.

The static SNR/duration order (C3) shows the opposite early pattern. It has the
*lowest* epoch-1 validation WER for both models (Base 36.45 vs 37.44; Medium 27.33 vs
27.69), but the advantage vanishes by epoch 3. That is the classic
"curriculum speeds early convergence but not the end point" result from the literature.
Discuss it against Bengio et al. (2009) and the ASR curriculum literature already cited
in Chapter 2.

### 1.2 Most models were still improving at the final epoch

Across the primary runs, 56 of 65 selected the final checkpoint (step 6,906). Only
dynamic Base curricula C5, C6, and C7, plus one Medium C0 and one Medium C1 seed,
selected epoch 2. Under a three-epoch budget, most conditions are therefore not
converged. The chapter mentions the epoch budget only as a limitation (§Limitations,
first paragraph).

It should state explicitly that all comparisons are *fixed-budget* comparisons, and
that curriculum effects (which act mainly early) and regularisation effects (which act
mainly late) are both truncated by this budget. This is also the strongest argument
for why masking helps: S2 and S4 still improve sharply from epoch 2 to 3 (Medium S2
25.18 → 24.55), while C0 flattens (26.03 → 25.66).

### 1.3 Runtime is misreported as "training time" *(high priority)*

Decomposing the recorded training-loop time (Fig. 3) gives:

| Execution | Optimiser steps | In-loop validation generation | Total |
|---|---:|---:|---:|
| Base C0, RTX 4090 | 19 min | 18 min | 37 min |
| Medium C0, RTX 4090 | 93 min | 59 min | 152 min |
| Medium C0, A100 | 158 min | 159 min | 317 min |

About half of the reported "training time" is validation decoding. On A100 the
optimiser steps are 1.7× slower and the generation 2.7× slower than on the RTX 4090.
An A100-SXM4 should not be slower than a 4090 at fp16 for this workload, so an examiner
will ask why. Likely causes are data loading or CPU contention from two concurrent
lanes, network storage (`/ext_data`), or a software-stack difference.

Either diagnose the cause or, at minimum, report the two components separately and say
that the A100 throughput is anomalous. As written, statements like "Medium is 4.08
times as long" and "2.09 times their historical RTX duration" are not interpretable as
model or hardware cost.

### 1.4 C3 (and Base C2) are one model, not three

C3's three seed records have identical predictions. The ordering is fixed
(`order: difficulty_ascending`) and no dropout is configured, so the seed has nothing
left to randomise. Two consequences follow:

- **C0's seed SD is purely data-order variance.** The seeds only change the shuffle.
  That is a useful, clean quantity: order alone moves Medium WAXAL test WER by
  SD ≈ 0.3 pp (range 23.56–24.14). Every clean-curriculum effect in the chapter is
  smaller than this. State it as a result, not just as a footnote about the D marker.
- **"C3 vs C0" compares one model with a mean of three.** The paired intervals are
  computed correctly, but the text should say plainly that C3's evidence base is
  n = 1 training outcome. Confirm the "no dropout" mechanism from the training code
  before writing it.

### 1.5 Contradiction with Chapter 3 about how C3 was chosen *(must fix)*

Chapter 3 (`chapter3_methodology_revised.tex` §Experimental sequence) states that the
augmentation study combined augmentation with "the best-performing curriculum", fixed
"using in-domain validation data". Chapter 4 says WAXAL *test* results contributed to
choosing C3 (Limitations, second paragraph). By validation WER, Base C1 (32.60) beats
C3 (32.84), so the validation rule would have selected SortaGrad.

The two chapters must tell the same story. Either correct Chapter 3, or justify C3 on
the grounds actually used. A selection-on-test admission also weakens the A3 results
and deserves one clear sentence in the threats-to-validity section, not a clause.

### 1.6 The factorial design is incomplete, so "interaction" is only partly estimable

Waveform augmentation is crossed only with {random, C3}. SpecAugment is crossed only
with {random, SortaGrad}. RQ2 asks about "combining curriculum schedules with
augmentation", but each augmentation type is tested with a different single curriculum.
Say this explicitly when framing RQ2, so an examiner does not infer a full 3×3 design.

The interaction contrasts, (A2−A3)−(C0−C3) and (S2−S3)−(C0−C1), are reported only as
point estimates. You already have the paired-bootstrap machinery, so compute intervals
for these two interactions. They are the quantities RQ2 actually asks about.

### 1.7 Asymmetric inference: Base gets no intervals

Chapter 3 describes Base as the primary screening study, yet its effects in
`tab:effects` are "descriptive" with no intervals, while Medium gets full paired
bootstrap intervals. That looks like an analysis performed only where results became
favourable. Where Base prediction files exist, run the same bootstrap. Where they are
missing, say so in the table, with a count.

### 1.8 Practical significance is never discussed

The chapter never says whether 1.1 pp matters. Medium S2 vs C0 is a 4.7% relative
WER reduction on WAXAL and 3.1% on FLEURS. Compare those figures with:

- the order-only noise floor (SD ≈ 0.3 pp);
- the Base→Medium gap (7.3 pp);
- the fine-tuning gain (about 100 pp).

That framing lets the reader judge magnitude without a significance test, and it
replaces many of the hedges.

### 1.9 No error analysis, despite difficulty-based curricula

The curricula are defined by SNR and duration, yet the chapter never reports WER by
SNR or duration bin. The obvious examiner question is: *does C3 or SortaGrad help on
the hard (low-SNR, long) utterances it was designed around, even if the corpus mean
barely moves?* Per-item predictions and manifest metadata exist, so add:

- a figure or table of WER by SNR tercile and duration tercile for C0, C1, C3, S2,
  and S3 (Medium);
- a substitution/deletion/insertion breakdown (summaries already contain S/D/I). This
  matters for the WER–CER disagreement and for the >30 s truncation issue, which
  should show up as deletions;
- three or four qualitative examples (reference vs hypothesis), including one from
  Medium zero-shot, illustrating the repetition behaviour.

Shona is agglutinative, so WER penalises a single wrong affix as a whole word. One
paragraph on the morphology explains why CER and WER rankings disagree (Base C3, Medium
A2, S2). The chapter currently notes the disagreement but never explains it.

### 1.10 Zero-shot baseline

- The Medium WAXAL zero-shot result is pending. It is a blocker (see §6).
- The zero-shot baselines produce repetitive, degenerate output (Medium FLEURS CER is
  100.5%; only 8 of 15,987 words are correct). The "gain from fine-tuning" is therefore
  mostly the gain from *producing Shona at all*. Report it, but say that the fine-tuning
  gain is not comparable with the between-condition effects. Perhaps report CER as the
  primary zero-shot metric.
- State the decoding prompt (forced `sn` language token?) in one sentence. This is the
  first thing a Whisper-literate examiner checks.

### 1.11 Minor technical points

- The exact sign-flip paragraph (p = 0.25 is the minimum attainable) is correct but
  belongs in a footnote or Chapter 3. In the results it reads as an argument with a
  reviewer.
- The memory section (§Memory measurements) spends about 30 lines establishing that
  Medium fits in about 16 GiB. Reduce it to two sentences plus an appendix table.
- Citation mismatch: Chapter 3 cites `Radford2022` and Chapter 4 `Radford2023`. Use
  the shared `.bib` file.
- RQ wording differs between chapters. Chapter 3 says "static and dynamic acoustic
  data-level curriculum learning … ASR foundation models". Chapter 4 says
  "acoustic-property, model-derived, and hybrid curriculum schedules". Use one wording,
  verbatim.

---

## 2. Structural issues

1. **History-driven organisation.** Phrases like "The previous broad statement … must be
   narrowed", "newly available", "resolve several earlier evidence gaps", "no longer",
   "review-stage analysis", and "this revision" refer to drafts the examiner never saw.
   Remove every such reference. The first paragraph of the Conclusion is entirely
   draft-history and should be deleted.
2. **Method content duplicated in Results.** WER/CER definitions, dataset sizes, the
   hyperparameters, SpecAugment parameters, condition definitions, and the bootstrap
   procedure belong in Chapter 3. Results should cross-reference them, keeping at most a
   short table recapping condition codes.
3. **Hardware migration placed before any results.** It is a threat to validity, not a
   finding. Move it to a short "Execution provenance" subsection at the end, or to an
   appendix, with one sentence early on: "All within-Medium comparisons use A100
   training (Appendix X)."
4. **Results organised by corpus, not by RQ.** The RQ1 answer is split between the WAXAL
   and FLEURS sections and has to be reassembled by the reader. Organise by RQ and
   report both corpora together inside each RQ.
5. **The uncertainty framework arrives after it is used.** `tab:effects` shows intervals
   in §4, but the method is explained in §5.1. State the inference rules once, briefly,
   at the start (or in Chapter 3), then just use them.
6. **No summary of findings.** Add a "Summary of findings" table at the end mapping each
   RQ to its answer and evidence strength (e.g. *supported / not supported /
   inconclusive*, with the key interval).
7. **Discussion vs Results.** Check whether your programme expects a separate Discussion
   chapter. If it does, interpretation (mechanisms, literature comparison, practical
   implications) moves there and this chapter gets leaner. If not, add a "Discussion"
   section before Limitations that relates findings to prior work. At present the
   chapter cites only four papers and makes no comparison with the literature.

---

## 3. Writing

- **Lead with the finding, then qualify once.** Current example:
  > "Both WER intervals include zero (Table~\ref{tab:effects}), so these results do not establish a disadvantage, an advantage, or equivalence. They provide no clear WER evidence favouring these clean curricula under the evaluated Medium protocol."

  Suggested replacement:
  > "Neither clean curriculum changed Medium's final WAXAL WER (C1: −0.04 pp, 95% CI [−0.36, 0.38]; C3: −0.04 pp [−0.40, 0.41]); both effects are an order of magnitude smaller than the variation caused by data order alone."
- **One caveat per claim, not per sentence.** Put the standing caveats (hardware
  confound in Base vs Medium; intervals condition on checkpoints; three seeds) in one
  "Interpreting the comparisons" paragraph, then stop repeating them.
- **Remove tooling and implementation vocabulary:** "cell", "lane", "export",
  "prediction hashes", "local mirror", "W&B", "CODEX", file paths in the body, and
  "seed-labelled records". Replace them with plain academic terms ("condition",
  "concurrent experiments", "saved predictions", "training runs").
- **Captions.** Many captions argue ("must not be treated", "should not be
  presented"). Captions should describe what is shown and how to read it. Arguments go
  in the text.
- **Paragraph topic sentences.** Many paragraphs open with a qualification. Each
  paragraph's first sentence should state its point.
- **Tense and voice.** Use the past tense for what was done and observed ("S2 reduced
  WER"), and the present for what tables show ("Table 4.3 shows").

---

## 4. Figures

Existing figures:

- **Fig. `wer-conditions`** (dot plot): good choice of form. Keep it. Consider dropping
  the Base/Medium overlay in favour of within-model panels, because the two models'
  WAXAL ranges (22–24 and 30–32) waste the middle of the axis.
- **Fig. `domain-scatter`**: useful for the "in-domain ranking ≠ external ranking" point.
  Keep it, but add a line of text noting that it has no uncertainty.

New figures, generated by `figures_chapter4_critique.py` from the audited JSON and
saved in `figures/` as PDF and PNG:

| File | Shows | Place in |
|---|---|---|
| `chapter4_effects_forest` | All ten matched contrasts × two corpora: Medium mean with 95% and 99.75% paired intervals, Base point estimate | Replaces or accompanies `tab:effects`; this is the chapter's central inferential figure |
| `chapter4_learning_curves` | Validation WER per epoch, coloured by ordering family | RQ1, §1.1–1.2 above (SortaGrad collapse, C3 early advantage, non-convergence) |
| `chapter4_runtime_decomposition` | Optimiser time vs in-loop validation-generation time | Computational-cost section, §1.3 above |

Recommended but not yet produced (they need per-item manifest joins):

- WER by SNR or duration tercile per condition (§1.9).
- S/D/I composition bar per condition.

LaTeX for the new figures:

```latex
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{chapter4_effects_forest.pdf}
\caption{Matched test-WER improvements (control minus treatment, percentage points).
Filled circles and thick bars: Whisper Medium mean over three paired training runs
with 95\% paired cluster-bootstrap intervals (speakers for WAXAL, prompts for FLEURS);
thin bars: 99.75\% intervals as a 20-comparison sensitivity check. Open diamonds:
Whisper Base means, for which no interval was computed. Positive values favour the
treatment.}
\label{fig:effects-forest}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{chapter4_learning_curves.pdf}
\caption{End-of-epoch WAXAL validation WER, mean over training runs, coloured by
training-order family. Duration-sorted first epochs (SortaGrad) produce a large
temporary increase in validation error that disappears once shuffled training
resumes. Vertical scales differ between panels.}
\label{fig:learning-curves}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=0.85\linewidth]{chapter4_runtime_decomposition.pdf}
\caption{Recorded training-loop time divided into optimiser steps and the three
in-loop validation decoding passes used for checkpoint selection.}
\label{fig:runtime}
\end{figure}
```

---

## 5. Proposed structure

```
4 Results
4.1 Introduction                    – RQs (verbatim from Ch.3), what the chapter reports,
                                      3–4 sentence headline answers
4.2 Interpreting the comparisons    – ≤ 1 page: seeds, D/S markers, pp convention,
                                      what intervals do/don't cover, Base–Medium confound
4.3 Baselines: zero-shot and clean fine-tuning   (Table zero; Base vs Medium gap)
4.4 RQ1: Curriculum learning
    4.4.1 Final-checkpoint effects, both corpora (WAXAL+FLEURS together; forest plot rows 1–2)
    4.4.2 Learning dynamics (learning-curve figure; SortaGrad collapse; C3 early gain;
          non-convergence)
    4.4.3 Additional Base curricula (C2, C4–C7 screen table; cost of C6)
    4.4.4 Error analysis by difficulty bin (if added)
    4.4.5 Answer to RQ1
4.5 RQ2: Augmentation and its combination with curricula
    4.5.1 Augmentation alone (A2, S2, S4 vs C0)
    4.5.2 Combining with a curriculum (A3, S3, S5; interaction estimates + CIs;
          masking dampens the SortaGrad collapse)
    4.5.3 In-domain vs external rankings (domain scatter)
    4.5.4 Answer to RQ2
4.6 Computational cost              (runtime decomposition; memory in 2 sentences)
4.7 Summary of findings             (table: RQ → finding → evidence strength)
4.8 Threats to validity             (selection on test; hardware; truncation >30 s;
                                      3 seeds; incomplete factorial; FLEURS clustering)
Appendix: execution provenance & hardware migration (Table migration, memory table)
```

The main tables move as follows: `tab:waxal` and `tab:fleurs` stay, with the FLEURS
table placed next to WAXAL inside each RQ or merged into one test-only table.
`tab:effects` becomes the forest plot, with the table moved to an appendix.
`tab:migration` and `tab:memory` move to the appendix.

---

## 6. Blocking before submission

1. Complete the Medium WAXAL zero-shot run or remove that comparison. A "Pending" cell
   cannot appear in a submitted thesis.
2. Reconcile the C3-selection description between Chapters 3 and 4 (§1.5).
3. Diagnose or properly report the A100 slowdown (§1.3).
4. Remove all draft-history language (§2.1).
5. Unify RQ wording and the bibliography with Chapter 3.
6. Compile the chapter. No LaTeX compiler was available when the earlier revision was
   produced, so it has never been checked visually.
