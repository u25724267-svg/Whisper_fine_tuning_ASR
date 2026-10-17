# Response to the Chapter 4 review

The review prompted a revision of `chapter4_codex.tex`. The original `chapter4_results.tex` is preserved. This response addresses the pasted feedback; no separate `chapter4_critical_analysis.md` was found in the inspected thesis directory.

## Changes and qualifications

| Reviewer point | Assessment and action |
|---|---|
| Three seeds and small differences | Accepted. Separate observed directions from population claims. Added paired evaluation-cluster intervals for all ten existing within-Medium contrasts on both test corpora. These condition on three trained checkpoints and do not solve training-seed uncertainty. No universal 0.5-pp noise threshold is supported by this design. |
| No figures | Accepted. Added a condition-by-model dot plot showing seed points, means and SD, and a WAXAL/FLEURS scatter showing changes in rankings. Both have PDF and PNG exports and reproducible source code. Point plots avoid implying that truncated bar axes show absolute magnitudes. |
| Arithmetic inconsistencies | The one specific example in the pasted review is a rounding difference, not an arithmetic error. Retained full-precision calculations, added a worked example next to the WAXAL table, and made rounding explicit in captions. All contrast cells were recomputed. |
| Hardware confound in headline | Accepted. The headline now compares the RTX-trained Base and A100-trained Medium execution groups. It does not identify a pure model-size effect. Within-Medium treatment comparisons use A100 controls. |
| Research questions unstated | Accepted. Restated both questions from the methodology at the start. |
| Overloaded dagger | Accepted. Replaced it with D (identical predictions for that evaluation), E (Base C3 external export mean, two distinct predictions across three records), and S (single-seed condition). None denotes significance. |
| Pending results | Retained transparently in this working draft: the user confirms Medium WAXAL zero-shot inference is running. The caption explicitly identifies the draft status. No result was fabricated or silently excluded. Before submission, complete this comparison or explicitly narrow its scope. |
| Excessive hedging | Consolidated inference rules into a dedicated uncertainty subsection and limitations into a separate section. Reworded the conclusion around the observed findings and what the intervals support. Necessary model/device and sampling boundaries remain. |
| Add a paired t-test | Did not adopt it as automatic confirmation with three seed pairs. Added cluster-bootstrap intervals using the actual paired predictions and an exact seed sign-flip sensitivity check. A t-test is a test of the paired mean under distributional assumptions, not a remedy for low replication. |
| Separate conclusion | Accepted. The conclusion now follows the limitations and answers RQ1/RQ2 directly. |

## Arithmetic audit

The table below gives control minus treatment from source means; cross-device comparisons are descriptive. Four decimal places are shown for readability, while calculations used the full source precision.

| Comparison | Control | Treatment | Difference (pp) | Reported (pp) |
|---|---:|---:|---:|---:|
| Base--Medium C0 WAXAL | 31.0446 | 23.7756 | 7.2690 | +7.27 |
| Medium A2--C0 WAXAL | 23.7756 | 23.2974 | 0.4782 | +0.48 |
| Base C1--C0 validation | 32.9058 | 32.6049 | 0.3009 | +0.30 |
| Medium S2--C0 FLEURS | 38.3645 | 37.1635 | 1.2010 | +1.20 |
| Base C3--C0 FLEURS | 58.3099 | 59.2461 | -0.9362 | -0.94 |

In particular, 31.0446 − 23.7756 = 7.2690 pp, reported as 7.27 pp. Subtracting already rounded entries gives 7.26 pp. Replacing the source-derived effect with the latter would introduce double rounding.

## New statistical evidence

The analysis uses 100,000 paired cluster resamples for each corpus: 24 WAXAL speakers and 348 FLEURS sentence prompts. One resample is shared across all checkpoints; errors and reference words are summed within sampled groups, then the three matched-seed corpus-WER differences are averaged. It does not resample utterances as independent, pool seed-by-recording pairs, or present three seeds as a large training sample.

For Medium S2 versus C0:

| Evaluation | Improvement | Unadjusted 95% interval | Approximate Bonferroni 99.75% interval |
|---|---:|---:|---:|
| WAXAL test | 1.12 pp | [0.56, 1.57] | [0.04, 1.86] |
| FLEURS test | 1.20 pp | [0.37, 2.00] | [−0.10, 2.44] |

The wider intervals address the 20 current contrasts (ten comparisons × two corpora), not historical condition selection or unobserved speaker dependence in FLEURS. This is explicitly a post-review exploratory analysis. The external gain therefore must not be presented as unqualified statistical confirmation.

For the three positive S2--C0 seed differences, the two-sided exact sign-flip sensitivity gives p = 0.25 under sign-exchangeability. With only 2³ sign patterns, that is the minimum attainable two-sided value. It does not conflict with positive cluster intervals: the analyses address different uncertainty. More seeds are still needed.

Method references: [SciPy paired t-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) defines the paired-mean question; [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) explains paired resampling and interval construction. The implemented cluster aggregation is documented in `analyse_chapter4_review_codex.py`; the software links do not validate the study's scientific assumptions.

## Reproducibility and validation

- `chapter4_uncertainty_codex.json` contains all 20 intervals, seed differences, sign-flip sensitivity values, and source hashes.
- `analyse_chapter4_review_codex.py` reconstructs the paired analyses; `--figures` creates both figures from the audited results. It requires NumPy and, for plots, Matplotlib. Plotting packages were installed separately under `/tmp/ch4_review_plot`; the training environment was not modified.
- `build_chapter4_codex.py` renders the tables and checks that the interval point estimates match the source-derived effects.
- Commands used: `OPENBLAS_NUM_THREADS=1 .venv/bin/python documents/thesis/analyse_chapter4_review_codex.py`, then `.venv/bin/python documents/thesis/build_chapter4_codex.py`. Figures used `PYTHONPATH=/tmp/ch4_review_plot MPLCONFIGDIR=/tmp/ch4_mplconfig .venv/bin/python documents/thesis/analyse_chapter4_review_codex.py --figures`.
- Numerical checks cover paired IDs/references, manifest and prediction hashes, corpus-WER reconstruction, deterministic results, and table agreement. Figure exports were visually inspected. LaTeX structural checks are recorded at delivery; a full chapter PDF has not been compiled because no LaTeX compiler is installed.
