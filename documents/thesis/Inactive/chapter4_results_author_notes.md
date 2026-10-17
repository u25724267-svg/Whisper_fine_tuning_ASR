# Results chapter: author notes

The review chapter is `chapter4_results_revised.tex`. It is a standalone LaTeX report, numbered as Chapter 4, with its own preamble and bibliography. To incorporate it into the main thesis, copy the body beginning at `\chapter{Results}`, omit the standalone document wrapper, and merge the four bibliography entries into the thesis bibliography. Five boxed figure instructions identify the plots to supply; the document has no dependency on missing image files.

The chapter covers the completed controlled RQ1 and RQ2 experiments, corrected FLEURS evaluation, zero-shot Base reference, and the completed Medium extension. Earlier partition diagnostics are kept separate. It does not turn unrelated pseudo-labelling work or unexecuted experiment configurations into results for these two research questions.

## Corrections to the pasted draft

- The clean Base validation WER is **32.9058 ± 0.4796**, and test WER is **31.0446 ± 0.2875**, using the item-prediction summaries and sample SD. The draft's validation mean of 33.24 and SD of 1.19 are unsupported by those runs.
- Strict SNR (C2) has prediction-identical seed results: validation/test WER **33.1183/31.5343**. The draft's later SNR seed rows duplicate S2S values instead of the saved C2 results.
- Clean SortaGrad test WER averages **30.9568**, not 30.92. Its test CER is slightly worse than the random baseline.
- Waveform augmentation with random ordering gives validation WER **33.0607**, above the corrected baseline mean. It does not improve the aggregate validation WER.
- The waveform curriculum is **joint SNR–duration (C3)**, not strict SNR (C2). Clean C3 is its appropriate augmentation control.
- LB random gives mean validation/test WER **31.7615/30.2018**. The duplicated waveform averages in the draft's SpecAugment table have been replaced.
- LD has separate saved results; its rows are not identical to LB. Both SortaGrad masking combinations are included.
- LB uses **one frequency mask and one time mask**. LD uses **two of each**. Both have maximum widths 27 bands/100 frames and maximum time proportion **1.0**; neither uses time warping.
- The v2 epoch-two boundary is **step 4,604**, not 4,606. Baseline WER continues to improve after it; C5/C6 WER worsens while loss decreases. The chapter distinguishes these trajectories and evaluates the selected checkpoint rather than assuming the final epoch was selected.
- LB random has the lowest mean Base WAXAL WER, but higher mean FLEURS test error than clean random. The available external results replace the draft's expectation of unmeasured external gains.

## Evidence that supersedes older reports

The saved C3 seed-44 run manifest is dated **25 September 2026** and is accompanied by training, evaluation, and prediction-summary artifacts. Its prediction hashes match C3 seeds 42 and 43. Older reports, including the existing evidence-checked methodology, state that seed 44 was not retrained. The results chapter acknowledges the historical reuse in the waveform aggregate while reporting the later available record. Please reconcile this completion statement in the methodology when merging the chapters; it has not been silently edited here.

Only **four Medium runs** have completed mirrored final results: C0 seeds 42–44 and C1 seed 42. The planned 27-run matrix must not be described as completed. No Medium FLEURS result is supplied.

Full-run peak GPU memory was not present in the inspected summaries. The chapter reports the actual short Medium diagnostic measurements and their scope, rather than treating them as full-training peaks. For a later full-run memory table, export a consistently defined allocated/reserved or device-memory series for every included run. Do not mix these measurement types or substitute the GPU's advertised capacity.

The original C5 seed-42 runtime is retained in the three-run mean. Its much faster exact rerun is disclosed separately, without increasing the accuracy replication count.

## Reproduction and checks

Run `python3 documents/thesis/build_chapter4_results.py` from the repository root (or invoke the script by absolute path). It regenerates nine marked numeric blocks in the chapter plus:

- `chapter4_results_per_run.csv`: 43 completed run records, including available FLEURS outcomes, runtime, loss, checkpoint selection, and source directory.
- `chapter4_results_evidence.json`: the same evidence with epoch histories, rendered table rows, and SHA-256 hashes for the 223 inputs read by the builder.

The builder checks completed three-epoch records, seed/device provenance, evaluation-manifest identity within each run, WER reconstructed from edit counts, and prediction-hash equality for C2/C3. It does not regenerate predictions or certify checkpoint weights that are not present locally. Recheck the narrative after regenerating against changed artifacts; it is intentionally not rewritten by the script.

Static LaTeX checks verified balanced braces/environments, unique labels, resolved internal references/citation keys, and generated table row endings. **No LaTeX compiler is installed in this environment**, so PDF compilation and final page-layout review remain to be done. The references were checked against the primary publication pages for Whisper, Deep Speech 2, SpecAugment, and FLEURS.
