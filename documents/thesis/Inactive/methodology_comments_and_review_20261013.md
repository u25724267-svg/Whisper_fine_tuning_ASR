# Methodology: Comment Responses, Specific Corrections, and Critical Review

Evidence cut-off: **13 October 2026**. This companion addresses the pasted draft, not merely the earlier repository Markdown chapter. It is intended to support selective revision if you do not adopt the full replacement.

The accompanying standalone chapter is [chapter3_methodology_evidence_checked_20261013.tex](chapter3_methodology_evidence_checked_20261013.tex). It includes its own preamble and author-year bibliography. An independently appearing [chapter3_methodology_revised.tex](chapter3_methodology_revised.tex) was left untouched; do not assume its facts or citation keys match this evidence-checked version.

## 1. How to use these outputs

- The full LaTeX file can be compiled as a standalone report chapter. For integration, use its chapter body with your university template and transfer its bibliography entries into your main bibliography; do not nest `document` environments.
- Citation keys below correspond to the embedded bibliography in the evidence-checked chapter. Your existing bibliography may use different keys, which must be reconciled by work/title, not by guessing from an author's surname.
- Each response gives a correction and an inline explanation. None claims that an untested parameter was empirically optimal.
- Dated experiment status must be refreshed before submission. The Medium queue was not actively progressing at this audit: four runs were complete end to end, one had completed training but not final evaluation/export, and 22 were unstarted in the inspected roots.
- No experiment was restarted, no model was changed, and no scientific result was overwritten during this writing task.

## 2. Corrections that materially change the methodology

These are not stylistic preferences. They affect whether the chapter describes the experiments actually implemented.

| Draft claim or implication | Evidence-checked correction |
|---|---|
| All relevant experiments differ in only one factor | Matched pairs are controlled; some comparisons are factorial, exploratory, post hoc, or test-informed. State which factors are fixed in each comparison. |
| All experiments use three independent seeds | Replication coverage varies. Base C3 seeds 42/43 were byte-identical; seed 44 was not retrained. A1 reused the single distinct C3 result against stochastic comparators. The exact C5 seed-42 rerun also added no independent observation. Identical reruns establish repeatability, not independent replication. |
| Controlled execution used both 4090 and A100 | RTX 4090 is recorded. An A100 scheduler request is not execution evidence. Supply run manifests before attributing controlled results to A100. |
| Source WAXAL collection totals equal local experiment totals | Distinguish publication/release totals from measured local raw, cleaned, trimmed, and re-partitioned hours. |
| All local files last at least 15 seconds | The paper states this collection description, but local records include shorter audio. Attribute the statement to the provider and disclose the difference. |
| Labels exist for exactly 10% of local Shona hours | The paper's 10% is an overall collection statement, not established as the local Shona split fraction. |
| Letters, digits and apostrophes were removed | They were retained. The sentence reverses the actual normalisation operation. |
| Held-out v2 audio was never trimmed | Trimming preceded repartitioning. Final v2 validation/test contain 58/38 previously trimmed original-training records. |
| No PCM matches means no leakage | It means no exact canonical representation matches under the stated audit; it does not exclude near duplicates, speaker overlap, sample-rate changes, or pretrained exposure. |
| WAXAL and corrected FLEURS normalisers are literally identical | They share a purpose but differ in Unicode categories and apostrophe mapping. Within-corpus model comparisons use identical frozen references. |
| Zero-shot was run before all fine-tuning | The completed Base zero-shot baseline was added retrospectively; no completed Medium zero-shot output was found. |
| Controlled C4 trains on 4,402/8,804/13,206 rows | Those describe an exploratory filtered subset. Controlled C4 has 4,603/9,205/13,807 eligible items and 13,807 presentations in every epoch. |
| Controlled dynamic curricula use 20% difficulty mixing | Controlled C5/C6/C7 explicitly disable mixing. |
| WER normalisation determines curriculum order | C6 sorts raw WER; min-max WER is an audit field. Other strategies use different scaling. |
| A-series is exact Ko-style speed augmentation | It uses pitch-preserving tempo, not resampling-based speed; probabilities and composition are local choices. |
| B-series training was run | Preparation exists; no B-series training artifacts were evidenced. Use prospective wording. |
| Evaluation loss is the main transcription-quality endpoint | WER is primary, CER secondary; teacher-forced evaluation loss is diagnostic. |
| All 27 Medium experiments have completed | Four complete, one training-complete/finalisation-incomplete, 22 not evidenced as started at this cut-off. |

## 3. Responses to each supervisor comment

**Replication wording to retain wherever seed counts are discussed:** The Base C3 control has no observed stochastic seed dimension: seeds 42 and 43 reproduced identical artifacts, seed 44 was not retrained, and A1 reused the verified C3 predictions. Comparisons against three stochastic controls therefore share the same trained C3 result. A reported zero C3 standard deviation is structural, not proof of negligible uncertainty. An exact C5 seed-42 rerun is likewise a repeatability check, not another independent observation. The audit trail is [the RQ1 final analysis](../rq1_final_analysis.md).

### C01. Research design: why evaluate unseen examples?

**Suggested replacement:**

> Models were evaluated on recordings excluded from supervised parameter updates because performance on training recordings would conflate recognition with memorisation. The principal held-out sets were also speaker-disjoint, allowing assessment on voices absent from fine-tuning rather than merely new recordings of familiar speakers.

**Do not imply:** every record remained wholly uninspected throughout corpus curation, or speaker disjointness proves domain independence.

### C02. “We included data augmentation”: incomplete independent variables

**Suggested replacement:**

> The curriculum comparisons manipulated presentation order or eligibility. The augmentation studies crossed a specified ordering with clean or transformed training inputs. The later Medium extension changed the pinned pretrained model variant within each otherwise matched condition. WER was the primary dependent variable, with CER and validation loss serving distinct secondary roles.

Include the C0-C7/C4R, A0-A3, and S0-S5 cell definitions. “Augmentation” without its type, application frequency, and clean control is not an operational variable.

### C03. Why WER, CER, and evaluation loss?

**Suggested replacement:**

> WER was selected because it directly measures substitutions, deletions, and insertions in generated word sequences. CER complements it by resolving partial spelling differences within a word, which is useful where a small morphological difference counts as a whole-word error. Validation cross-entropy measures the probability assigned to reference tokens under teacher forcing and was retained as an optimisation diagnostic rather than a substitute for recognition error.

Avoid repeating “main dependent variable” and later declaring evaluation loss the main quality measure.

### C04. Why two GPUs? CUDA and memory terminology

**Suggested replacement supported by the inspected records:**

> Recorded controlled runs used an NVIDIA GeForce RTX 4090 with 24 GB of advertised VRAM and a PyTorch build supporting CUDA 12.4. Device and software versions were stored per run because computational cost can depend on hardware. An A100 was mentioned in legacy scheduler material, but the inspected controlled-run provenance does not establish its use; no controlled timing result is attributed to it here.

“CUDA 12.4 driver” confuses the toolkit/runtime interface with the driver version. If A100 runs must remain in the thesis, identify their experiment IDs, GPU memory, driver/runtime, and role before explaining why they used different hardware. Do not invent an allocation story.

### C05. Why NeMo Speech Data Explorer?

**Suggested replacement:**

> NVIDIA NeMo Speech Data Explorer (SDE) was used because it links dataset statistics with waveform, spectrogram, audio playback, and transcript inspection. This enabled suspicious duration or transcript-density records to be reviewed in context. The numerical cleaning rules were executed separately and recorded in manifests, so the result did not depend on an undocumented interactive filtering session.

Cite the actual [SDE tool documentation](https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer). Do not assume the unresolved `Bakhturina2021` entry is a reference for this particular tool.

### C06. Replace “materialized the audio source files”

**Suggested replacement:**

> The acquisition script saved each available recording as a local audio file and wrote its path and metadata to a JSONL manifest, preserving the source encoding rather than deliberately re-encoding the audio at acquisition.

Define JSONL as one independent JSON object per line. Later waveform augmentation *does* decode and write new FLAC audio, so do not generalise “preserved encoding” to the entire pipeline.

### C07. What is SoundFile, and why frame-derived duration?

**Suggested replacement:**

> SoundFile, a Python package built on libsndfile, was used to inspect the recording. Duration was calculated from its frame count divided by sample rate so that manifest duration described the actual local file instead of relying solely on provider metadata.

Use “frame count and sample rate,” not “frame counts and sample.” The acquisition implementation is outside this repository at `/home/casper/Speech/convert_waxal_to_sde.py`; its presence does not establish a pinned historical Hub revision.

### C08. Define character and word rates

**Suggested replacement:**

> Character rate is the supplied transcript's Python string length divided by audio duration in seconds; this includes spaces and punctuation at the cleaning stage. Word rate is the number of whitespace-separated tokens divided by duration. They are transcript-density diagnostics, not direct measurements of how quickly speech sounds were articulated.

Use `r_i^{\mathrm{word}}` consistently, not alternately `word` and `words`. Explain that mismatched audio/text and wrong durations can inflate rates, while fast but valid speech can also be flagged.

### C09. Why reject duration below one second?

**Suggested replacement:**

> A one-second minimum was adopted as a conservative corpus-screening rule because very short clips can provide little supervised content and make duration-normalised diagnostics unstable. It was a local engineering threshold, not a validated lower bound on meaningful Shona speech.

Crucially, the saved rejection records show no observed rejections attributable to this duration rule in the inspected export. Do not imply that many sub-second recordings were actually removed.

### C10. Why rates 1-25 characters/s and at most 4 words/s?

**Suggested replacement:**

> The fixed rate bounds operationalised transcript-density anomalies identified during inspection: unusually high rates can indicate duration or audio/text alignment problems, and very low rates can indicate silence or missing transcription. The thresholds were not validated as universally optimal Shona limits; therefore the rejected-record audit was retained and error rates are interpreted for the resulting curated population.

Actual passing inequalities are `duration >= 1`, `1 <= character_rate <= 25`, and `word_rate <= 4`. All 392 rejected rows triggered both upper-rate conditions. Do not replace this observed fact with a speculative distribution of rejection reasons.

### C11. Why apply cleaning to all splits?

**Suggested replacement:**

> Applying the same fixed rules to the three original partitions defined a consistent annotation-quality population and avoided comparing differently curated reference sets. This also restricts the scope of evaluation: results apply to retained examples, not the entire unfiltered release. Because corpus inspection informed these rules, preparation is treated as a development-stage curation step rather than evidence that held-out corpus properties were never inspected.

Uniform preprocessing is not permission to tune thresholds until test WER improves.

### C12. Why the 40 dB activity threshold?

**Suggested replacement:**

> A frame was active when its RMS exceeded or equalled one hundredth of the recording's maximum frame RMS, equivalent to a relative amplitude difference of -40 dB. The relative comparison accommodates recording-level variation, but it remains a heuristic: quiet speech, background noise, and reverberation can affect the decision.

RMS is root-mean-square amplitude; dB is a logarithmic ratio. Say “40 dB below the peak-frame RMS,” not an absolute “40 dB sound level.” The value is a local setting, not an empirically established optimum.

### C13. Why 100 ms frames, two-second silence gate, and 250 ms padding?

**Suggested replacement:**

> The 100 ms frame provided a coarse energy measurement for the audit. Trimming was restricted to tails of at least two seconds so that short natural pauses were less likely to trigger intervention, and 250 ms was retained after the final detected active region to reduce the risk of clipping speech at an uncertain boundary. These values were fixed engineering choices rather than the result of a parameter-optimisation study.

Add the chronology: 530 original-training replacements preceded re-partitioning; 434/58/38 occur in final training/validation/test. The original split labels and final v2 labels are not interchangeable.

### C14. Why FLAC?

**Suggested replacement:**

> Trimmed waveforms were stored as new Free Lossless Audio Codec (FLAC) files to reduce storage relative to uncompressed audio without introducing a lossy-codec transformation of the encoded samples. The original files were retained. Lossless encoding does not negate any quantisation or sample conversion performed before writing.

Do not write “no information was ever changed”: trimming itself removes samples.

### C15. Why Unicode NFC?

**Suggested replacement:**

> Unicode Normalization Form C (NFC) was used to compose canonically equivalent character sequences into a consistent representation, preventing visually equivalent references from differing solely because of Unicode encoding.

Cite Unicode Standard Annex 15 (`UnicodeNFC`). NFC is not spell checking, transliteration, or a translation operation.

### C16. Why case correction?

Replace **“case corrected”** with **“case-folded.”**

> Case folding removed capitalization distinctions from the target-text contract, reducing lexical mismatches caused by orthographic convention rather than spoken content.

Case folding is a specified transformation, not evidence that the correct proper-name capitalization was recovered. WAXAL and corrected FLEURS use case folding; the earlier exploratory combined preparation used a different contract.

### C17. Why punctuation replacement, and what was retained?

**Mandatory factual repair:**

> Letters, digits, and canonical apostrophes were retained; other selected punctuation and symbols were replaced by spaces, after which repeated whitespace was collapsed. Space replacement avoided concatenating words that had previously been separated by punctuation.

Your current sentence states the opposite. Add that this is the WAXAL operation; corrected FLEURS uses letter/mark/number Unicode categories and slightly different apostrophe mapping. Do not claim literal implementation equality.

### C18. Why preserve original text?

**Suggested replacement:**

> The original reference was retained in `text_raw` while the processed reference was stored in `text`, allowing the effect of normalisation to be inspected and alternative scoring contracts to be reconstructed without losing the annotator's original string.

Use the actual field name `text_raw`, not `textraw`.

### C19. Why manually review digits?

**Suggested replacement:**

> Digit-bearing references were flagged because a written numeral can correspond to different spoken or written-out forms, creating apparent recognition errors even when the audio is intelligible. Digits were not automatically deleted or verbalised with English rules. The saved audit identified 27 original training records and four original test records containing digits.

Do not claim that all flagged records were manually corrected unless correction records exist. Flagging and completion of review are separate events.

### C20. Why 16 kHz and mono?

**Suggested replacement:**

> Recordings were provided as a single audio channel and resampled to 16,000 samples per second to match the selected Whisper processor's input specification. Native WAXAL recordings were 48 kHz, whereas corrected FLEURS was already 16 kHz; this operation standardised model input rather than increasing recording fidelity.

Use “kHz,” not “khz”; 16 kHz means sample frequency, not bit depth. Do not imply resampling occurred in the initial acquisition script merely because it accepted an argument named sampling rate.

### C21. Explain the log-Mel spectrogram

**Suggested replacement:**

> A short-time Fourier transform represents the frequency content of overlapping waveform windows. A Mel-spaced filter bank combines those frequencies into perceptually spaced bands, and logarithmic scaling produces a time-by-frequency matrix. Whisper Base and Medium receive 80 such channels, padded or truncated to 3,000 time frames corresponding to a 30-second input window.

Add the important limitation: some recordings exceed 30 s and the current references are not realigned after audio truncation. The denominator “79.762 h” is manifest hours, not necessarily retained encoder audio hours.

### C22. Why an attention mask?

**Suggested replacement:**

> An attention-mask array identifies feature frames originating from the recording rather than padding. It is passed through the collator and used to keep time masks within valid audio frames. Its presence does not imply that all operations in the pinned model completely ignore padding.

Frequency masking in the paper-style implementation spans the full time axis, including padded frames. Avoid an absolute statement that “padding is never modified.”

### C23. Clarify the Shona tokenizer setting

**Suggested replacement:**

> The multilingual Whisper tokenizer converts references into subword and control-token identifiers. Setting the language to Shona and the task to transcription establishes the language/task context; it does not create a new tokenizer, limit the complete vocabulary to Shona, or fine-tune the tokenizer itself.

Distinguish this reference-side setting from generation configuration, which also specifies language/task and clears forced decoder IDs and suppression lists consistently.

### C24. Why pad targets?

**Suggested replacement:**

> References contain different numbers of tokens. Padding them to the longest reference within a batch creates a rectangular tensor that can be processed jointly without discarding shorter examples.

Do not conflate variable-length target padding with fixed 3,000-frame acoustic padding.

### C25. How does -100 exclude loss?

**Suggested replacement:**

> Padded label positions were assigned -100, the ignore index recognised by the cross-entropy calculation, so they were excluded when summing and normalising the supervised token loss. The model was not trained to predict -100. These positions are target tokens, not padded acoustic frames.

Clarify the difference between token-mean training loss and summed token NLL/audio-second curriculum score. A beginning-of-sequence removal is conditional; not all special tokens are excluded from supervised loss.

### C26. “We decoded the PCM hashes”: explain and correct

**Suggested replacement:**

> Audio was decoded to samples and converted into a canonical pulse-code modulation (PCM) representation. Samples were clipped, rounded to signed 16-bit little-endian values, and hashed with SHA-256 together with sample rate and channel count. The resulting fingerprint identifies equal canonical sample content even when filenames differ. Hashes themselves were not decoded.

State limits: quantisation, rate-sensitive hashes, cropped/noisy near duplicates, and pretrained exposure. It is incorrect to equate this exact-content audit with all forms of leakage prevention.

### C27. Why a 20% held-out speaker cap?

**Suggested replacement:**

> The first local speaker-disjoint split left one validation speaker supplying about 51.1% of recordings and one test speaker about 35.1%. A 20% cap on both rows and duration was therefore introduced to limit domination of held-out metrics by one voice. The exact cap was a study-specific design choice, not a universal statistical threshold or a WER-tuned optimum.

Both row and hour shares matter; equality at 20% is allowed. The final validation largest shares are 19.48% rows/19.79% hours; test 17.18%/17.35%.

### C28. “Look at this later”: substantiate speaker overlap

**Suggested replacement:**

> The original retained partitions contained substantial speaker overlap: 94/97 validation speakers and 101/110 test speakers also appeared in training, and 1,700/1,711 original test utterances came from training-seen speakers. The locally reconstructed v2 splits had zero overlap and contained 113/24/24 speakers in training/validation/test.

Original speaker overlap is not exact-recording duplication. Label the original overlapping, first local disjoint v1, and local constrained v2 protocols explicitly; do not call them all simply “WAXAL v1/v2” without explaining the local naming.

### C29. How were utterances grouped by speaker?

**Suggested replacement:**

> The three normalised original manifests were pooled and indexed by speaker identifier. Each speaker identifier mapped to the list of all associated utterances. Split assignment operated on these groups, so moving a speaker to validation moved every associated recording with that speaker.

Mention 17,193 records/161 speakers and the uniqueness/non-empty assertions. Grouping by speaker is not grouping by utterance duration or random recording IDs.

### C30. Explain the 50,000 assignments and objective

**Suggested replacement:**

> Starting from sorted speaker identifiers, a generator seeded with 42 produced 50,000 shuffled candidate assignments, each with 24 validation and 24 test speakers. Candidates exceeding the held-out speaker-share caps were rejected. Among the feasible candidates, the assignment minimising the sum of squared relative deviations from the 80/10/10 row and hour targets was retained.

Explain both objective terms: matching count alone can mismatch duration; relative normalisation prevents training's larger absolute size dominating the objective. This is sampled search, not a globally optimal solution. The cap, speaker count, and search budget are label-outcome-independent local choices.

### C31. Why FLEURS?

**Suggested replacement:**

> FLEURS was selected because it supplies labelled Shona read speech from a different collection task and text distribution, enabling a cross-corpus generalisation check after WAXAL-only fine-tuning. This differs from holding out WAXAL speakers and cannot identify which aspect of the domain shift causes a performance change.

Define **Few-shot Learning Evaluation of Universal Representations of Speech**. Retain 2,463/393/925 as preparation counts, but say controlled inference uses validation/test only. Corrected recording IDs are unique while sentence `source_id` repeats. Counts of prompt clusters are 150 and 348. Do not claim speaker-independent FLEURS statistics when speaker IDs are unavailable.

### C32. Why were larger Whisper models dropped?

**Suggested replacement:**

> Base was used for the broad controlled screen because its computational cost permitted more matched conditions and replications within the study budget. Medium was selected for a subsequent capacity extension. Other model sizes remained exploratory and were not included in the repeated controlled matrix; this limits scope rather than demonstrating their inferiority or universal infeasibility.

Do not write “Large cannot fit on the 4090” as a conclusion without an appropriate test. The actual Medium memory probes demonstrated feasibility of the inherited batch-6 configuration.

### C33. Why three epochs?

**Suggested replacement:**

> Three epochs fixed a tractable exposure budget for comparing multiple methods. It was not claimed to be the optimum for every condition. Keeping the same budget prevented a treatment from gaining an advantage solely through extra updates. Epoch-boundary checkpoint selection identifies the best of three observations, not necessarily the best state between them.

Mention 13,807 presentations per epoch, 2,302 batches with batch 6, and 6,906 scheduled update attempts. FP16 skips qualify the term “updates.” C4/C4R preserve presentations despite restricted early eligibility.

### C34. Why 500 warm-up steps?

**Suggested replacement:**

> Five hundred warm-up steps, approximately 7.24% of the scheduled 6,906-step budget, gradually raised the learning rate before linear decay. The value was inherited from the fixed baseline to avoid confounding the curriculum with an optimiser search; no claim is made that it is Shona-optimal.

Citation supports the general optimisation approach, not this precise number unless a directly applicable source says so.

### C35. Why FP16?

**Suggested replacement:**

> FP16 automatic mixed precision reduces memory and exploits GPU arithmetic while retaining higher precision where needed. Gradient scaling helps avoid underflow, but overflow can cause a step to be skipped and the scale reduced. The setting was held constant across matched Base and Medium conditions.

Use `Micikevicius2018` for mixed precision. Do not imply that all weights, optimiser states, and operations are stored or evaluated in 16 bits.

### C36. Why gradient checkpointing?

**Suggested replacement:**

> Gradient checkpointing reduces activation memory by discarding selected intermediate values during the forward pass and recomputing them during back-propagation. It trades extra computation for lower memory demand, allowing the fixed batch contract to be tested on available hardware. The same non-reentrant setting was used in matched runs.

Use `Chen2016`. This is different from saving model checkpoints to disk.

### C37. “Justify parameters”: learning rate, batch, decoding, checkpoints

**Suggested replacement:**

> A learning rate of $10^{-5}$ was retained as a conservative baseline fine-tuning setting to limit abrupt changes to pretrained representations. Batch six, accumulation one, epoch evaluation/saving, and greedy generation capped at 225 tokens were fixed across matched comparisons. Their constancy supports attribution of differences to the studied treatment; it does not establish their global optimality.

Include the table of exact settings, LR schedule, seed rules, and selected-checkpoint criterion. Greedy search is cheaper and avoids a beam-width factor, not inherently guaranteed more accurate. Report additional final validation/test and export passes separately from the three selection evaluations.

### C38. What informed 0.40/0.25/0.20/0.15 weights?

**Correction before justification:** those belong to the exploratory 13,206-row quality-ready protocol, not the current controlled acoustic curriculum.

**Suggested replacement:**

> The earlier four-signal weighted score is retained as an exploratory construction; the located records do not establish an optimisation study for its weights. The controlled joint acoustic score instead assigns equal weight to duration percentile and inverse SNR percentile, a transparent local choice that avoids privileging one component without supporting tuning evidence.

Do not invent literature-derived precision for the earlier weight vector. If that exploratory result must be examined in detail, recover its exact generating code and presentation schedule first.

### C39. Did cumulative epochs see different amounts of data?

**Suggested replacement for the controlled C4 method:**

> Eligible utterances expanded at epoch boundaries from 4,603 to 9,205 to all 13,807. Earlier pools were repeated and shuffled to maintain 13,807 presentations per epoch. Thus the number of distinct examples changed, but the presentation and scheduled-update budgets did not. C4R used the same pacing with a random priority list to distinguish acoustic ordering from generic early restriction.

The 4,402/8,804/13,206 counts must remain separately labelled exploratory. Do not mix the two schedules in one equation or table.

### C40. Why divide S2S NLL by duration?

**Suggested replacement:**

> Summed token NLL tends to grow with the amount of transcript. Dividing by manifest audio duration defines a surprise-per-second score, reducing the direct advantage of short examples in the raw sum. It does not remove dependence on tokenisation or speaking rate, and manifest durations longer than the truncated model window remain a limitation.

The score is **sum NLL / seconds**, not mean token loss / seconds. Use the actual non-padding mask. Do not describe $d_i$ as optionally bytes, token count, or any generic length in this ASR implementation.

### C41. Explain what the S2S equation does, not just symbols

**Suggested replacement:**

> For each target position, the model assigns a probability to the correct next token given audio and preceding reference tokens. Taking its negative logarithm penalises low probability, and summing valid positions produces the utterance's sequence loss. Dividing by seconds yields the curriculum score. Lower scores therefore prioritise examples the current model explains with less surprise per audio second; the ranking from one epoch is applied in the next.

Use `y_{i,t}`, `y_{i,<t}`, and `\theta_{e,b}`. The conditioning text is teacher-forced reference text, not an ambiguous “preceding generated/target” mixture. Scores from different minibatches use different model states.

### C42. Why capture scores during runtime?

**Suggested replacement:**

> Scores were collected during the supervised training forward pass to avoid an additional corpus-wide scoring pass. This efficiency makes them online and path-dependent because parameters change between examples; they are not scores from one common frozen checkpoint. Completed-epoch scores determine the next epoch's ordering.

Use `Karakasidis2022` for the related criteria and explicitly call the implementation an adaptation. Avoid claiming equivalence to the published best uniform-mixing system.

### C43. Why min-max “the transcriptions”?

**Suggested replacement:**

> Min-max scaling was applied to the numerical utterance-WER scores stored for audit, not to transcript strings. Actual C6 ordering used raw WER. When the score range is nonzero, min-max scaling preserves order; it therefore does not justify a claim that this transformation improved the recognition criterion.

Other curricula use training-set percentile ranks; do not extend this min-max description to all scores.

### C44. Briefly explain SpecAugment

**Suggested replacement:**

> SpecAugment perturbs the log-Mel representation rather than changing lexical labels: time masks conceal short temporal regions, frequency masks conceal bands, and the original method also includes time warping. Our controlled LB/LD variants retained masking but omitted warping, so they are adaptations rather than complete replications.

Use `Park2019` and include the LB/LD table. The old 0.05/eight-span pilot is a different implementation, not LB under another name.

### C45. Why training mode only?

**Suggested replacement:**

> Masking was restricted to training to regularise input exposure while leaving the evaluation distribution fixed. Applying random masks during ordinary validation or test would combine a training intervention with a changing evaluation input and complicate comparison.

Do not confuse this with deliberately generated, separately labelled DN/DR/DNR validation conditions in the prospective robustness battery.

### C46. Explain time masking and probabilities

**Suggested replacement for the pilot:**

> Time masking hides contiguous frames across the frequency axis. In the pilot, 0.05 specified a masking budget and eight the span length, with a minimum of one span. These were mild local settings, not an independently validated optimum or a per-utterance masking probability.

**For controlled LB/LD instead:** one/two independent attempts; integer width from 0 to `min(100, valid_frames)`; valid-frame start; no padding-tail entry. The policy's `application_probability=1.0` is separate from sampled span width. Zero-width spans and overlap are permitted.

### C47. Explain frequency masking and justify settings

**Suggested replacement:**

> Frequency masking conceals adjacent Mel bands across time, discouraging reliance on a narrow spectral region. The mild pilot used eight-bin spans with a 0.05 budget and minimum one span; the controlled policies transfer $F=27$ from Park et al., with one LB attempt or two LD attempts. These define literature-informed treatments to be tested rather than guarantees of an optimal Shona masking dose.

An 80-bin input with a mandatory eight-bin span is not simply “exactly 5% masked.” LD doubles counts without widening the maximum mask. Zero-valued log-Mel coefficients are not literally equivalent to acoustic silence.

### C48. Define RIR

**Suggested replacement:**

> A room impulse response (RIR) describes how an impulse propagates through an environment, including delayed reflections. Convolving speech with a normalised RIR approximates reverberant propagation. The A-series retains only the pre-convolution waveform length, whereas later follow-up views preserve tails up to the model-input cap.

Use `Ko2017`. Do not claim the same tail policy for both implementations.

### C49. Clarify the SLR28 subsets

**Suggested replacement:**

> OpenSLR resource 28 provides measured and simulated RIRs and different noise categories. The experiment used its simulated-RIR and point-source-noise subsets. Simulated RIRs are generated room responses; point-source noises represent localised acoustic sources. These are assets used in augmentation, not additional labelled Shona utterances.

Use `SLR28` with `Ko2017`. Do not imply that selecting a noise asset also provides a clean reference or a true natural-recording SNR.

### C50. Why 10-25 dB, probability 0.3/0.5, and factors 0.9/1.1?

**Suggested replacement:**

> The factors and transformation families were literature motivated, while the 10-25 dB noise range and application probabilities defined a fixed moderate study recipe. They were not established as universal Shona optima. The implementation applied pitch-preserving tempo with probability 0.5, RIR with probability 0.3, and additive noise with probability 0.5, recording the selected operations and achieved SNR for each materialised utterance.

Your draft omits the **noise application probability of 0.5**. Specify operation order (tempo, RIR, noise, gain limit), fixed one-view reuse across epochs, possible noise tiling in A-series, FLAC quantisation, and peak limit 0.99. Ko (2015) recommends three speed variants 0.9/1.0/1.1; our one-view, pitch-preserving procedure is not that exact algorithm.

### C51. Why WER and CER? Explain each package's role

**Suggested replacement:**

> WER measures edits in generated word sequences and is used for checkpoint selection; CER gives a complementary view of partial word-form accuracy. Evaluate supplies the training-time WER interface, while JiWER directly computes word/character alignments for the canonical item-level exporter. Tokenizer control symbols are skipped during decoding; reference normalisation occurs during corpus preparation, and the exporter does not silently remove all generated punctuation.

Remove “evaluation loss was the main indicator of transcription quality.” State that corpus ratios aggregate counts before division, that WER can exceed 100%, and that separate evaluation passes can have slightly different scores. Use canonical item-level summaries for comparisons.

## 4. Missing or incorrect unannotated sections

### U01. Complete the SNR and hybrid placeholders

Replace “Test txt” with the implemented method, not a generic discussion:

> The SNR proxy is the capped difference between the 75th-percentile active-frame RMS level and the 20th-percentile all-frame RMS level, using a 40 dB-relative activity detector and a 60 dB ceiling. C2 sorts descending estimated SNR in every epoch. C3 averages duration percentile and inverse SNR percentile. C7 averages that fixed acoustic score and the percentile of the preceding epoch's duration-normalised NLL, after a seeded random first epoch.

Specify identifier/index tie-breaks for static orders and duration/identifier/index for dynamic orders. Do not call the proxy a measurement using separate clean/noise channels.

### U02. Correct uniform difficulty mixing

Remove the mixing subsection from the controlled description, or relocate it to a clearly labelled historical exploration paragraph. Current C5/C6/C7 configs set `difficulty_mixing: false`. The 356/533 figures are not supported as controlled counts. Even apparently simple proportions depend on rounding, tier sizes, whether items are swapped, and whether sampling is with replacement. Do not supply those details from intuition.

### U03. Separate HF mild pilots from paper-style masks

Add the controlled matrix explicitly:

| Cell | Ordering | Masking |
|---|---|---|
| S0 / C0 | Random | None |
| S1 / C1 | SortaGrad | None |
| S2 | Random | LB, F27/count1, T100/count1, p1, no warp |
| S3 | SortaGrad | Same LB |
| S4 | Random | LD, F27/count2, T100/count2, p1, no warp |
| S5 | SortaGrad | Same LD |

The old mild/mixed50 conditions and `paper9` pilot are not equivalent to these policies. “SpecAugment” by itself is insufficient to reproduce the experiment.

### U04. B-series status and superseded protocols

Replace “A second set ... was run” with:

> A mechanism-isolation follow-up was specified and its augmentation inventories and validation views were prepared. At the reporting cut-off, its model-training outputs were not evidenced. The later protocol amendment specifies a seed-42 validation-only sweep and frozen confirmation seeds 43-46 after selection; these planned conditions are not included as completed results.

The authoritative validation battery has eight views with the same 1,715 IDs: clean, noise at 5/10/15 dB, RIR, and combined noise/RIR at 5/10/15 dB. Earlier gain-altered clean and periodically tiled-noise views are superseded and must not be used as if equivalent. See [preparation audit](../data/rq2_followup_validation_v1.md) and [v2 amendment](../data/rq2_rq4_parameter_sweep_amendment_v2.md).

### U05. Add the actual Medium extension

**Paste-ready core:**

> A matched extension was prepared using `openai/whisper-medium` at revision `abdf7c39ab9d0397620ccaea8974cc764cd0953e`. It contains 27 runs: seeds 42, 43, and 44 for C0, C1, C3, A2, A3, and S2-S5. Each run starts independently from the pretrained Medium model and inherits its same-seed Base data, curriculum, augmentation, optimiser, three-epoch schedule, decoding, and checkpoint criterion. Train and evaluation batch sizes remain six with accumulation one. The earlier 2-by-3 memory accommodation was superseded before training to avoid introducing a second experimental factor.

> Bounded clean and LD-mask tests on the RTX 4090 completed two non-skipped AdamW updates and generation evaluation, with approximately 14.235 GiB allocated and 14.994 GiB reserved at peak. Initial FP16-skipped attempts were not misclassified as successful optimiser-state tests. These probes support attempting the exact configuration, not a universal full-run memory guarantee.

> At 13 October 2026, C0 seeds 42-44 and C1 seed42 were complete end to end. C1 seed43 had saved the third-epoch checkpoint but was interrupted during final validation; the other 22 planned runs had no located output directories. No Medium training process was active at inspection. The interruption's cause was not established by a logged traceback. Medium FLEURS and zero-shot evaluations were not evidenced as complete.

Use the full chapter for the longer rationale, provenance checks, and interpretation. A restored process or stale waiting log is not evidence that these counts should be described differently without checking artifacts.

### U06. Add uncertainty and multiple-comparison methods

Specify 10,000 paired resamples, WAXAL speaker clusters, FLEURS prompt clusters, the sign of improvement, and percentile intervals. Keep seed variation separate from evaluation-sample variation. State that the original RQ1 screen was unadjusted and that planned Holm/joint interaction analyses must not be reported as completed without their outputs. Do not call bootstrap support a posterior probability or a nominal confidence interval a family-wise significance test.

### U07. Selection contamination and “confirmatory” language

The recorded research history says WAXAL test evidence contributed to carrying C3 forward, and FLEURS has since been inspected. Preserve that fact:

> These were matched controlled comparisons, but later selections were not wholly independent of held-out evidence. Test-informed selections retain a development/selection role, and later uses of inspected FLEURS are adaptive external evaluation rather than untouched final confirmation.

Do not retrospectively claim validation-only selection for the entire history. A control group alone does not erase selection bias.

### U08. Fix the zero-shot chronology

Base zero-shot outputs exist, but were added retrospectively in September. Replace “The pretrained checkpoint underwent zero-shot evaluation before fine-tuning” with “An untouched pretrained Base reference was additionally evaluated retrospectively under the controlled inference contract.” Restrict the statement to models with actual reports; no Medium zero-shot result was found.

### U09. Fix file-size and hour claims

Use local raw labelled hours (about 99.388), cleaned hours (99.278), and pooled normalised/trimmed hours (98.910) only with their correct stages. Published 99.2/574.2 values may refer to rounded, release-specific or figure-derived statistics; they were not independently reproduced as the local experiment totals. The publication's overall 10% annotation statement must not be used as a calculation of the Shona labelled fraction.

### U10. Distinguish historical v2 from upstream v2

The live dataset card now describes provider-distributed `ASR_v2` splits. Our `waxal_shona_speaker_disjoint_v2` is a local seed-42 constrained partition with the specific counts in the chapter. Do not assume the two are the same because both names contain “v2.”

## 5. Critical assessment of the writing

### 5.1 Main weakness: inconsistent evidential levels

The draft has a useful chronological skeleton and many concrete quantities, but it repeatedly moves from a plausible rationale to an asserted fact without showing whether that fact came from code, an artifact, literature, or a proposed protocol. That is more serious than grammar. The strongest revision principle is: **name the experimental family, specify the operation, give its rationale, and qualify the evidence.**

For example, “we replaced 20% of the easy category” sounds precise, but becomes misleading when the controlled implementation disables that operation. Precision of wording cannot rescue an incorrect experimental referent. Separate exploratory and controlled descriptions before improving prose.

### 5.2 Do not invent post-hoc justification

An examiner may ask why 40 dB, 20%, 0.3, or 500 was chosen. A defensible answer can be “a fixed engineering setting intended to serve a stated purpose, held constant across comparisons, without a claim of optimality.” An invented assertion that prior studies prescribe that exact value is less defensible. Distinguish:

1. **Published mechanism:** masking, SortaGrad, resampling, reverberation, mixed precision.
2. **Local parameter:** exact probabilities, thresholds, rank weights, evaluation frequency.
3. **Measured evidence:** local counts, concentration, successful memory tests, recorded runtimes.
4. **Planned test:** a sweep or replication not yet executed.

Inline rationales should combine the relevant categories without pretending they are the same.

### 5.3 Replace overclaims with bounded interpretations

| Avoid | Prefer |
|---|---|
| “Differences could be attributed to that factor.” | “Matched conditions reduce alternative explanations; remaining threats are described.” |
| “This more accurately represented difficulty.” | “This was more directly aligned with the recognition endpoint, although noisy for short references.” |
| “No leakage was found.” | “No exact canonical-PCM matches and no speaker overlap were found under the stated audits.” |
| “The optimal checkpoint.” | “The lowest-validation-WER checkpoint among the three saved epochs.” |
| “The model starts to overfit at ...” | “Validation error increased between the observed checkpoints; onset within that interval is unresolved.” |
| “Larger model is the only variable.” | “The pretrained model variant is the intended changed modeling factor; architecture and weights change together.” |
| “Three independent repeats.” | “Three seed-labelled runs; independence is qualified where outputs are deterministic or reused.” |
| “GPU time.” | “Measured elapsed runtime,” unless GPU-active time was actually integrated. |

### 5.4 Keep tense tied to completion

Use past tense for executed preparation and completed training. Use present tense for definitions and software behaviour. Use future/prospective wording for B-series and unexecuted Medium cells. A table listing cells is not completion evidence. Put a dated status sentence or table near incomplete work, then update it before submission rather than making the prose permanently ambiguous.

### 5.5 Define operations before notation

Your long “where” lists define symbols but sometimes fail to explain the computation. A better sequence is:

1. A sentence explaining the practical operation.
2. The mathematical expression.
3. Only the symbols not already introduced.
4. One sentence explaining the direction/interpretation.
5. An implementation detail or caveat that affects reproducibility.

For S2S, explain token surprise and summation before listing indices. Keep $d_i$ as seconds throughout. Use a different notation for reference word count and corpus sample count to avoid overloading $N$.

### 5.6 Reduce repetition without removing reproducibility

State the common training contract in one table; refer back to it for each condition. State only changed settings and exceptional details in later subsections. Keep the three writer/hash/durability concepts distinct, but place exact paths, complete commit identifiers, and artifact inventories in an appendix or reproducibility table if the university limits chapter length. Repetition of “all other parameters remained the same” is not a substitute for showing which parameters were checked.

### 5.7 Terminology and grammar repairs

| Original | Correction |
|---|---|
| “experiments where setup” | “experiments were set up” |
| “participants where shown” | “participants were shown” |
| “finetuning” / “finetuned” | “fine-tuning” / “fine-tuned,” used consistently |
| “base” for a control | “baseline”; reserve “Whisper Base” for the model size |
| “matric” | “metric” |
| “formulars” | “formulae” or “formulas,” consistently |
| “wordrate” | “word rate” |
| “readspeech read speech” | “read speech” |
| “signal to noise ration” | “signal-to-noise ratio” |
| “Bernouli” | “Bernoulli” |
| “case corrected” | “case-folded” |
| “proceeding epochs” | “subsequent epochs” |
| “models training loss” | “model's training loss” |
| “may may” | “may” |
| “Nemo/NEMO” | “NeMo” |
| “Fleurs/fleurs” | “FLEURS” |
| “16khz”, “40dbs”, “24GB VRAM of memory” | “16 kHz”, “40 dB”, “24 GB VRAM” |
| “metadata” / “meta data” | “metadata” consistently |

Prefer a clear verb (“saved”, “resampled”, “ranked”, “truncated”) to “utilised” or “implemented” when the specific operation is known. “We used” is not inherently unacademic; clarity matters more than inflated vocabulary. Use British or American spelling consistently; the revised version predominantly uses British spelling.

### 5.8 LaTeX repairs

- `\graphicspath{{{images/}}` has unmatched braces. Correct form: `\graphicspath{{images/}}` with `graphicx` loaded.
- The pasted excerpt starts `\begin{document}` without a class/preamble. Decide whether it is a standalone document or an included chapter. Do not retain nested document environments in a thesis master.
- A report/book thesis normally uses `\chapter{Methodology}`, then sections/subsections. An article-style thesis may deliberately use `\section`; follow the actual template, not mixed hierarchy.
- `\text{CER} &= ...` is invalid alignment inside a plain `equation` environment. Remove `&` or use `aligned`/`align`.
- Use `10^{-5}`, not textual `1×10^(-5)`; use `y_{i,t}` and `\theta` rather than ambiguous parenthesised subscripts or raw Unicode notation inside LaTeX.
- Repeated `tab:placeholder`/`fig:placeholder` labels are not meaningful and can collide. Use unique descriptive names.
- `[H]` needs the `float` package and should not be used merely to force every object into place. Prefer standard floating placement unless the layout requires otherwise.
- `\todo` and `\hl` need their respective packages, and highlighted text can interact badly with citations/math. Resolve review annotations before submission rather than relying on a clean-looking PDF to imply all facts were checked.
- The evidence-checked version omits the unavailable pipeline image. If reinstated, the diagram should distinguish exploratory, controlled, and prospective branches, and show that test/FLEURS do not feed prospective selection.
- Tables should have informative captions, units, and evidential scope. A polished table of unexecuted conditions must still say “planned.”

## 6. Citation corrections and primary-source map

The replacement embeds bibliography entries so it does not depend on the absent local `.bib` file. Convert them to the university style when integrating.

| Subject | Recommended key/source | What it supports, and what it does not |
|---|---|---|
| WAXAL collection | `Diack2026`, [arXiv v3](https://arxiv.org/html/2602.02734v3) | Image prompts, at least 50 topics, collection description, paid local annotators, overall 10% statement. Does not establish exact local counts or enforced local 15 s minimum. |
| Shona resource fields/licence | `WaxalDataset`, [dataset card](https://huggingface.co/datasets/google/WaxalNLP) | Shona provider and provider-specific licence; live card can differ from historical release. |
| Whisper | `Radford2023`, [ICML publication](https://proceedings.mlr.press/v202/radford23a.html) | Pretraining, model architecture/family; use 2022 only when explicitly citing the preprint. Not a source for our selected epoch count or LR optimum. |
| FLEURS | `Conneau2022`, [2022 preprint](https://arxiv.org/abs/2205.12446) | Dataset/benchmark design. Published proceedings can carry different year metadata; cite a consistent version. |
| SDE | `NeMoSDE`, [official tool](https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer) | Tool functionality. The identity/applicability of `Bakhturina2021` was not established from an existing bibliography. |
| General curriculum | `Bengio2009`, DOI 10.1145/1553374.1553380 | Motivation, not local score weights. |
| SortaGrad | `Amodei2016`, [ICML record](https://proceedings.mlr.press/v48/amodei16.html) | Short-first then random strategy; 2015 is preprint year, not the ICML publication year. |
| SNR curricula | `Braun2017`, DOI 10.23919/EUSIPCO.2017.8081267 | Noise-related curriculum motivation, not our percentile proxy as a direct replication. |
| Dynamic criteria | `Karakasidis2022`, [ISCA record](https://www.isca-archive.org/interspeech_2022/karakasidis22_interspeech.html) | Model-loss/WER criteria and curriculum comparison. Replace opaque `Karakasidis1382` only after identifying the actual bibliography entry. |
| SpecAugment | `Park2019`, DOI 10.21437/Interspeech.2019-2680 | Masking family and named LB/LD policies; not our HF mild pilot parameters or exact no-warp Whisper replication. |
| Speed augmentation | `Ko2015`, [ISCA record](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html) | Three speed variants 0.9/1.0/1.1; not exact equivalence to pitch-preserving one-view tempo. |
| Reverberation | `Ko2017`, DOI 10.1109/ICASSP.2017.7953152 | Reverberant augmentation mechanism. |
| Assets | `SLR28`, [OpenSLR](https://www.openslr.org/28/) | Asset types and Apache 2.0 licence. |
| AdamW | `Loshchilov2019`, arXiv:1711.05101 | Decoupled weight decay; not evidence of eight-bit optimisation. |
| FP16 | `Micikevicius2018`, arXiv:1710.03740 | Mixed precision and scaling rationale. |
| Gradient checkpointing | `Chen2016`, arXiv:1604.06174 | Recompute/memory trade-off. |
| Bootstrap | `BisaniNey2004`, DOI 10.1109/ICASSP.2004.1326000 | ASR bootstrap comparisons; our cluster choice follows this dataset's dependence structure. |
| Multiplicity | `Holm1979`, Scandinavian Journal of Statistics 6(2), 65-70 | Step-down correction; not proof it was applied to earlier analyses. |
| Seed variability | `Reimers2017`, DOI 10.18653/v1/D17-1035 | Reporting score distributions; does not make deterministic repetitions independent. |

**Licence discrepancy to retain:** the WAXAL paper uses a general CC-BY-4.0 statement, whereas the live dataset card assigns Digital Umuganda/Shona CC-BY-SA-4.0. Follow the provider-specific distribution terms and preserve the card/release used; obtain clarification before asserting the discrepancy is resolved. Do not silently choose the more permissive wording.

**Unverified bibliography keys:** no original bibliography was supplied, so exact mappings for `Bakhturina2021` and `Karakasidis1382` cannot be guaranteed. The alternative references above are identifiable, not guesses about what those keys originally meant.

## 7. Evidence ledger for future updates

### Repository evidence

- [Current training setup](../../asr_experiments/training/whisper.py): processor/model settings, preprocessing, metric callback, resolved argument construction.
- [Training/collator/masking implementation](../../asr_experiments/commands/training/train_full.py): label masking, mask schema, training-only feature masking.
- [Static/SortaGrad samplers](../../experiment_core/samplers.py): percentiles, ties, eligible pools, balanced cycling.
- [Dynamic curriculum state](../../experiment_core/dynamic_curriculum.py): preceding-epoch scores, ranking, audit normalisation.
- [Dynamic runner](../../asr_experiments/commands/training/train_s2s_curriculum.py): summed valid-token loss, duration denominator, WER generation.
- [C5 frozen config](../../experiments/rq1/c5_s2s_margin_seed42/config.json): no difficulty mixing, score and tie-break contract.
- [Acoustic metadata audit](../data/waxal_v2_acoustic_metadata.md): proxy definition, 60 dB ceiling, 2,665 capped training records, hash checks.
- [Corrected FLEURS preparation](../../asr_experiments/commands/data/prepare_fleurs_corrected_v2.py): normalisation, unique recording IDs, PCM audit.
- [Waveform implementation](../../asr_experiments/commands/data/prepare_waveform_augmentation.py) and [policy](../../configs/rq2-waveform-mild.json): exact transform order and probabilities.
- [LB protocol](../data/rq2_specaugment_lb_no_warp_protocol.md), [LD protocol](../data/rq2_specaugment_ld_no_warp_protocol.md), [FLEURS protocol](../data/rq2_specaugment_fleurs_protocol.md).
- [Medium protocol](../data/whisper_medium_rq1_rq2_protocol.md) and [queue definition](../../experiments/medium/queue.json): prepared design, not authoritative live completion status.
- [Memory reports](../../artifacts/experiment_outputs/medium/diagnostics/README.md): successful updates, initial inconclusive probe, GPU allocation/reservation.
- [Medium C0 seed42 selected state](../../artifacts/experiment_outputs/medium/rq1/c0_random_seed42_v2/trainer_state.json): step 6906 training and checkpoint-4604 selection illustrate why “final epoch” and “selected model” differ.
- [Base zero-shot summary](../../artifacts/experiment_outputs/baselines/whisper_base_zero_shot_waxal_fleurs_v1/summary.json).
- [RQ1 literature/validity review](../rq1_rq2_literature_defensibility.md) and [final RQ1 analysis](../rq1_final_analysis.md): adaptation classifications and selection-history limitations.

### Local evidence outside this repository

These are audit references, not portable paths an examiner must follow to understand the chapter:

- `/home/casper/Speech/convert_waxal_to_sde.py`: original acquisition/export implementation, including absence of a pinned labelled revision.
- `/home/casper/Speech/clean_waxal_manifest.py`: rate definitions and inequality boundaries.
- `/home/casper/Speech/data_analysis.ipynb`: executed trimming and normalisation; audit distinguished its outputs from merely present code.
- `/home/casper/Speech/data/waxal/sna_asr/{train,validation,test}/*.cleaning.json`: retained/rejected counts and threshold records.
- `/home/casper/Speech/data/waxal/sna_asr/sna_asr_split_audit.json`: original overlap.
- `/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/split_summary.json`: final counts/hours/shares.
- `/ext_data/casper/asr_data/fleurs_shona_corrected_v2/preparation_summary.json`: frozen corrected FLEURS provenance and audit counts.
- `/ext_data/casper/asr_experiment_outputs/medium/rq1/c1_sortagrad_seed43/logs/train.log`: interrupted final evaluation; cause not established.

For final submission, archive the required non-sensitive provenance beside the thesis repository or research deposit so that reproducibility does not depend solely on one researcher's home directory.

## 8. Submission checklist and unresolved matters

1. Confirm the university template's chapter hierarchy and citation style; bibliography entries here are self-contained author-year references, not a claim of template compliance.
2. Compile the LaTeX twice to resolve references and inspect tables/line breaks. No local TeX engine was available for this task; static checks are not a rendered-PDF test.
3. Resolve any A100 attribution with actual run records or remove it.
4. Attach a release-specific source if retaining exact 99.2/574.2 published hours; label local measurements separately.
5. Preserve the original labelled-acquisition revision limitation; do not substitute an unlabelled-pool revision.
6. Keep the distinction between original-split and final-split silence treatment and disclose 30-second audio-only truncation.
7. Do not restore 20% mixing, exploratory subset counts, or unsupported rounding rules to the controlled methods.
8. Update Medium completion from model/result/prediction artifacts, not merely checkpoint existence or `training_launched` metadata in a prepared queue file.
9. Report Base zero-shot retrospectively and add Medium zero-shot/FLEURS only when their actual outputs exist.
10. Keep B-series prospective until training/analysis artifacts establish completion; follow the latest amendment rather than silently combining old and new plans.
11. Distinguish supported statistical analyses from planned Holm or interaction procedures; report unadjusted historical intervals honestly.
12. Resolve the provider-specific WAXAL licensing discrepancy for redistribution and cite a Shona linguistic source if making a detailed morphological/orthographic argument beyond the illustrative metric rationale.

The most important improvements are factual separation and transparent inference, not merely changing informal vocabulary to formal synonyms. An examiner should be able to reconstruct what was compared, identify which choices were local, and see exactly where the evidence stops.