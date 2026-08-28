# Chapter 3: Methodology

> **Draft status.** This chapter distinguishes between completed experiments,
> exploratory archival experiments, and proposed experiments. Sections marked
> **planned** must be revised into past tense only after the corresponding
> experiments have been executed. The research questions below are working
> formulations and should be aligned with the final wording in Chapter 1.

## 3.1 Introduction

This chapter describes the methodology used to investigate curriculum learning
for automatic speech recognition (ASR) in Shona, a comparatively low-resource
African language. The study used pretrained multilingual speech models and
evaluated whether controlling the order and difficulty of training examples
could improve recognition performance. The methodology was organized around a
sequence of experimental conditions rather than the chronological order in
which the experiments were conducted. A conventional fine-tuning baseline was
first established, after which curriculum-learning strategies, data
augmentation, multilingual transfer, and proposed pseudo-labeling were treated
as separate experimental factors.

Two generations of experiments contributed to the study. The first was an
exploratory, notebook-based investigation containing model-scaling,
curriculum-learning, multilingual, and connectionist temporal classification
(CTC) experiments. The second used a configuration-driven training pipeline
with pinned model revisions, hashed data manifests, deterministic seeds, and
experiment tracking. Because these generations used different cleaned versions
of the WAXAL Shona manifests, they are documented as separate protocols and are
not treated as directly interchangeable experimental samples.

The remainder of this chapter presents the research design, dataset preparation,
baseline models, curriculum framework, augmentation and pseudo-labeling
protocols, experimental controls, evaluation measures, and implementation
environment.

## 3.2 Research Design

The study adopted a quantitative experimental design. ASR systems were trained
under controlled conditions and compared using corpus-level word error rate
(WER) on held-out speech. The principal independent variable was the training
schedule used to present labelled utterances. Additional independent variables
were model capacity, acoustic augmentation policy, and the inclusion of speech
from related languages. The principal dependent variable was WER; evaluation
loss was retained as a secondary optimization diagnostic.

The methodology was guided by the following provisional research questions:

- **RQ1:** Does curriculum-based ordering of training utterances improve Shona
  ASR performance relative to conventional shuffled fine-tuning?
- **RQ2:** Which investigated curriculum strategy is most effective for Shona
  ASR: static difficulty tiers, duration-based SortaGrad, sequence-loss margin,
  or WER-based margin?
- **RQ3:** Does acoustic feature augmentation improve Shona ASR performance
  when the model, data split, optimization schedule, and decoding procedure are
  held constant?
- **RQ4:** Does combining the most promising curriculum strategy with data
  augmentation provide an additional improvement over either intervention
  alone?
- **RQ5:** Can pseudo-labelled unlabelled speech improve the selected Shona ASR
  system beyond training on manually labelled speech alone?

The experimental logic was therefore:

1. establish a non-curriculum baseline;
2. compare alternative curriculum strategies;
3. evaluate augmentation independently;
4. evaluate curriculum and augmentation jointly; and
5. evaluate pseudo-labelled speech using the selected supervised system.

Only comparisons in which the model, dataset, training budget, random seed,
decoding configuration, and evaluation split were held constant were interpreted
as controlled comparisons. Earlier archival runs that changed more than one of
these factors were retained as exploratory evidence but were not used to make
causal claims about curriculum learning. Most completed experiments used one
random seed (42). Consequently, small differences were treated cautiously and
were not interpreted as conclusive evidence of superiority without replication
or confidence intervals.

## 3.3 Dataset and Data Preparation

### 3.3.1 WAXAL acquisition and NeMo manifest conversion

The primary data source was the Shona ASR subset (`sna_asr`) of the WAXAL speech
corpus. The published train, validation, and test splits were retained. Each
record included speech, a reference transcription, an utterance identifier,
speaker identifier, language, gender where available, and related metadata.

A conversion script loaded each split from `google/WaxalNLP`, skipped records
with empty transcriptions, and converted it to the JSON-lines manifest format
used by NVIDIA NeMo Speech Data Explorer (SDE). Audio bytes or source files were
materialized under a local split-specific audio directory, while source audio
encoding was preserved. Duration was measured from the audio frame count and
sample rate using SoundFile rather than accepted from unverified metadata. Each
manifest row stored the absolute audio path, measured duration, transcription,
utterance ID, and available speaker, language, and gender fields.

The resulting raw local manifests contained 14,109 training records, 1,727
validation records, and 1,749 test records. These local manifests allowed the
same audio and metadata to be inspected in SDE and processed reproducibly by
the subsequent scripts.

### 3.3.2 SDE-guided quality analysis and filtering

Speech Data Explorer was used to inspect utterance duration, transcript length,
speaking-rate outliers, unexpected characters, and suspect records. The
observations made during this inspection were translated into fixed rules in a
data-analysis notebook and a standalone cleaning script. Thus, SDE supported
exploration and review, while the actual inclusion decisions were generated by
code and recorded in auditable manifests.

For each utterance, the analysis calculated character and word counts and the
following rate measures:

$$
r_i^{\mathrm{char}}=\frac{\lvert y_i\rvert}{d_i},
\qquad
r_i^{\mathrm{word}}=\frac{\text{words}(y_i)}{d_i}.
$$

An utterance was automatically rejected when its duration was less than one
second, its character rate was below 1 or above 25 characters per second, or
its word rate exceeded 4 words per second. The same thresholds were applied to
all three published splits. Unexpected symbols and embedded newlines were
flagged for review but did not, by themselves, cause automatic rejection.

| Split | Input | Retained | Rejected |
|---|---:|---:|---:|
| Train | 14,109 | 13,799 | 310 |
| Validation | 1,727 | 1,683 | 44 |
| Test | 1,749 | 1,711 | 38 |

For each split, the process wrote five artifacts: a cleaned manifest, a rejected
manifest containing rejection reasons, an enriched manifest containing all
derived measurements, a CSV file for manual review, and a JSON policy record.
The policy record stored the fixed thresholds, input-manifest SHA-256 hash,
retained and rejected counts, and output paths. No source audio was deleted.

### 3.3.3 Silence trimming and acoustic audit

The retained training audio was examined for long trailing silence using
100-millisecond frames. A frame was considered active when its root mean square
(RMS) energy was within 40 dB of the recording's peak frame RMS. When at least
two seconds of trailing silence was detected, the recording was trimmed after
the last active region while retaining 250 milliseconds of end padding. Trimmed
audio was written as a new FLAC file and the original file was preserved. This
procedure modified 530 of the 13,799 retained training recordings; the remaining
13,269 recordings were unchanged. A CSV audit linked every original path and
duration to its output path, duration, and number of seconds removed.

An acoustic audit subsequently decoded every retained training recording and
measured duration agreement, sample rate, channel count, peak and global RMS
levels, clipping fraction, DC offset, active-speech ratio, leading and trailing
silence, and an SNR proxy. Exact decoded PCM hashes were also generated. All
training files decoded successfully and matched their manifest durations. No
clipping or residual trailing-silence failures were found under the selected
policy. Potentially difficult speech, including recordings with long leading
silence, a low active-speech ratio, or low amplitude, was flagged for review
rather than automatically removed.

### 3.3.4 Transcript normalization

One text contract was applied to train, validation, and test. Transcriptions
were normalized to Unicode NFC, typographic apostrophes were mapped to the ASCII
apostrophe, typographic dashes were standardized, and text was case-folded.
Letters, digits, and apostrophes were retained; other punctuation and symbols
were converted to spaces, after which repeated whitespace was collapsed. The
original transcription was preserved in a `text_raw` field and the normalized
form was stored in `text`. Normalization produced no empty transcriptions and no
non-Latin-script transcriptions. Twenty-seven training records containing
digits were flagged for Shona-specific verbalization review rather than being
expanded using English assumptions.

### 3.3.5 Experimental data protocols

Three related data protocols appeared across the experiments:

| Protocol | Purpose | Train | Validation | Test |
|---|---|---:|---:|---:|
| Archival cleaned protocol | Earlier model and dynamic-curriculum experiments | 13,343 | 1,620 | 1,635 |
| SDE-normalized full protocol | Reproducible baselines and augmentation | 13,799 | 1,683 | 1,711 |
| SDE curriculum-ready protocol | Static cumulative curriculum | 13,206 | 1,683 | 1,711 |

The archival protocol read separate `*.cleaned.json` manifests and then retained
audio between 0.1 and 30 seconds. It predates the SDE-normalized protocol and is
not directly interchangeable with it. The SDE-normalized full protocol contained
approximately 79.28 hours of training speech, 9.70 hours of validation speech,
and 9.93 hours of test speech. The curriculum-ready training subset excluded
593 quality-review records from the full normalized training set, but retained
them separately for audit rather than deleting them. No additional record met
the hard rejection rules after silence trimming and normalization.

Before model training, the newer pipeline verified that every referenced audio
file existed and every transcription was non-empty. SHA-256 hashes of all three
manifests were stored in each run manifest.

### 3.3.6 Model input preprocessing

All audio was decoded as mono speech and resampled to 16 kHz. The Whisper
feature extractor transformed each waveform into an 80-channel log-Mel
spectrogram. Inputs were padded or truncated to 3,000 frames, corresponding to
Whisper's 30-second input window. An attention mask was retained to distinguish
valid audio frames from padding. Target transcriptions were tokenized with the
Whisper multilingual tokenizer configured for Shona transcription. Target
sequences were padded within each batch, and padding tokens were replaced with
`-100` so that they did not contribute to the sequence-to-sequence loss.

### 3.3.7 Duplicate and split-leakage audit

Decoded PCM hashes were compared within and across all splits. No exact audio
duplicates were found within any split or across train, validation, and test.
Two repeated-transcription groups, containing four training records, were
retained because their audio differed. One normalized transcription was shared
between train and validation, but its audio was not duplicated.

The original WAXAL split did not provide a strong speaker-independent test. Of
97 validation speakers, 94 appeared in training; of 110 test speakers, 101
appeared in training. Furthermore, 1,700 of the 1,711 test utterances were
spoken by speakers represented during training. The reported experiments
therefore measure in-domain recognition more strongly than generalization to
unseen speakers. A speaker-disjoint split is required before making claims about
speaker-independent generalization.

### 3.3.8 Related-language speech

Exploratory multilingual experiments supplemented Shona with Tshivenda and
isiZulu speech from the African Next Voices dataset. Auxiliary-language records
were retained when their duration was between 1 and 30 seconds, their character
rate was between 1 and 25 characters per second, their word rate did not exceed
four words per second, and their transcription was non-empty. The two-language
training set contained 36,286 examples. The three-language set contained 53,186
examples and approximately 75.60 hours of Shona, 100.00 hours of Tshivenda, and
89.19 hours of recoverable isiZulu speech. These multilingual experiments were
exploratory and were not curriculum-learning controls.

## 3.4 Baseline ASR Model

The principal ASR architecture was Whisper, a pretrained encoder-decoder
Transformer. Whisper converts a log-Mel spectrogram into hidden acoustic
representations with an encoder and autoregressively generates text tokens with
a decoder. The multilingual checkpoints were appropriate for transfer to Shona
because their pretraining included multilingual speech recognition and
translation objectives.

Whisper Tiny, Base, Small, Medium, Large, and Large-v3 checkpoints were examined
in the exploratory archive. The reproducible experiments concentrated on Base,
Medium, and Large. For these runs, generation was explicitly configured with
the language set to Shona and the task set to transcription. Forced decoder IDs
and the default suppressed-token list were cleared consistently across the
controlled runs. Predictions were generated greedily with a maximum sequence
length of 225 tokens.

The normalized baseline protocol fine-tuned each model for three epochs using a
learning rate of $1 \times 10^{-5}$, 500 warm-up steps, FP16 precision, gradient
checkpointing, and an effective batch size of six. Base used a physical batch
size of six. Medium used a physical batch size of two with three gradient
accumulation steps. Large used a physical batch size of one with six accumulation
steps and an 8-bit AdamW optimizer to fit GPU memory. Validation WER selected the
best checkpoint.

The model-scaling experiments established the appropriate capacity for later
experiments; differences between model sizes were not attributed to curriculum
or augmentation. A `facebook/wav2vec2-base` CTC system was also trained in the
legacy archive as a secondary architectural benchmark. Its text normalization
and CTC objective differed from the Whisper protocol, so its WER was used only
as contextual evidence rather than a direct controlled comparison.

## 3.5 Curriculum Learning Framework

Curriculum learning changes the order or sampling distribution of training
examples so that optimization begins with examples considered easier and
progressively incorporates more difficult material. Let the labelled training
set be

$$
\mathcal{D} = \{(x_i, y_i, d_i)\}_{i=1}^{N},
$$

where $x_i$ is an utterance, $y_i$ is its reference transcription, and $d_i$ is
its duration. A curriculum assigns each example a difficulty score $s_i$ and
constructs an ordering or stage schedule from those scores.

### 3.5.1 Static cumulative curriculum

The static curriculum was constructed from the SDE-normalized training data.
The 593 review-flagged records were kept outside this experiment, leaving 13,206
quality-ready utterances. Four percentile-ranked signals were combined: duration
$q_i^{d}$, absolute deviation from the median word rate $q_i^{r}$, inverse SNR
rank $q_i^{n}$, and inverse active-speech-ratio rank $q_i^{s}$. The static
difficulty score was

$$
s_i^{\mathrm{static}} = 0.40q_i^{d} + 0.25q_i^{r}
+ 0.20q_i^{n} + 0.15q_i^{s}.
$$

The ranked data were divided into equal easy, medium, and hard thirds. Training
began with 4,402 easy examples (20.90 hours), expanded to 8,804 easy and medium
examples (45.55 hours), and finally used all 13,206 examples (75.68 hours).
Within each tier, records were ordered deterministically by difficulty score,
duration, and utterance ID. One implementation traversed these three cumulative
stages once, while a second repeated the full-data stage three times. Because
the variants had different total exposure counts, they were treated as separate
experiments rather than replications.

### 3.5.2 Duration-based SortaGrad

SortaGrad used utterance duration as a model-independent proxy for difficulty.
During the first epoch, examples were stably ordered from shortest to longest.
Subsequent epochs used seeded random permutations. No examples were removed or
oversampled. This strategy tested whether presenting shorter utterances first
could stabilize early optimization without repeatedly imposing a curriculum.

### 3.5.3 Sequence-to-sequence margin curriculum

The sequence-to-sequence margin (S2S-M) strategy used the model's own training
loss to update the ordering. The first epoch was ordered by duration. During
that epoch, the summed token negative log-likelihood was recorded for each
utterance and normalized by its audio duration:

$$
s_i^{\mathrm{S2S}} =
\frac{-\sum_{t=1}^{T_i}\log p_\theta(y_{i,t}\mid y_{i,<t},x_i)}{d_i}.
$$

At the next epoch boundary, examples were ranked from lower to higher score,
with duration and original index used as deterministic tie-breakers. Scores
were recomputed as the model changed, making this a dynamic, model-dependent
curriculum.

### 3.5.4 WER-margin curriculum

The WER-margin (WER-M) strategy estimated example difficulty from recognition
errors. Before the training forward pass, each batch was decoded greedily and a
per-example WER was calculated:

$$
s_i^{\mathrm{WER}} =
\frac{S_i+D_i+I_i}{N_i},
$$

where $S_i$, $D_i$, and $I_i$ are the substitution, deletion, and insertion
counts for utterance $i$, and $N_i$ is the number of words in its reference.
Scores were min-max normalized within the completed scoring period and used to
rank the next epoch. This method more directly represented recognition
difficulty than duration, but it added generation overhead to training.

### 3.5.5 Uniform difficulty mixing

The dynamic margin curricula divided the ranked examples into easy, medium, and
hard thirds. To prevent the beginning of an epoch from containing only easy
speech, 20% of the easy third was replaced by examples sampled from the other
tiers. Of the replacement examples, 40% came from the medium tier and 60% from
the hard tier. With 13,343 examples, this injected approximately 356 medium and
533 hard examples into the early portion of each completed epoch. Sampling used
a deterministic seed.

The archival curriculum experiments are valuable for comparing candidate
mechanisms, but they do not form a complete causal ablation because training
steps, total sample exposures, evaluation frequency, and in some cases data
splits differed. A final confirmatory comparison should rerun the clean control,
SortaGrad, S2S-M, and WER-M with the same normalized manifests, model revision,
three-epoch budget, optimizer, batch size, and seeds.

## 3.6 Data Augmentation

Data augmentation was investigated using SpecAugment applied to Whisper's
log-Mel input features. Time warping was omitted; only time masking and frequency
masking were used. Augmentation was performed online and only while the model was
in training mode. Validation, test, and inference features remained unchanged.

The mild policy used a time-mask probability of 0.05, a time-mask length of
eight frames, and at least one time mask. It also used a frequency-mask
probability of 0.05, a frequency-mask length of eight Mel channels, and at least
one frequency mask. Masked feature values were set to zero.

Three principal conditions were defined:

| Condition | Clean presentation probability | Mild SpecAugment probability |
|---|---:|---:|
| Clean control | 1.0 | 0.0 |
| Always-augmented | 0.0 | 1.0 |
| Mixed augmentation | 0.5 | 0.5 |

For the mixed condition, a Bernoulli draw was made independently for each
example at each presentation. Thus, the physical dataset was not doubled and no
augmented features were stored. Across three epochs, each utterance was expected
to be presented approximately 1.5 times clean and 1.5 times with newly sampled
masks. PyTorch's seeded random-number generator and deterministic training mode
controlled both the mask generation and clean-versus-augmented selection.

The controlled augmentation experiments kept model revision, manifests, seed,
training schedule, effective batch size, decoding, and evaluation fixed. A mild
always-on policy was completed for Base and Medium, and the Base mixed policy
was also completed. A stronger literature-derived policy was configured but
should not be described as executed unless run artifacts are confirmed.

An exploratory continuation study was also performed after the three-epoch
Medium mild-SpecAugment run. The selected model was first trained for 2,000
additional optimizer steps with a learning rate of $2 \times 10^{-6}$ and 100
warm-up steps. It was then trained for a further 6,000 steps with a learning
rate of $5 \times 10^{-7}$ and 100 warm-up steps. Both stages retained the same
effective batch size of six and the same always-on mild augmentation policy.
Because the parent checkpoints contained model weights but not optimizer and
scheduler states, each continuation used a fresh AdamW optimizer and linear
schedule. These stages were therefore treated as new optimization phases, not
exact resumptions of the original trajectory.

**Planned combined condition for RQ4.** The selected curriculum should be
crossed with augmentation in a factorial comparison containing: conventional
clean training, curriculum only, augmentation only, and curriculum plus
augmentation. All four cells must use identical data, model, training budget,
seed set, and evaluation. This controlled combined experiment has not yet been
completed and therefore cannot currently support an RQ4 claim.

## 3.7 Pseudo-Labeling

> **Planned protocol; not yet executed.** No completed pseudo-labeling run was
> found in either experiment repository. This section records the intended
> protocol and must be updated with the actual corpus, thresholds, and sample
> counts if pseudo-labeling is included in the final thesis.

Pseudo-labeling is intended to exploit speech for which manually verified Shona
transcriptions are unavailable. The best supervised model selected using the
validation split would act as a teacher and generate candidate transcriptions
for an unlabelled speech pool. The unlabelled pool must first be checked for
audio duplication and speaker overlap with the validation and test sets. Test
audio must never enter teacher selection, confidence-threshold selection, or
student training.

For each unlabelled utterance $x_j$, the teacher would generate a transcription
$\hat{y}_j$ and a length-normalized sequence confidence score. Candidate pairs
would be filtered using a threshold selected on development data and basic
quality constraints such as non-empty transcription, valid duration, and
plausible speaking rate. The retained pseudo-labelled set would be

$$
\widehat{\mathcal{D}}_u =
\{(x_j,\hat{y}_j):c_j \geq \tau\},
$$

where $c_j$ is teacher confidence and $\tau$ is the fixed acceptance threshold.
The student model would then be trained on the union of manually labelled and
pseudo-labelled examples. To limit error reinforcement, manually labelled data
should receive at least equal sampling weight, and pseudo-labelled examples may
be down-weighted in the loss or introduced progressively by confidence.

The controlled comparison for RQ5 should contain a supervised-only condition
and an otherwise identical supervised-plus-pseudo-labelled condition. The
teacher checkpoint, unlabelled corpus version, filtering thresholds, accepted
and rejected counts, pseudo-label manifest hash, and student initialization must
all be recorded. Final evaluation must use the same untouched test set as the
supervised control.

## 3.8 Experimental Design

The experiments were organized into the following matrix:

| Family | Conditions | Protocol | Evidential role |
|---|---|---|---|
| Model baseline | Whisper Tiny, Base, Small, Medium, Large, Large-v3 | Legacy | Exploratory capacity study |
| Reproducible baseline | Whisper Base, Medium, Large; three epochs | Normalized | Non-curriculum reference |
| Static curriculum | Cumulative tiers v1 and v2 | Legacy | Exploratory curriculum |
| Ordering curriculum | SortaGrad | Legacy | Exploratory curriculum |
| Dynamic curriculum | S2S-M and WER-M | Legacy | Exploratory strategy comparison |
| CTC baseline | Wav2Vec2 Base; partial XLS-R | Legacy | Secondary architecture benchmark |
| Multilingual transfer | Shona+Tshivenda; Shona+Tshivenda+isiZulu | Legacy | Exploratory transfer study |
| Augmentation | Clean, always-on mild, 50/50 mixed | Normalized | Controlled augmentation study |
| Curriculum + augmentation | Four-cell factorial comparison | Normalized | Planned confirmatory study |
| Pseudo-labeling | Labelled only vs labelled+pseudo-labelled | To be fixed | Planned study |

For controlled experiments, the validation set was evaluated periodically and
the checkpoint with the lowest validation WER was selected. The test set was
reserved for final evaluation of the selected checkpoint. Test WER was not used
to choose a checkpoint or tune hyperparameters. Where earlier notebook
experiments repeatedly evaluated the test set, this was documented as a risk of
implicit test-set adaptation.

The normalized Base, Medium, and Large experiments used seed 42 and three
epochs. Medium and Large were evaluated, logged, and checkpointed every 1,000
optimizer steps; Base used epoch-based evaluation in the original baseline and
1,000-step evaluation in the deterministic augmentation controls. Augmentation
effects were interpreted only against the same model size and data protocol.
Changes smaller than 0.5 absolute WER points were predeclared as requiring
additional evidence rather than being treated as practically meaningful.

For final confirmatory experiments, at least three seeds should be used. The
same seeds must be shared across conditions to reduce comparison variance. In
addition to point estimates, paired bootstrap resampling over test utterances
should be used to estimate a 95% confidence interval for the WER difference
between each intervention and its matched control.

## 3.9 Evaluation Metrics

The primary evaluation metric was corpus-level word error rate:

$$
\mathrm{WER}=\frac{S+D+I}{N}\times 100\%,
$$

where $S$, $D$, and $I$ are the total word substitutions, deletions, and
insertions, and $N$ is the total number of reference words. Lower WER indicates
better recognition performance. Corpus WER was calculated with the `evaluate`
and `jiwer` packages after removing special tokens from model predictions.

The normalized Whisper experiments used the normalized manifest text as the
reference and did not apply an additional post-decoding normalizer. The legacy
Whisper experiments generally preserved their reference casing and punctuation,
whereas the CTC experiments lowercased text, removed combining marks, and
restricted the character inventory. WER values from these different text
protocols were therefore not treated as directly comparable.

Relative WER reduction may be reported as

$$
\mathrm{WERR}=\frac{\mathrm{WER}_{\mathrm{control}}-
\mathrm{WER}_{\mathrm{system}}}{\mathrm{WER}_{\mathrm{control}}}\times100\%.
$$

Evaluation loss was recorded to monitor optimization but was not used as the
primary measure of transcription quality. Training loss, runtime, GPU memory
requirements, and checkpoint size were retained as implementation diagnostics.
Given the speaker overlap in WAXAL v1, full-split WER should be supplemented by
speaker-disjoint evaluation before making generalization claims.

## 3.10 Implementation Details

The reproducible pipeline was implemented in Python 3.10.12 using PyTorch
2.5.1, Transformers 4.46.3, Datasets 2.20.0, Accelerate 1.1.1, Evaluate 0.4.2,
JiWER 3.0.4, NumPy 1.26.4, bitsandbytes 0.45.5, and Weights & Biases 0.28.2.
Training was performed on an NVIDIA GeForce RTX 4090 with 24 GB of GPU memory
and CUDA 12.4.

The newer training system used versioned JSON experiment configurations and a
shared `Seq2SeqTrainer` implementation. Each configuration specified the model
identifier and immutable revision, manifest paths, language and task, random
seed, batch and accumulation settings, optimization schedule, augmentation
policy, checkpoint behavior, output location, and experiment-tracking metadata.
Long-running jobs were launched inside detached `tmux` sessions so that training
continued after SSH disconnection. Resource guards checked available GPU memory
and disk capacity before launch.

Before training, the pipeline wrote the resolved experiment configuration and a
run manifest containing SHA-256 hashes of the trainer, configuration,
dependency file, data manifests, and any local parent model. The manifest also
recorded package versions, Python and PyTorch versions, CUDA availability, GPU
model, effective seed, training duration, and output directory. W&B logging was
restricted to an approved existing project.

The archival notebook experiments predated this infrastructure. Their methods
were reconstructed from notebook source cells, saved `trainer_state.json`
files, result JSON files, checkpoints, and local W&B metadata. Saved metric and
trainer-state files were treated as stronger evidence than displayed notebook
outputs because some notebooks contained stale or copied output cells. Runs
interrupted by out-of-memory errors or manual termination were identified as
partial rather than completed experiments.

## 3.11 Chapter Summary

This chapter defined a research-question-driven protocol for evaluating
curriculum learning in low-resource Shona ASR. It separated exploratory legacy
experiments from the newer reproducible baseline and augmentation experiments,
described the two WAXAL data protocols, and documented the Whisper baseline,
four curriculum strategies, SpecAugment policies, multilingual transfer study,
and proposed pseudo-labeling procedure. WER was selected as the principal
metric, with validation-based checkpoint selection and held-out test evaluation.
The chapter also identified important methodological constraints: severe
speaker overlap in the original WAXAL split, mostly single-seed experiments,
inconsistent budgets in the archival curriculum study, and the absence of
completed curriculum-plus-augmentation and pseudo-labeling experiments. These
constraints define the additional controlled experiments required before the
final thesis claims are made.

## Author Notes Before Final Submission

- Align the exact wording and numbering of RQ1-RQ5 with Chapter 1.
- Add formal citations for WAXAL, NVIDIA NeMo Speech Data Explorer, Whisper,
  SpecAugment, curriculum learning, SortaGrad, Wav2Vec2, XLS-R, WER, and
  pseudo-labeling.
- Decide whether exploratory multilingual and CTC studies belong in Chapter 3
  or an appendix.
- Transfer the verified Base 50/50 and Medium Stage-3 outcomes to Chapter 4.
- Run a matched clean-versus-curriculum comparison on one fixed data protocol.
- Do not retain Section 3.7 as a completed method unless pseudo-labeling is run.
- Add a speaker-disjoint evaluation or narrow the thesis claims to in-domain
  recognition on the WAXAL v1 split.