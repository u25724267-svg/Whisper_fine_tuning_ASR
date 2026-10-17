# Specific Inline Responses & Text Additions for the Methodology Draft

This document provides exact, copy-pasteable text additions and responses for every `\todo` comment in your original LaTeX draft (`chapter3_methodology_draft.tex`). You can copy these directly into your current text without having to rephrase the entire chapter.

---

## 1. Research Design Comments

### `\todo{why}` (Line: "evaluation was done using the Word Error Rate (WER) on examples not seen during training\todo{why}")
**Replace with / Add:**
> "...evaluation was conducted using the Word Error Rate (WER) on examples not seen during training because the objective of an ASR system is to generalize to novel acoustic utterances in real-world deployment; evaluating on training speech would merely measure the model's capacity to memorize acoustic-text pairs rather than its ability to recognize unseen speech."

---

### `\todo{This is incomplete}` (Line: "We included data augmentation\todo{This is incomplete}")
**Replace with / Add:**
> "We included data augmentation as a secondary independent factor, specifically evaluating (i) feature-level masking (SpecAugment) applied to log-Mel spectrograms, (ii) time-domain waveform corruptions including additive non-speech noise, simulated room impulse response (RIR) reverberation, and tempo stretching, and (iii) the interaction between curriculum order and data augmentation within balanced factorial designs."

---

### `\todo{Why}` (Line: "evaluation loss and character error rate (CER) as additional dependent variables \todo{Why}")
**Replace with / Add:**
> "...we also monitored evaluation loss and character error rate (CER) as secondary dependent variables. CER was included because Shona is an agglutinative, morphologically rich language where a single incorrect affix causes an entire word to be penalized as an error under WER; CER exposes whether the model is learning phonetically and orthographically close representations. Evaluation loss (sequence cross-entropy) was recorded strictly as an optimization diagnostic to track convergence behavior, though not as a substitute for recognition accuracy since cross-entropy on teacher-forced references does not always correlate monotonically with greedy decoding WER."

---

## 2. Implementation Details Comments

### `\todo{elaborate why 2 different gpus}` (Line: "Nvidia Geforce RTX 4090 with 24GB VRAM of memory and CUDA 12.4 driver and Nvidia A100 with 40GB VRAM\todo{elaborate why 2 different gpus}")
**Replace with / Add:**
> "All controlled confirmatory experiments (Whisper Base and Medium across seeds 42, 43, and 44) were trained on a single workstation equipped with an NVIDIA GeForce RTX 4090 (24~GB VRAM, CUDA 12.4). An NVIDIA A100 (40~GB VRAM) was referenced in legacy SLURM cluster scheduling scripts for exploratory large-model pilots that exceeded 24~GB under unoptimized 32-bit precision, but was not used for any controlled comparison in order to eliminate hardware-induced numerical divergence."
*(Note: If no A100 was actually used in the final runs, simply remove the mention of the A100 and state that all experiments were trained on the RTX 4090 to preserve floating-point consistency across conditions.)*

---

## 3. Dataset Acquisition, SDE, and Preprocessing Comments

### `\todo{why sde}` (Line: "Nvidia NEMO Speech Data Explorer (SDE) \cite{Bakhturina2021} \todo{why sde}")
**Replace with / Add:**
> "...using the NVIDIA NeMo Speech Data Explorer (SDE) \citep{Bakhturina2021}. SDE was selected because it provides an interactive, browser-based interface designed specifically for ASR quality assurance, enabling synchronized inspection of audio waveforms, spectrograms, text transcripts, and automated outlier distributions (such as duration and speaking-rate extremes) before encoding cleaning rules into code."

---

### `\todo{use clearer language than materialize}` (Line: "\hl{We materialized the audio source files and preserved the source audio encoding}\todo{use clearer language than materialize}")
**Replace with / Add:**
> "We extracted the raw audio byte streams from the remote repository and saved them as standalone local audio files organized by split, preserving the source audio sample rate and encoding format without lossy transcode."

---

### `\todo{what is soundfile : is it a package/ or actual audio file etc}` (Line: "using the SoundFile\todo{what is soundfile : is it a package/ or actual audio file etc}")
**Replace with / Add:**
> "...using the Python \texttt{soundfile} library (which interfaces with the C-based \texttt{libsndfile} library to read exact header frame counts and audio sample rates) rather than accepting unverified duration values from dataset metadata."

---

### `\todo{what is character & word rate}` (Line: "For each utterance we calculate the character rate and wordrate using the following formulars\todo{what is character & word rate}")
**Replace with / Add:**
> "The character rate ($r_i^{\text{char}}$) and word rate ($r_i^{\text{word}}$) measure the density of text produced per second of audio. In genuine conversational speech, human speech production operates within well-established physiological limits (typically 10--15 characters per second and 2--3 words per second). Extreme values indicate transcription-audio misalignment, such as an incomplete transcription for a long recording (extremely low rate) or text describing unrecorded content (implausibly high rate)."

---

### `\todo{why}` (Line: "Utterances with a duration below one second were automatically rejected\todo{why}")
**Replace with / Add:**
> "Utterances shorter than one second were rejected because the WAXAL elicitation task required speakers to describe complex images; recordings under one second typically represent accidental microphone clicks, false starts, or truncated cut-offs that lack intelligible linguistic content."

---

### `\todo{why}` (Line: "character rate below one and above 25 or the word rate exceeded 4 words per second were rejected\todo{why}")
**Replace with / Add:**
> "...were rejected because these thresholds demarcate clear physical impossibilities in natural Shona speech. A character rate below 1 char/sec indicates that the audio contains substantial untranscribed silence, while a rate above 25 char/sec or a word rate exceeding 4 words/sec indicates that the text transcription contains substantially more content than could phonetically be articulated within the clip's duration."

---

### `\todo{why is it necessary to apply to all data splits not just training or something else}` (Line: "These thresholds were applied to all data splits\todo{why is it necessary to apply to all data splits not just training or something else}")
**Replace with / Add:**
> "Applying identical filtering rules to all three splits (training, validation, and test) is necessary to ensure consistent data distributions. If evaluation splits retained severely misaligned audio-text pairs, the model would be penalized with spurious word errors for failing to predict words that were never spoken in the audio, degrading evaluation validity."

---

## 4. Silence Trimming, Normalization, & Preprocessing

### `\todo{why do we choose 40dbs}` (Line: "RMS energy is within 40 decibels (db) of the recording’s peak frame RMS\todo{why do we choose 40dbs}")
**Replace with / Add:**
> "A relative energy threshold of $-40$~dB (corresponding to an RMS amplitude of 1\% of the recording's peak frame) was selected because WAXAL recordings were captured across diverse mobile devices and acoustic environments with variable overall gain. A relative threshold adapts automatically to each device's dynamic range, distinguishing quiet speech sounds (such as fricatives) from background static hiss."

---

### `\todo{justify parameters}` (Line: "trimmed after the last region detected as active with a padding of 250 millisecond\todo{justify parameters}")
**Replace with / Add:**
> "Recordings with at least 2.0 seconds of trailing silence were trimmed, retaining 250 milliseconds of end padding. The 2-second threshold ensures that natural, meaningful conversational pauses are not aggressively cut, while the 250~ms margin protects the trailing energy decay of final consonants from being clipped."

---

### `\todo{why flac}` (Line: "saved as a new flac file\todo{why flac}")
**Replace with / Add:**
> "Trimmed recordings were written in the Free Lossless Audio Codec (FLAC) format because FLAC provides lossless compression—preserving exact pulse-code modulation (PCM) numerical fidelity without the compression artifacts of lossy codecs (such as MP3)—while reducing disk footprint."

---

### `\todo[color=green]{WHY unicode nfc}` (Line: "transcriptions where therefore normalized to Unicode NFC\todo[color=green]{WHY unicode nfc}")
**Replace with / Add:**
> "Transcriptions were normalized to Unicode Normalization Form Canonical Composition (NFC) because Unicode allows accented and composite characters to be encoded either as a single precomposed character or as a base character followed by combining diacritics. NFC composes these into canonical single codepoints, preventing visually identical characters from being treated as different tokens by the tokenizer."

---

### `\todo[color=green]{Why}` (Line: "the text was case corrected\todo[color=green]{Why}")
**Replace with / Add:**
> "...text was case-folded to lowercase because capital letters carry no acoustic distinction in speech, and capitalization in the WAXAL annotations was applied inconsistently; evaluating case-sensitive text would penalize orthographic casing decisions rather than speech recognition capability."

---

### `\todo[color=green]{Why}` (Line: "other punctuation symbols were converted to spaces \todo[color=green]{Why} and repeated spaces were collapsed")
**Replace with / Add:**
> "...other punctuation marks were replaced with spaces rather than deleted so that words separated solely by punctuation marks (e.g., \texttt{shoko,shoko}) would not be erroneously merged into a single non-existent word token, after which repeated spaces were collapsed."

---

### `\todo{why was it important to retain the original text}` (Line: "retained the original text transcription in a "textraw" field\todo{why was it important to retain the original text}")
**Replace with / Add:**
> "The original reference text was preserved in the \texttt{text\_raw} manifest field to maintain an auditable provenance trail, enabling manual inspection of the normalization transformations and allowing future researchers to apply alternate orthographic contracts without re-acquiring the raw corpus."

---

### `\todo[color=green]{Why!}` (Line: "27 training records containing digits for human review \todo[color=green]{Why!}")
**Replace with / Add:**
> "The 27 training records containing numerical digits were flagged for manual review rather than automatically expanded because verbalizing numerals into Shona requires language-specific grammatical noun-class concordial agreements (e.g., the number 2 varies as \emph{vaviri}, \emph{zviviri}, \emph{mavirikiti} depending on noun class); automatic English-rule conversion would introduce erroneous verbalizations."

---

### `\todo[color=green]{Why!}` (Line: "resampled to 16khz \todo[color=green]{Why!}")
**Replace with / Add:**
> "Audio was resampled from its native 48~kHz rate to 16~kHz because Whisper's log-Mel convolutional front-end was pretrained exclusively on 16~kHz audio; supplying audio at any other sampling rate would alter the frequency bin mapping and corrupt the spectral representations."

---

### `\todo[color=green]{Elaborate more: What is the spectrogram?}` (Line: "log-Mel spectrogram with 80 channels \todo[color=green]{Elaborate more: What is the spectrogram?}")
**Replace with / Add:**
> "A log-Mel spectrogram is a 2D time-frequency visual representation of sound energy. It is generated by computing the Short-Time Fourier Transform (STFT) over 25~ms windows with 10~ms stride, passing the Fourier magnitude bins through an 80-channel filterbank spaced according to the non-linear Mel perceptual pitch scale (which mirrors the human ear's higher sensitivity to low frequencies), and applying a logarithmic amplitude compression."

---

### `\todo{Why}` (Line: "attention mask to distinguish between audio frames and padding \todo{Why}")
**Replace with / Add:**
> "An acoustic attention mask was retained to delineate between valid audio frames and padded silence frames, ensuring that SpecAugment time masks are placed strictly across real speech and not wasted on padded regions."

---

### `\todo{Need Clarification}` (Line: "tokenizer with its configuration set to Shona\todo{Need Clarification}")
**Replace with / Add:**
> "The multilingual byte-pair encoding (BPE) tokenizer was configured with its prefix prompt forced to \texttt{<|startoftranscript|><|sn|><|transcribe|><|notimestamps|>}, explicitly instructing the autoregressive decoder to generate Shona text without timestamps rather than translating into English."

---

### `\todo{why do we need to pad}` (Line: "padded the target sequences\todo{why do we need to pad}")
**Replace with / Add:**
> "Target token sequences within each batch were padded to the length of the longest transcript in that batch to assemble rectangular tensors required for parallel GPU matrix multiplication."

---

### `\todo{elaborate how replacing with -100 does not contribute to loss}` (Line: "replaced with -100 which prevents the padded frames from contributing to the sequence to sequence loss \todo{elaborate how replacing with -100 does not contribute to loss}")
**Replace with / Add:**
> "Padding token positions were assigned a label value of $-100$ because PyTorch's cross-entropy loss function sets its default \texttt{ignore\_index} to $-100$, which strictly masks those positions out of both the numerator sum and denominator token count during loss and gradient computation."

---

## 5. Duplicate Audit & Speaker-Disjoint Split Comments

### `\todo{Elaborate/ what are pcm hashes}` (Line: "Pulse-Code Modulation (PCM) hashes and compared them across the different data \todo{Elaborate/ what are pcm hashes}")
**Replace with / Add:**
> "Canonical Pulse-Code Modulation (PCM) hashes were generated by decoding audio into raw 16-bit uncompressed integer samples and computing a SHA-256 cryptographic digest over the sample byte stream. Unlike file hashes (which change if file metadata, container tags, or headers differ), PCM hashing detects whether the underlying digital audio waveforms are sample-for-sample identical."

---

### `\todo{elaborate why the 20\% cap}` (Line: "limits held out speakers to 20\% of respective split rows and hours.\todo{elaborate why the 20\% cap}")
**Replace with / Add:**
> "A 20\% ceiling was enforced so that no single speaker contributes more than 20\% of utterances or audio duration in validation or test. In initial unconstrained assignments (v1), one prolific speaker accounted for 51.1\% of validation speech and 35.1\% of test speech, which meant evaluation results primarily reflected recognition of a single individual rather than broad speaker generalization."

---

### `\todo{how}` (Line: "pooled the original normalized WAXAL manifests and grouped the 17193 utterances by speaker\todo{how}")
**Replace with / Add:**
> "...by constructing a dictionary mapping each of the 161 unique speaker identifiers to the complete list of utterances recorded by that speaker, ensuring that each speaker's entire collection was assigned as an indivisible unit to exactly one split."

---

### `\todo{explain the process}` (Line: "selected assignment having 24 speakers allocated to each held out set\todo{explain the process}")
**Replace with / Add:**
> "A seeded randomized search evaluated 50,000 candidate speaker assignments using seed 42. In each trial, 24 speakers were assigned to validation, 24 to test, and the remaining 113 to training. Candidates violating the 20\% concentration cap were discarded, and the feasible assignment minimizing the squared relative deviation objective $J(a)$ (achieving $J(a) = 0.0029$) was selected."

---

### `\todo{Why fleurs}` (Line: "Shona component of the Google Fleurs \cite{Conneau2022} dataset which contains 2463 training samples, 393 validation, and 925 test examples\todo{Why fleurs}")
**Replace with / Add:**
> "Google FLEURS was selected as the out-of-domain evaluation corpus because it represents read speech recorded from Wikipedia sentences in different acoustic settings by different speakers, providing a natural out-of-domain contrast to WAXAL's image-prompted conversational speech."

---

## 6. Baseline Whisper Configuration Comments

### `\todo{why were the larger models dropped}` (Line: "focused on the base, and medium\todo{why were the larger models dropped}")
**Replace with / Add:**
> "Whisper Large was omitted from confirmatory multi-seed matrices because exploratory runs showed diminishing returns (a modest 1.02\% WER reduction over Medium at the cost of a 6-fold increase in runtime from 2.56h to 15.35h per run), and because Large required 8-bit quantization and micro-batching to fit within 24~GB VRAM, which would confound model size with quantization and batching changes."

---

### `\todo{Why}` (Line: "finetuned each model with three epochs\todo{Why}")
**Replace with / Add:**
> "...fine-tuned for exactly three epochs (6,906 optimizer steps) because three epochs provided sufficient optimization convergence on 79.8 hours of training speech while keeping the multi-condition, multi-seed matrix computationally feasible within our GPU allocation."

---

### `\todo{Why}` (Line: "500 warm-up steps\todo{Why}")
**Replace with / Add:**
> "...using 500 warm-up steps ($\approx 7.2\%$ of training updates) to prevent large early gradient updates from destabilizing pretrained Transformer attention weights before the AdamW optimizer's running moment estimates stabilized."

---

### `\todo{Why}` (Line: "FP16 precision\todo{Why}")
**Replace with / Add:**
> "...using 16-bit automatic mixed precision (FP16) to approximately halve activation memory and leverage NVIDIA Tensor Cores for faster computation while retaining 32-bit master weights for numerical stability."

---

### `\todo{Why}` (Line: "gradient checkpointing\todo{Why}")
**Replace with / Add:**
> "...with non-reentrant gradient checkpointing enabled, which trades compute for memory by discarding intermediate activations during the forward pass and recomputing them during backpropagation, enabling Whisper Medium to train with physical batch size 6 without out-of-memory errors."

---

### `\todo{Justify Parameters}` (Line: "effective batch size of six on the base model... \todo{Justify Parameters}")
**Replace with / Add:**
> "A physical batch size of 6 with gradient accumulation 1 was selected because it saturated GPU compute without exceeding VRAM, and was held strictly identical between Base and Medium to eliminate optimization batching artifacts."

---

## 7. Curriculum Learning Framework Comments

### `\todo{What informed the weights}` (Line: "$s_i^{\text{static}} = 0.40 q_i^d + 0.25 q_i^r + 0.20 q_i^n + 0.15 q_i^s$\todo{What informed the weights}")
**Replace with / Add:**
> "These heuristic weights were adopted during exploratory notebooks to prioritize duration (0.40) and speaking rate regularity (0.25) over noise (0.20) and activity ratio (0.15). In the confirmatory protocol, this formulation was superseded by the two-signal joint score $s_i^{\text{C3}} = 0.5 P(d_i) + 0.5(1 - P(\widehat{\text{SNR}}_i))$ to isolate acoustic factors without researcher degrees of freedom."

---

### `\todo{elaborate more on how this was done...}` (Line: "Training started on the 4402 easy examples... \todo{elaborate more on how this was done. was the data increased at every epoch. Did the first epoch see less data than the other epochs?}")
**Replace with / Add:**
> "In the confirmatory cumulative condition (C4), the eligible pool expanded from 4,603 easy utterances in epoch 0, to 9,205 in epoch 1, to all 13,807 in epoch 2. Crucially, to prevent early epochs from seeing fewer gradient updates than the baseline, the eligible pool was cycled until exactly 13,807 presentations were delivered in every epoch, ensuring exact update parity with C0."

---

### `\todo{why}` (Line: "negative log-likelihood of each utterance and normalized it by its audio duration\todo{why}")
**Replace with / Add:**
> "...normalized by audio duration $d_i$ because unnormalized cross-entropy accumulates across tokens, causing longer utterances to naturally yield larger losses regardless of acoustic or linguistic difficulty; dividing by duration measures token surprise per second of speech."

---

### `\todo{go further than defining variables, tell us how the function works}` (Line: S2S formula definition)
**Replace with / Add:**
> "During the forward pass of each batch, the model computes the negative logarithm of the softmax probability assigned to each ground-truth token conditioned on preceding reference tokens (teacher forcing). The function sums these negative log-probabilities across all non-padding positions and divides by the audio duration $d_i$."

---

### `\todo{why capture during runtime}` (Line: "captured at runtime during each experiment’s forward pass\todo{why capture during runtime}")
**Replace with / Add:**
> "...captured online during the scheduled training forward pass to avoid the prohibitive compute cost of running an extra full-dataset forward pass over all 13,807 utterances at each epoch boundary."

---

### `\todo{Why minmax the transcriptions}` (Line: "min-max \todo{Why minmax the transcriptions} normalized the scores")
**Replace with / Add:**
> "Min-max normalization was applied to the resulting numerical WER error scores (scaling the minimum observed WER to 0 and maximum to 1) for diagnostic auditing, though sorting on raw WER yields an identical presentation permutation."

---

## 8. Data Augmentation Comments

### `\todo{briefly describe SpecAugment}` (Line: "SpecAugment \cite{Park2019} technique applied to Whisper’s Log-Mel spectrogram input\todo{briefly describe SpecAugment}")
**Replace with / Add:**
> "SpecAugment \citep{Park2019} is a data augmentation method that acts directly on the 2D log-Mel spectrogram by setting blocks of consecutive time frames (time masking) and frequency channels (frequency masking) to zero, forcing the neural network to utilize context rather than relying on narrow spectral cues."

---

### `\todo{why}` (Line: "training mode only\todo{why}, validation, test and inference features remained unchanged")
**Replace with / Add:**
> "...applied strictly during training mode because evaluation should benchmark the model's accuracy on uncorrupted natural speech; masking during inference would degrade recognition performance."

---

### `\todo{briefly explain time masking and and flow into why the chosen probabilities}` (Line: time masking)
**Replace with / Add:**
> "Time masking zeroes out $t$ consecutive time frames across all 80 frequency channels. In our controlled LB policy, one mask of width up to $T=100$ frames (1.0 second) is applied, simulating brief temporal dropouts or acoustic packet loss."

---

### `\todo{briefly explain frequency masking and why we chose these parameters}` (Line: frequency masking)
**Replace with / Add:**
> "Frequency masking zeroes out $f$ consecutive Mel channels across the entire time axis. In our LB policy, one mask of width up to $F=27$ Mel channels is applied, simulating channel noise or microphone frequency response cut-offs."

---

### `\todo{What is RIR}` (Line: "simulated-RIR\todo{What is RIR}")
**Replace with / Add:**
> "A Room Impulse Response (RIR) mathematically characterizes the acoustic transfer function between a sound source and microphone in an enclosed space, capturing direct sound followed by early reflections and late reverberation decay."

---

### `\todo{clarify rir subsets}` (Line: "simulated RIR subsets\todo{clarify rir subsets}")
**Replace with / Add:**
> "...specifically using the OpenSLR SLR28 simulated RIR subset (comprising 60,000 computationally simulated room impulse responses across diverse room dimensions and reflection coefficients) and point-source noise subset (843 non-speech ambient noises)."

---

### `\todo{why}` (Line: "tempo perturbation with a probability of 0.5 using a factor of either 0.9 or 1.1\todo{why}")
**Replace with / Add:**
> "Tempo perturbation applies pitch-preserving time-stretching ($\pm 10\%$ duration modification) to simulate natural variations in speaking rate without altering the speaker's vocal pitch."

---

## 9. Evaluation Metrics Comments

### `\todo{why WER and CER}` (Line: evaluation metrics)
**Replace with / Add:**
> "WER serves as the standard benchmark metric for speech recognition, quantifying the exact proportion of word errors. CER serves as an essential secondary metric in Bantu languages like Shona, where agglutination forms long multi-morphemic words; CER reveals whether a model has learned the correct stem and phonemes even when minor affixation errors cause whole-word mismatches."

---

### `\todo{elaborate how, what was the role of each package}` (Line: "Jiwer and evaluate packages\todo{elaborate how, what was the role of each package}")
**Replace with / Add:**
> "Hugging Face \texttt{evaluate} was used during training to monitor validation WER at epoch boundaries for checkpoint saving. Standalone \texttt{jiwer} was used in post-training scripts to generate canonical item-level prediction logs, computing exact Levenshtein alignments and recording individual substitution, deletion, and insertion counts."
