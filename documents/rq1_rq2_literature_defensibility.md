# RQ1 and RQ2 Literature Defensibility

## Verdict

RQ1 and RQ2 are defensible as controlled, literature-informed experiments on
low-resource Whisper fine-tuning. They are not all direct replications. C1 is
the closest algorithmic replication; C2--C7 and the waveform policy adapt or
combine published ideas under study-specific definitions. The thesis must make
that distinction explicit.

The strongest aspects are the matched controls, immutable manifests, pinned
model revision, saved curriculum orders, speaker-disjoint evaluation, retained
negative results, additional seeds for selected conditions, and paired cluster
bootstraps. Literature motivates the hypotheses; these controls make the local
Shona tests defensible within their stated scope.

## RQ1 literature map

| Condition | Literature basis | Classification | Defensible claim |
|---|---|---|---|
| C0 random | Conventional shuffled empirical-risk training | Control | Matched reference condition, not a literature contribution |
| C1 duration SortaGrad | Deep Speech 2 orders shorter utterances first, then shuffles later epochs | Close replication transferred to Whisper | Tests whether SortaGrad transfers from CTC-style training to pretrained sequence-to-sequence fine-tuning |
| C2 strict SNR | SNR curricula in noise-robust ASR, including Braun et al. | Literature-motivated adaptation | Tests a frozen active-speech SNR proxy under strict full-data ordering; it is not a replication of ACCAN or another published SNR schedule |
| C3 SNR-duration | Static duration/SNR difficulty criteria in ASR curriculum literature | Study-specific composite | Equal weights, percentile ranks, and strict every-epoch order are original operational choices |
| C4 cumulative tiers | Bengio-style easy-to-hard exposure, self-paced admission, and ASR pacing studies | Literature-motivated adaptation | Tests fixed acoustic ranking plus cumulative exposure, not Kumar et al.'s jointly optimized self-paced objective |
| C4R random pacing | Matched exposure and pool-size control | Study-specific control | Isolates difficulty ranking from generic early pool restriction |
| C5 S2S-NLL | Model-loss criteria in Karakasidis et al. | Substantial adaptation | Uses online, path-dependent duration-normalized token NLL; it is not a hypothesis margin or exact published implementation |
| C6 utterance WER | Dynamic utterance-WER criterion in Karakasidis et al. | Closest direct dynamic criterion, adapted | Uses greedy Whisper decoding and strict next-epoch ordering without published uniform mixing |
| C7 acoustic + S2S | Hybrid curriculum principle | Original composite adaptation | Equal acoustic/model weighting and the exact nested percentile score are study-designed |

Karakasidis et al. evaluated duration, sequence loss and utterance WER, dynamic
updates, difficulty mixing, and pacing in end-to-end ASR. Their best 2022 result
used WER scoring with uniform difficulty mixing. Our confirmatory C5--C7 remove
that mixing to isolate the score, so they test related mechanisms rather than
reproduce the published best system. This difference is scientifically useful
and must be stated.

## RQ2 literature map

| Component | Literature basis | Study-specific choice |
|---|---|---|
| Tempo factors 0.9/1.1 | Ko et al. recommend 0.9, 1.0, and 1.1 speed variants | Our pitch-preserving time stretch, 0.5 probability, and one-view replacement differ from Kaldi resampling and threefold expansion |
| Point-source noise | Ko et al. and SLR28 support point-source noise augmentation | Probability 0.5, continuous 10--25 dB SNR, and RMS scaling are local choices |
| Simulated RIR | Ko et al. show simulated RIR convolution improves reverberant ASR | Probability 0.3, RIR sampling, and convolution truncation are local choices |
| SLR28 | Official Apache-2.0 RIR/noise resource associated with Ko et al. | Restricting to simulated RIRs and 843 point-source noises is the frozen study subset |
| Composition | Standard waveform augmentation families | Tempo then RIR then noise then anti-clipping gain is a study-specific composition |
| SpecAugment | Park et al. establish time/frequency masking | Mild 0.05/8 settings, always-on use, and the mixed-50 policy are Whisper adaptations, not canonical Park policies |
| A0--A3 factorial | Standard controlled factorial logic | The exact curriculum-by-waveform combination is the thesis experiment, not a published recipe |

Because the implementation uses pitch-preserving time stretching, thesis text
should call it **tempo perturbation using Ko-inspired factors**, not a direct
implementation of Kaldi speed perturbation. The independent transform
probabilities imply 82.5% of rows receive at least one transform and 7.5%
receive all three. These quantities communicate the actual study-specific dose.

## Statistical defensibility

- The speaker-disjoint WAXAL split protects against direct speaker leakage and
  supports claims about unseen speakers within this corpus. Each evaluation
  split contains only 24 speaker clusters, so broad population claims are not
  justified.
- Paired speaker-cluster bootstrap intervals preserve paired utterances and
  within-speaker dependence. They quantify evaluation-sample uncertainty
  conditional on fixed checkpoints, not training or model-selection uncertainty.
- RQ1 screened several methods without multiplicity adjustment. Report that an
  **unadjusted bootstrap interval excluded zero**, not that a screen result was
  statistically significant in a family-wise sense.
- C3 is deterministic and one distinct model is reused against three random
  controls. This shows consistent ranking across control shuffles, not three
  independent C3 replications. A1's zero SD is structural reuse.
- WAXAL test results contributed to selecting C3. Its test result is therefore
  selection-stage evidence, not a pristine final confirmation. Corrected FLEURS
  was opened later and provides a stronger independent OOD check, although its
  speakers are unavailable and prompt-clustered uncertainty cannot model
  speaker dependence.
- RQ2 interaction values are descriptive difference-in-differences. No
  interaction-cluster bootstrap was run, so claim that no beneficial interaction
  was observed, not that an interaction was statistically absent.
- The 0.5 absolute-WER-point threshold is a preregistered practical decision
  rule, not a literature-derived significance threshold.

## Thesis-safe conclusions

The investigated curricula did not improve consistently across WAXAL
validation, WAXAL test, and corrected FLEURS. C3 had a modest WAXAL-specific
ranking advantage under one deterministic implementation but lacked consistent
validation and OOD support. C4/C4R results are consistent with early pool
restriction being harmful. The frozen waveform policy did not improve primary
validation performance and was not promoted. These conclusions apply to
Whisper Base, Shona WAXAL, the three-epoch budget, and the exact operational
definitions tested; they do not establish universal curriculum or augmentation
effects.

## Primary references

- Bengio et al. (2009), [Curriculum Learning](https://doi.org/10.1145/1553374.1553380).
- Kumar et al. (2010), [Self-Paced Learning for Latent Variable Models](https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html).
- Amodei et al. (2016), [Deep Speech 2](https://arxiv.org/abs/1512.02595).
- Braun et al. (2017), [A Curriculum Learning Method for Improved Noise Robustness in ASR](https://doi.org/10.23919/EUSIPCO.2017.8081267).
- Karakasidis et al. (2022), [Comparison and Analysis of New Curriculum Criteria for End-to-End ASR](https://doi.org/10.21437/Interspeech.2022-10046).
- Karakasidis et al. (2024), [expanded Speech Communication study](https://doi.org/10.1016/j.specom.2024.103113).
- Kuznetsova et al. (2022), [Curriculum Optimization for Low-Resource Speech Recognition](https://doi.org/10.1109/ICASSP43922.2022.9746674).
- Park et al. (2019), [SpecAugment](https://doi.org/10.21437/Interspeech.2019-2680).
- Ko et al. (2015), [Audio Augmentation for Speech Recognition](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html).
- Ko et al. (2017), [Reverberant Speech Augmentation](https://doi.org/10.1109/ICASSP.2017.7953152).
- [OpenSLR SLR28](https://www.openslr.org/28/), Room Impulse Response and Noise Database.

## Remaining documentation work

Before submission, convert these links into the university's required
bibliography style and add primary citations for Whisper, WAXAL, FLEURS, JiWER,
the selected bootstrap reference, and the specific Whisper masking API.
Preserve the literature/adaptation classifications above in Chapters 2 and 3.