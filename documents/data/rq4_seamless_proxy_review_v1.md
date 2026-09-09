# RQ4 SeamlessM4T-v2 Proxy Review v1

## Decision

`facebook/seamless-m4t-v2-large` is technically eligible as an independent
proxy at revision `5f8cc790b19fc3f67a61c105133b20b34e3dcb76`. Shona language
code `sna` supports source speech and target text. The dedicated
`SeamlessM4Tv2ForSpeechToText` class is used with 16 kHz mono audio, greedy
decoding, and `tgt_lang="sna"`.

The model is zero-shot for Shona in the Seamless paper. API support is therefore
not evidence of adequate Shona accuracy. It is a distinct architecture/provider
proxy, not ground truth and not statistically independent of all public data.
Proxy use is retained only if labelled-training speaker-held-out evaluation
reaches ROC-AUC at least 0.70 for teacher WER above 50% and exceeds the
confidence-only AUC.

## License gate

Model weights are CC BY-NC 4.0. Attribution is required and commercial use is
prohibited. The inference script requires an explicit
`--confirm-noncommercial-license` flag. This records user confirmation but is
not legal advice or institutional approval. No model download or proxy inference
may start until the researcher confirms that the university thesis use is
noncommercial and permitted by institutional policy.

The researcher confirmed approved noncommercial university Master's thesis use
on 2026-09-08. This satisfies the study's launch gate while retaining the model
card, license, revision, and required attribution in immutable provenance.

## Pilot gate

Before full labelled or shortlist inference:

1. pin the model and processor to the revision above;
2. run 32--64 labelled WAXAL-training items across speakers, durations, and
   teacher-error strata;
3. compare repeated greedy runs and batch sizes 1/2/4;
4. retain at least 20% measured GPU-memory headroom;
5. verify raw/normalized text, Shona token `__sna__`, EOS handling, and
   normalized character disagreement; and
6. run the complete labelled-training proxy pass before the unlabeled shortlist.

The 32-row error-stratified pilot passed at batch 1 with 78.96% reserved-memory
headroom, no empty outputs, and no maximum-length outputs. A repeated batch-1
run was byte-identical. Batches 2 and 4 changed three of 32 transcripts despite
greedy decoding, so batch 1 is frozen for all proxy inference. The pinned model
uses token ID 3 as both decoder start and EOS and token ID 0 as padding; token
audits retain the initial 3 and stop at the next EOS.

## Primary sources

- [Official SeamlessM4T-v2 model card](https://huggingface.co/facebook/seamless-m4t-v2-large).
- [Pinned model revision](https://huggingface.co/facebook/seamless-m4t-v2-large/tree/5f8cc790b19fc3f67a61c105133b20b34e3dcb76).
- [Transformers 4.46.3 SeamlessM4T-v2 documentation](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/seamless_m4t_v2).
- [CC BY-NC 4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/).
- [Seamless paper](https://arxiv.org/abs/2312.05187).