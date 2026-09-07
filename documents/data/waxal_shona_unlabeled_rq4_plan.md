# WAXAL Shona Unlabeled Pool for RQ4

## Selected candidate

- Dataset: `google/WaxalNLP`
- Configuration: `sna_asr`
- Split: `unlabeled`
- Pinned revision: `5f4d8ca24f2b9d168b2ee545f1febaaff4b40580`
- Rows reported by the dataset server: 85,384
- Parquet payload: approximately 26.5 GB
- In-memory bytes reported by the dataset server: 27,406,469,998
- Shona provider: Digital Umuganda / AfriVoice
- License: CC-BY-SA-4.0

The dataset card states that the unlabeled split contains samples without a
corresponding transcription. This is a distinct source split, not labeled WAXAL
with transcripts hidden. The currently local WAXAL tree contains only labeled
train, validation, and test exports; the unlabeled split has not been downloaded.

## Admission gate

The source is selected as the RQ4 candidate but is not yet admitted to training.
Before pseudo-label generation, a pinned preparation must:

1. Verify every retained row has language `sna` and no transcription.
2. Decode as finite mono audio and record exact duration and PCM SHA-256.
3. Remove IDs and decoded-PCM hashes overlapping any labeled WAXAL split.
4. Remove decoded-PCM hashes overlapping corrected FLEURS.
5. Exclude every speaker appearing in speaker-disjoint validation or test.
6. Deduplicate the unlabeled pool by decoded PCM and record duplicate groups.
7. Reject empty, corrupt, shorter-than-1-second, and longer-than-30-second audio.
8. Select a deterministic 20–40-hour pool without using teacher confidence or
   test performance.
9. Record source revision, license, row counts, hours, speaker counts, rejection
   reasons, manifests, and all hashes.
10. Draw a stratified manual-audit sample before teacher inference.

The target is 40 usable hours if available after filtering. A smaller retained
pool is accepted only if at least 20 hours remain; otherwise RQ4 is reported as
a feasibility failure.

## Deferred dependencies

Teacher selection remains result-dependent. The teacher must be frozen from
supervised validation results only after the RQ2 primary analysis is complete.
Confidence calibration, pseudo-label filtering, student weighting, and optional
confidence pacing are not part of the unconditional RQ2 queue.