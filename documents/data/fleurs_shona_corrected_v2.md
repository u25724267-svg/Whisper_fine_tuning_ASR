# FLEURS Shona Corrected Protocol v2

## Purpose

Provide an immutable FLEURS Shona (`sn_zw`) protocol for later cross-corpus OOD
evaluation without repeating the digit-removal and identifier defects in the
exploratory preparation.

## Source

- Dataset: `google/fleurs`
- Configuration: `sn_zw`
- Revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
- Audio: reused unchanged from the exploratory local preparation
- Output: `/ext_data/casper/asr_data/fleurs_shona_corrected_v2`

## Normalization

Transcripts use Unicode NFC, case folding, ASCII apostrophes, spaces in place of
dashes and other punctuation/symbols, and collapsed whitespace. Letters,
combining marks, Unicode numbers, and apostrophes are retained. Original text is
preserved in `text_raw`.

Unicode numbers include decimal digits and characters such as `¾`. They are not
automatically verbalized using English assumptions.

## Identifier correction

FLEURS source IDs identify sentence prompts and repeat when multiple speakers
record the same sentence. The exploratory train manifest therefore had 2,463
recordings but only 1,403 unique IDs. Version 2 derives the utterance ID from the
unique audio filename and preserves the exploratory ID in `v1_id` and the source
sentence ID in `source_id`.

## Results of preparation

| Split | Rows | Hours | Text changed from v1 | Rows with numbers |
|---|---:|---:|---:|---:|
| Train | 2,463 | 9.98 | 556 | 531 |
| Validation | 393 | 1.55 | 58 | 58 |
| Test | 925 | 3.82 | 179 | 167 |

All rows and official split memberships were retained.

## Audio audit

- 3,781 readable mono recordings at 16 kHz
- Maximum manifest-duration error: 0 seconds
- Duplicate audio within splits: 0
- Duplicate audio across FLEURS splits: 0
- Decoded-PCM duplicates shared with WAXAL: 0

PCM hashes use the same canonical procedure as the WAXAL audit: float32 decode,
clipping and rounding to little-endian int16, with sample rate and channel count
included in the SHA-256 input.

## Locked use

FLEURS is not used for training or curriculum selection. Full validation/test
metrics remain uncomputed until the RQ1 curriculum is selected using WAXAL
validation. The exploratory FLEURS-only and WAXAL+FLEURS runs remain pilots and
are not confirmatory OOD evidence.