# RQ2 Follow-up Validation Battery v1

## Status

Preparation and audit completed on 2026-09-07. This battery is validation-only.
No test asset was opened, no model was trained, and no W&B run was created.

Root: `/ext_data/casper/asr_data/rq2_followup_v1`

## Asset partitions

The SLR28 inventory contains 60,843 decoded-audio content hashes and no
duplicates. Content intersections between train, validation, and test are zero.

| Asset | Train | Validation | Test |
|---|---:|---:|---:|
| Point-source noise | 689 | 77 | 77 |
| Simulated RIR | 48,118 | 5,891 | 5,991 |

- Archive SHA-256: `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7`
- Inventory SHA-256: `98823d3a309441a9b79e0abd968fc8b9887f9a39c7e96735a6834c9be2cad88a`
- Train inventory SHA-256: `70188ea6ac1a3c928de359c26721ce264159122eae9c60132e36254d1abba4ce`
- Validation inventory SHA-256: `683af0a02dd9f08b8bd0e750a5aa793d7297bcca46f1b22416d22ba354c04d95`
- Test inventory SHA-256: `3432d4f1cf37f08ef4cc631ff586c22830ceeb0f98c25fcc0289780053a1ea0e`

## Authoritative validation views

Every view contains the same 1,715 utterance IDs in source-manifest order.

| View | Directory | Repeated noise | Gain-limited rows | RIR tails capped at 30 s |
|---|---|---:|---:|---:|
| D0 | `d0_clean_source_v2` | 0 | 0 | 0 |
| DN 5 dB | `dn_5db_no_loop_v2` | 0 | 717 | 0 |
| DN 10 dB | `dn_10db_no_loop_v2` | 0 | 526 | 0 |
| DN 15 dB | `dn_15db_no_loop_v2` | 0 | 398 | 0 |
| DR | `dr_rir` | 0 | 521 | 54 |
| DNR 5 dB | `dnr_5db_no_loop_v2` | 0 | 691 | 55 |
| DNR 10 dB | `dnr_10db_no_loop_v2` | 0 | 588 | 55 |
| DNR 15 dB | `dnr_15db_no_loop_v2` | 0 | 546 | 55 |

The maximum absolute target-versus-achieved SNR error is below
$5.2\times10^{-7}$ dB. Gain limiting is recorded per row and prevents clipping.
RIR convolution preserves the tail up to Whisper's 30-second input limit; rows
that reach the cap are recorded by `rir_tail_truncated`.

## Superseded outputs

The following immutable directories remain on disk for audit history but must
not be used:

- `d0_clean`: cached clean audio changed gain for 171 rows and capped 21 rows.
- `dn_5db`, `dn_10db`, `dn_15db`: short noises were periodically tiled.
- `dnr_5db`, `dnr_10db`, `dnr_15db`: short noises were periodically tiled.

The versioned directories in the authoritative table correct those defects
without deleting or overwriting prior artifacts.

## Implementation

- `prepare_rq2_asset_partitions.py` creates canonical decoded-audio inventories.
- `prepare_rq2_followup_data.py` creates paired immutable views and refuses test
  assets unless `--allow-test-partition` is explicit.
- `scripts/prepare_rq2_followup_validation.sh` generates only validation assets.
- `tests/test_rq2_followup.py` covers content identity, SNR, speed resampling,
  clean-byte preservation, test locking, pairing, and factorial calculations.

Training manifests and B-series configurations remain intentionally uncreated
until this audit is accepted and their frozen policy is recorded.