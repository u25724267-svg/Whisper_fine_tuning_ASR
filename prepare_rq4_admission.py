import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from inventory_waxal_unlabeled import PINNED_REVISION, sha256_file


DEFAULT_AUDIT_ROOT = Path("/ext_data/casper/asr_data/waxal_shona_unlabeled_v1")
DEFAULT_GOLD_ROOT = Path("/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2")
DEFAULT_WAXAL_HASHES = Path(
    "/home/casper/Speech/data/waxal/sna_asr/sna_asr_audio_hashes.csv"
)
DEFAULT_FLEURS_HASHES = Path(
    "/ext_data/casper/asr_data/fleurs_shona_corrected_v2/audio_hashes.csv"
)
DEFAULT_OUTPUT_DIR = Path("/ext_data/casper/asr_data/waxal_shona_unlabeled_admission_v1")
SPLITS = ("train", "validation", "test")
MIN_DURATION = 1.0
MAX_DURATION = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the model-independent RQ4 WAXAL admission manifests."
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--waxal-hashes", type=Path, default=DEFAULT_WAXAL_HASHES)
    parser.add_argument("--fleurs-hashes", type=Path, default=DEFAULT_FLEURS_HASHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Validation mode: process only the first N shards.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_hashes(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as source:
        return {row["pcm_sha256"] for row in csv.DictReader(source)}


def canonical_pcm_hash(audio_bytes: bytes) -> Dict[str, Any]:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if not len(audio) or sample_rate <= 0:
        raise ValueError("Decoded audio is empty or has an invalid sample rate")
    if not np.isfinite(audio).all():
        raise ValueError("Decoded audio contains non-finite samples")
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2", copy=False)
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{audio.shape[1]}:".encode())
    digest.update(pcm.tobytes())
    return {
        "pcm_sha256": digest.hexdigest(),
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(len(audio)),
        "duration": len(audio) / sample_rate,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    audit_root = args.audit_root.expanduser().resolve()
    gold_root = args.gold_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    final_audit_path = audit_root / "inventory_full.json"
    if not final_audit_path.is_file():
        raise FileNotFoundError(f"Missing full WAXAL audit: {final_audit_path}")
    final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    if not final_audit.get("complete"):
        raise ValueError("WAXAL source audit is not complete")
    if final_audit.get("pinned_revision") != PINNED_REVISION:
        raise ValueError("WAXAL source revision does not match the frozen revision")

    if args.max_shards is not None and args.max_shards < 1:
        raise ValueError("--max-shards must be at least 1")
    source_shards = final_audit["source_shards"]
    if args.max_shards is not None:
        source_shards = source_shards[: args.max_shards]
    validation_mode = len(source_shards) != final_audit["expected_shards"]

    gold_rows = {
        split: load_jsonl(gold_root / f"{split}.jsonl") for split in SPLITS
    }
    gold_ids = {str(row["id"]) for rows in gold_rows.values() for row in rows}
    gold_speakers = {
        split: {str(row["speaker_id"]) for row in rows if row.get("speaker_id")}
        for split, rows in gold_rows.items()
    }
    excluded_eval_speakers = gold_speakers["validation"] | gold_speakers["test"]
    waxal_hashes = load_hashes(args.waxal_hashes.expanduser().resolve())
    fleurs_hashes = load_hashes(args.fleurs_hashes.expanduser().resolve())

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Admission protocol already complete: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "shard_admission"
    state_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    eligible_rows = []
    rejected_rows = []
    rejection_counts = Counter()
    split_speaker_overlap_rows = Counter()
    decoded_hashes = set()

    for position, shard in enumerate(source_shards, start=1):
        shard_index = int(shard["index"])
        report_path = audit_root / "shard_audits" / f"shard-{shard_index:05d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        parquet_path = Path(report["parquet_path"])
        if sha256_file(parquet_path) != report["parquet_sha256"]:
            raise ValueError(f"Parquet hash mismatch: {parquet_path}")
        state_path = state_dir / f"shard-{shard_index:05d}.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["parquet_sha256"] != report["parquet_sha256"]:
                raise ValueError(f"Admission checkpoint hash mismatch: {state_path}")
            shard_rows = state["rows"]
            if len(shard_rows) != report["rows_count"]:
                raise ValueError(f"Admission checkpoint row mismatch: {state_path}")
            print(f"Reusing completed admission checkpoint {state_path}", flush=True)
            for row in shard_rows:
                decoded_hashes.add(row["pcm_sha256"])
                all_rows.append(row)
                for split in row["gold_speaker_splits"]:
                    split_speaker_overlap_rows[split] += 1
                if row["rejection_reasons"]:
                    rejected_rows.append(row)
                    rejection_counts.update(set(row["rejection_reasons"]))
                else:
                    eligible_rows.append(row)
            print(
                f"[{position}/{len(source_shards)}] shard {shard_index:05d}: "
                f"{len(shard_rows)} rows, {len(eligible_rows)} eligible total",
                flush=True,
            )
            continue

        parquet = pq.ParquetFile(parquet_path)
        row_index = 0
        shard_rows = []
        for batch in parquet.iter_batches(
            batch_size=32,
            columns=["id", "speaker_id", "language", "gender", "audio"],
        ):
            for source_row in batch.to_pylist():
                audio = source_row.get("audio")
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if audio_bytes is None:
                    raise ValueError(
                        f"Missing embedded audio in shard {shard_index}, row {row_index}"
                    )
                decoded = canonical_pcm_hash(audio_bytes)
                utterance_id = str(source_row.get("id") or "")
                speaker_id = str(source_row.get("speaker_id") or "")
                reasons = []
                overlapping_splits = []
                if utterance_id in gold_ids:
                    reasons.append("gold_id_overlap")
                for split, speakers in gold_speakers.items():
                    if speaker_id and speaker_id in speakers:
                        overlapping_splits.append(split)
                        split_speaker_overlap_rows[split] += 1
                if speaker_id in excluded_eval_speakers:
                    reasons.append("evaluation_speaker_overlap")
                if decoded["duration"] < MIN_DURATION:
                    reasons.append("duration_below_1s")
                if decoded["duration"] > MAX_DURATION:
                    reasons.append("duration_above_30s")
                if decoded["channels"] != 1:
                    reasons.append("non_mono")
                if decoded["pcm_sha256"] in waxal_hashes:
                    reasons.append("gold_waxal_pcm_overlap")
                if decoded["pcm_sha256"] in fleurs_hashes:
                    reasons.append("fleurs_pcm_overlap")
                if decoded["pcm_sha256"] in decoded_hashes:
                    reasons.append("unlabeled_pcm_duplicate")
                decoded_hashes.add(decoded["pcm_sha256"])

                row = {
                    "id": utterance_id,
                    "speaker_id": speaker_id,
                    "language": str(source_row.get("language") or ""),
                    "gender": str(source_row.get("gender") or ""),
                    "duration": decoded["duration"],
                    "sample_rate": decoded["sample_rate"],
                    "channels": decoded["channels"],
                    "pcm_sha256": decoded["pcm_sha256"],
                    "source_shard_index": shard_index,
                    "source_row_index": row_index,
                    "source_parquet": str(parquet_path),
                    "gold_speaker_splits": overlapping_splits,
                    "eligible": not reasons,
                    "rejection_reasons": reasons,
                }
                shard_rows.append(row)
                all_rows.append(row)
                if reasons:
                    rejected_rows.append(row)
                    rejection_counts.update(set(reasons))
                else:
                    eligible_rows.append(row)
                row_index += 1
        if row_index != report["rows_count"]:
            raise ValueError(
                f"Shard {shard_index} row mismatch: {row_index} vs {report['rows_count']}"
            )
        write_json(
            state_path,
            {
                "schema_version": 1,
                "shard_index": shard_index,
                "parquet_sha256": report["parquet_sha256"],
                "rows": shard_rows,
            },
        )
        print(
            f"[{position}/{len(source_shards)}] shard {shard_index:05d}: "
            f"{row_index} rows, {len(eligible_rows)} eligible total",
            flush=True,
        )

    all_path = output_dir / "all_rows.jsonl"
    eligible_path = output_dir / "eligible.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    write_jsonl(all_path, all_rows)
    write_jsonl(eligible_path, eligible_rows)
    write_jsonl(rejected_path, rejected_rows)
    summary = {
        "schema_version": 1,
        "protocol": "waxal_shona_unlabeled_admission_v1",
        "validation_mode": validation_mode,
        "source_revision": PINNED_REVISION,
        "source_audit": str(final_audit_path),
        "source_audit_sha256": sha256_file(final_audit_path),
        "processed_shards": len(source_shards),
        "rows": len(all_rows),
        "hours": sum(row["duration"] for row in all_rows) / 3600,
        "eligible_rows": len(eligible_rows),
        "eligible_hours": sum(row["duration"] for row in eligible_rows) / 3600,
        "rejected_rows": len(rejected_rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "gold_speakers": {
            split: len(speakers) for split, speakers in gold_speakers.items()
        },
        "unlabeled_rows_with_gold_speaker": dict(
            sorted(split_speaker_overlap_rows.items())
        ),
        "duration_gate_seconds": [MIN_DURATION, MAX_DURATION],
        "manifests": {
            "all": {"path": str(all_path), "sha256": sha256_file(all_path)},
            "eligible": {
                "path": str(eligible_path),
                "sha256": sha256_file(eligible_path),
            },
            "rejected": {
                "path": str(rejected_path),
                "sha256": sha256_file(rejected_path),
            },
        },
        "reference_hashes": {
            "waxal": {
                "path": str(args.waxal_hashes.expanduser().resolve()),
                "sha256": sha256_file(args.waxal_hashes.expanduser().resolve()),
            },
            "fleurs": {
                "path": str(args.fleurs_hashes.expanduser().resolve()),
                "sha256": sha256_file(args.fleurs_hashes.expanduser().resolve()),
            },
        },
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()