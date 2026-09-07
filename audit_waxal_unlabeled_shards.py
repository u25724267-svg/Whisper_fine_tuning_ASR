import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import soundfile as sf
from huggingface_hub import hf_hub_download

from inventory_waxal_unlabeled import (
    CONFIG_NAME,
    DATASET_ID,
    EXPECTED_LANGUAGE,
    EXPECTED_ROWS,
    EXPECTED_SHARDS,
    PINNED_REVISION,
    SPLIT_NAME,
    audio_info,
    sha256_file,
    verify_revision,
)


DEFAULT_CACHE_DIR = Path("/ext_data/casper/huggingface_cache")
DEFAULT_OUTPUT_ROOT = Path("/ext_data/casper/asr_data/waxal_shona_unlabeled_v1")
FIELDS = ("id", "speaker_id", "transcription", "language", "gender", "audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and audit pinned WAXAL Shona shards resumably."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--download-attempts", type=int, default=8)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def shard_filename(shard_index: int) -> str:
    return f"data/ASR/sna/sna-unlabeled-{shard_index:05d}.parquet"


def download_shard(
    shard_index: int, cache_dir: Path, attempts: int
) -> Path:
    filename = shard_filename(shard_index)
    for attempt in range(1, attempts + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=filename,
                    repo_type="dataset",
                    revision=PINNED_REVISION,
                    cache_dir=cache_dir,
                )
            ).resolve()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(60, 2 ** (attempt - 1))
            print(
                f"Shard {shard_index:05d} download attempt {attempt}/{attempts} "
                f"failed: {error}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable download retry state")


def audit_shard(shard_index: int, parquet_path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    missing_fields = Counter()
    languages = Counter()
    genders = Counter()
    sample_rates = Counter()
    channels = Counter()
    formats = Counter()
    rows = []
    invalid_audio_headers = 0
    nonempty_transcriptions = 0

    for batch in parquet.iter_batches(batch_size=64, columns=list(FIELDS)):
        for row in batch.to_pylist():
            for field in FIELDS:
                value = row.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_fields[field] += 1
            transcription = str(row.get("transcription") or "")
            nonempty_transcriptions += bool(transcription.strip())
            language = str(row.get("language") or "")
            gender = str(row.get("gender") or "")
            languages[language] += 1
            genders[gender] += 1

            audio = row.get("audio")
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            try:
                metadata = audio_info(audio)
            except (RuntimeError, TypeError, ValueError, sf.LibsndfileError):
                invalid_audio_headers += 1
                metadata = {
                    "duration": None,
                    "sample_rate": None,
                    "channels": None,
                    "format": None,
                }
            if metadata["sample_rate"] is not None:
                sample_rates[str(metadata["sample_rate"])] += 1
                channels[str(metadata["channels"])] += 1
                formats[str(metadata["format"])] += 1
            rows.append(
                {
                    "id": str(row.get("id") or ""),
                    "speaker_id": str(row.get("speaker_id") or ""),
                    "duration_seconds": metadata["duration"],
                    "audio_sha256": (
                        hashlib.sha256(audio_bytes).hexdigest()
                        if audio_bytes is not None
                        else None
                    ),
                }
            )

    if len(rows) != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Shard {shard_index:05d} row mismatch: "
            f"read {len(rows)}, metadata reports {parquet.metadata.num_rows}"
        )
    return {
        "schema_version": 1,
        "shard_index": shard_index,
        "repository_filename": shard_filename(shard_index),
        "parquet_path": str(parquet_path),
        "parquet_bytes": parquet_path.stat().st_size,
        "parquet_sha256": sha256_file(parquet_path),
        "rows_count": len(rows),
        "rows": rows,
        "missing_fields": dict(sorted(missing_fields.items())),
        "languages": dict(sorted(languages.items())),
        "genders": dict(sorted(genders.items())),
        "sample_rates": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "formats": dict(sorted(formats.items())),
        "invalid_audio_headers": invalid_audio_headers,
        "nonempty_transcriptions": nonempty_transcriptions,
    }


def merge_counters(reports: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counter = Counter()
    for report in reports:
        counter.update(report[field])
    return dict(sorted(counter.items()))


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def combine_reports(reports: list[Dict[str, Any]], complete: bool) -> Dict[str, Any]:
    all_rows = [row for report in reports for row in report["rows"]]
    utterance_ids = [row["id"] for row in all_rows]
    speaker_ids = {row["speaker_id"] for row in all_rows if row["speaker_id"]}
    audio_hashes = [row["audio_sha256"] for row in all_rows if row["audio_sha256"]]
    durations = sorted(
        float(row["duration_seconds"])
        for row in all_rows
        if row["duration_seconds"] is not None
    )
    languages = merge_counters(reports, "languages")
    inspected_rows = len(all_rows)
    invalid_audio_headers = sum(report["invalid_audio_headers"] for report in reports)
    nonempty_transcriptions = sum(
        report["nonempty_transcriptions"] for report in reports
    )
    return {
        "schema_version": 2,
        "dataset": DATASET_ID,
        "config": CONFIG_NAME,
        "split": SPLIT_NAME,
        "pinned_revision": PINNED_REVISION,
        "license": "CC-BY-SA-4.0",
        "provider": "Digital Umuganda / AfriVoice",
        "expected_rows": EXPECTED_ROWS,
        "expected_shards": EXPECTED_SHARDS,
        "audited_shards": len(reports),
        "inspected_rows": inspected_rows,
        "complete": complete,
        "unique_utterance_ids": len(set(utterance_ids)),
        "duplicate_utterance_ids": inspected_rows - len(set(utterance_ids)),
        "unique_nonempty_speaker_ids": len(speaker_ids),
        "unique_audio_hashes": len(set(audio_hashes)),
        "duplicate_audio_rows": len(audio_hashes) - len(set(audio_hashes)),
        "invalid_audio_headers": invalid_audio_headers,
        "nonempty_transcriptions": nonempty_transcriptions,
        "missing_fields": merge_counters(reports, "missing_fields"),
        "languages": languages,
        "genders": merge_counters(reports, "genders"),
        "sample_rates": merge_counters(reports, "sample_rates"),
        "channels": merge_counters(reports, "channels"),
        "formats": merge_counters(reports, "formats"),
        "duration": {
            "hours": sum(durations) / 3600,
            "min_seconds": durations[0],
            "p05_seconds": percentile(durations, 0.05),
            "median_seconds": percentile(durations, 0.5),
            "p95_seconds": percentile(durations, 0.95),
            "max_seconds": durations[-1],
        },
        "source_shards": [
            {
                "index": report["shard_index"],
                "filename": report["repository_filename"],
                "rows": report["rows_count"],
                "bytes": report["parquet_bytes"],
                "sha256": report["parquet_sha256"],
            }
            for report in reports
        ],
        "admission_checks": {
            "all_expected_shards": complete and len(reports) == EXPECTED_SHARDS,
            "expected_full_row_count": complete and inspected_rows == EXPECTED_ROWS,
            "all_languages_shona": set(languages) == {EXPECTED_LANGUAGE},
            "all_transcriptions_empty": nonempty_transcriptions == 0,
            "all_audio_headers_valid": invalid_audio_headers == 0,
            "all_ids_unique": len(utterance_ids) == len(set(utterance_ids)),
            "all_compressed_audio_unique": len(audio_hashes) == len(set(audio_hashes)),
        },
    }


def main() -> None:
    args = parse_args()
    if args.download_attempts < 1:
        raise ValueError("--download-attempts must be at least 1")
    if args.max_shards is not None and not 1 <= args.max_shards <= EXPECTED_SHARDS:
        raise ValueError(f"--max-shards must be between 1 and {EXPECTED_SHARDS}")

    resolved_revision = verify_revision()
    if resolved_revision != PINNED_REVISION:
        raise RuntimeError("Pinned WAXAL revision did not resolve exactly")

    cache_dir = args.cache_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    state_dir = output_root / "shard_audits"
    final_path = output_root / "inventory_full.json"
    shard_limit = args.max_shards or EXPECTED_SHARDS
    if args.max_shards is None and final_path.exists():
        raise FileExistsError(f"Full audit already exists: {final_path}")
    state_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for shard_index in range(shard_limit):
        report_path = state_dir / f"shard-{shard_index:05d}.json"
        print(
            f"[{shard_index + 1}/{shard_limit}] {shard_filename(shard_index)}",
            flush=True,
        )
        parquet_path = download_shard(shard_index, cache_dir, args.download_attempts)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["parquet_sha256"] != sha256_file(parquet_path):
                raise RuntimeError(f"Cached shard changed: {parquet_path}")
            print(f"Reusing completed audit {report_path}", flush=True)
        else:
            report = audit_shard(shard_index, parquet_path)
            atomic_write_json(report_path, report)
            print(
                f"Audited {report['rows_count']} rows -> {report_path}", flush=True
            )
        reports.append(report)

    complete = shard_limit == EXPECTED_SHARDS
    result = combine_reports(reports, complete=complete)
    result["resolved_revision"] = resolved_revision
    result["preparer"] = str(Path(__file__).resolve())
    result["preparer_sha256"] = sha256_file(Path(__file__).resolve())
    output_path = final_path if complete else output_root / f"inventory_{shard_limit}_shards.json"
    atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()