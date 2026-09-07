import argparse
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import datasets
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import HfApi


DATASET_ID = "google/WaxalNLP"
CONFIG_NAME = "sna_asr"
SPLIT_NAME = "unlabeled"
PINNED_REVISION = "5f4d8ca24f2b9d168b2ee545f1febaaff4b40580"
EXPECTED_LANGUAGE = "sna"
EXPECTED_ROWS = 85_384
EXPECTED_SHARDS = 52
DEFAULT_MAX_ROWS = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a read-only inventory of the pinned WAXAL Shona unlabeled split. "
            "The default inspects a bounded sample; use --full explicitly for all rows."
        )
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    scope.add_argument("--full", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional immutable JSON output path; existing files are never overwritten.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory (prefer local scratch storage).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def audio_info(audio: Any) -> Dict[str, Any]:
    if not isinstance(audio, dict):
        raise ValueError("audio is not a dictionary")
    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")
    if audio_bytes is not None:
        info = sf.info(io.BytesIO(audio_bytes))
    elif audio_path:
        info = sf.info(audio_path)
    else:
        raise ValueError("audio has neither bytes nor path")
    if info.samplerate <= 0 or info.frames <= 0 or info.channels <= 0:
        raise ValueError("audio header has non-positive dimensions")
    duration = info.frames / info.samplerate
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("audio duration is not finite and positive")
    return {
        "duration": duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def verify_revision() -> str:
    resolved_revision = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION).sha
    if resolved_revision != PINNED_REVISION:
        raise RuntimeError(
            f"Revision mismatch: expected {PINNED_REVISION}, got {resolved_revision}"
        )
    return resolved_revision


def unlabeled_shard_urls() -> list[str]:
    base_url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{PINNED_REVISION}"
        "/data/ASR/sna"
    )
    return [
        f"{base_url}/sna-unlabeled-{shard_index:05d}.parquet"
        for shard_index in range(EXPECTED_SHARDS)
    ]


def inventory(max_rows: Optional[int], cache_dir: Optional[Path]) -> Dict[str, Any]:
    resolved_revision = verify_revision()
    stream = load_dataset(
        "parquet",
        data_files={SPLIT_NAME: unlabeled_shard_urls()},
        split=SPLIT_NAME,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    ).cast_column("audio", Audio(decode=False))

    missing_fields = Counter()
    languages = Counter()
    genders = Counter()
    sample_rates = Counter()
    channels = Counter()
    formats = Counter()
    speaker_ids = set()
    utterance_ids = set()
    duplicate_ids = 0
    nonempty_transcriptions = 0
    invalid_audio = 0
    durations = []

    for row_index, row in enumerate(stream):
        if max_rows is not None and row_index >= max_rows:
            break
        for field in ("id", "speaker_id", "transcription", "language", "gender", "audio"):
            if is_missing(row.get(field)):
                missing_fields[field] += 1

        utterance_id = str(row.get("id") or "")
        if utterance_id in utterance_ids:
            duplicate_ids += 1
        else:
            utterance_ids.add(utterance_id)

        speaker_id = str(row.get("speaker_id") or "")
        if speaker_id:
            speaker_ids.add(speaker_id)
        language = str(row.get("language") or "")
        gender = str(row.get("gender") or "")
        languages[language] += 1
        genders[gender] += 1
        if str(row.get("transcription") or "").strip():
            nonempty_transcriptions += 1

        try:
            metadata = audio_info(row.get("audio"))
        except (RuntimeError, TypeError, ValueError, sf.LibsndfileError):
            invalid_audio += 1
            continue
        durations.append(metadata["duration"])
        sample_rates[str(metadata["sample_rate"])] += 1
        channels[str(metadata["channels"])] += 1
        formats[metadata["format"]] += 1

    inspected_rows = sum(languages.values())
    if inspected_rows == 0:
        raise RuntimeError("The stream returned no rows")
    duration_summary = None
    if durations:
        sorted_durations = sorted(durations)
        duration_summary = {
            "hours": sum(sorted_durations) / 3600,
            "min_seconds": sorted_durations[0],
            "median_seconds": sorted_durations[len(sorted_durations) // 2],
            "max_seconds": sorted_durations[-1],
        }

    return {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "config": CONFIG_NAME,
        "split": SPLIT_NAME,
        "pinned_revision": PINNED_REVISION,
        "resolved_revision": resolved_revision,
        "license": "CC-BY-SA-4.0",
        "provider": "Digital Umuganda / AfriVoice",
        "expected_rows": EXPECTED_ROWS,
        "expected_shards": EXPECTED_SHARDS,
        "scope": "full" if max_rows is None else "bounded_sample",
        "requested_max_rows": max_rows,
        "inspected_rows": inspected_rows,
        "complete": max_rows is None,
        "unique_utterance_ids": len(utterance_ids),
        "duplicate_utterance_ids": duplicate_ids,
        "unique_nonempty_speaker_ids": len(speaker_ids),
        "nonempty_transcriptions": nonempty_transcriptions,
        "missing_fields": dict(sorted(missing_fields.items())),
        "languages": dict(sorted(languages.items())),
        "genders": dict(sorted(genders.items())),
        "invalid_audio_headers": invalid_audio,
        "duration": duration_summary,
        "sample_rates": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "formats": dict(sorted(formats.items())),
        "admission_checks": {
            "revision_matches": resolved_revision == PINNED_REVISION,
            "all_languages_shona": set(languages) == {EXPECTED_LANGUAGE},
            "all_transcriptions_empty": nonempty_transcriptions == 0,
            "all_audio_headers_valid": invalid_audio == 0,
            "all_ids_unique_within_inventory": duplicate_ids == 0,
            "expected_full_row_count": max_rows is None and inspected_rows == EXPECTED_ROWS,
        },
        "software": {
            "datasets": datasets.__version__,
        },
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    args = parse_args()
    if not args.full and args.max_rows < 1:
        raise ValueError("--max-rows must be at least 1")
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    result = inventory(None if args.full else args.max_rows, cache_dir)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()