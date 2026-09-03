import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import soundfile as sf


DEFAULT_V1_DIR = Path("/ext_data/casper/whisper_data/waxal_fleurs/sna_asr")
DEFAULT_OUTPUT_DIR = Path("/ext_data/casper/asr_data/fleurs_shona_corrected_v2")
DEFAULT_WAXAL_HASHES = Path(
    "/home/casper/Speech/data/waxal/sna_asr/sna_asr_audio_hashes.csv"
)
SPLITS = ("train", "validation", "test")
APOSTROPHES = {"'", "‘", "’", "‚", "‛", "′", "ʼ"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create corrected immutable FLEURS Shona v2 manifests from v1."
    )
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--waxal-hashes", type=Path, default=DEFAULT_WAXAL_HASHES)
    parser.add_argument("--hash-workers", type=int, default=8)
    return parser.parse_args()


def normalize_asr_text_v2(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    cleaned_characters = []
    for character in normalized:
        category = unicodedata.category(character)
        if character in APOSTROPHES:
            cleaned_characters.append("'")
        elif character.isspace():
            cleaned_characters.append(" ")
        elif category[0] in {"L", "M", "N"}:
            cleaned_characters.append(character)
        else:
            cleaned_characters.append(" ")
    return re.sub(r"\s+", " ", "".join(cleaned_characters)).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def decoded_pcm_hash(row: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = Path(row["audio_filepath"])
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing FLEURS audio: {audio_path}")
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    measured_duration = len(audio) / sample_rate
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2", copy=False)
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{audio.shape[1]}:".encode())
    digest.update(pcm.tobytes())
    return {
        "split": row["source_split"],
        "id": row["id"],
        "audio_filepath": str(audio_path),
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "frames": len(audio),
        "measured_duration": measured_duration,
        "manifest_duration": float(row["duration"]),
        "duration_error": abs(measured_duration - float(row["duration"])),
        "pcm_sha256": digest.hexdigest(),
    }


def load_waxal_hashes(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing WAXAL audio hash file: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        return {row["pcm_sha256"] for row in csv.DictReader(source)}


def main() -> None:
    args = parse_args()
    v1_dir = args.v1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    waxal_hash_path = args.waxal_hashes.expanduser().resolve()
    summary_path = output_dir / "preparation_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Corrected FLEURS protocol already exists at {output_dir}; "
            "choose a new --output-dir"
        )
    if args.hash_workers < 1:
        raise ValueError("--hash-workers must be at least 1")

    v1_summary_path = v1_dir / "preparation_summary.json"
    if not v1_summary_path.is_file():
        raise FileNotFoundError(f"Missing v1 preparation summary: {v1_summary_path}")
    v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8"))
    v1_manifest_paths = {
        split: v1_dir / "manifests" / f"fleurs_{split}.jsonl" for split in SPLITS
    }
    missing_manifests = [str(path) for path in v1_manifest_paths.values() if not path.is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing v1 FLEURS manifests: {missing_manifests}")

    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_rows_by_split = {}
    split_audits = {}
    all_corrected_rows = []
    for split in SPLITS:
        v1_rows = read_jsonl(v1_manifest_paths[split])
        corrected_rows = []
        changed_rows = 0
        raw_digit_rows = 0
        corrected_digit_rows = 0
        v1_id_counts = Counter(str(row["id"]) for row in v1_rows)
        for row in v1_rows:
            text_raw = str(row.get("text_raw") or "")
            corrected_text = normalize_asr_text_v2(text_raw)
            if not corrected_text:
                raise ValueError(f"Empty corrected text for {split} row {row.get('id')}")
            if corrected_text != row["text"]:
                changed_rows += 1
            raw_digit_rows += any(character.isnumeric() for character in text_raw)
            corrected_digit_rows += any(character.isnumeric() for character in corrected_text)
            utterance_id = Path(row["audio_filepath"]).stem
            corrected_row = {
                **row,
                "id": utterance_id,
                "v1_id": row["id"],
                "text": corrected_text,
            }
            if any(
                not (
                    character.isalpha()
                    or character.isnumeric()
                    or character.isspace()
                    or character == "'"
                    or unicodedata.category(character).startswith("M")
                )
                for character in corrected_text
            ):
                raise ValueError(f"Unexpected corrected character in row {row.get('id')}")
            for identity_field in (
                "audio_filepath",
                "duration",
                "source",
                "source_split",
                "source_id",
                "text_raw",
            ):
                if corrected_row.get(identity_field) != row.get(identity_field):
                    raise RuntimeError(
                        f"Identity field changed for {split} row {row.get('id')}: "
                        f"{identity_field}"
                    )
            corrected_rows.append(corrected_row)

        ids = [str(row["id"]) for row in corrected_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in corrected {split} split")
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, corrected_rows)
        corrected_rows_by_split[split] = corrected_rows
        all_corrected_rows.extend(corrected_rows)
        split_audits[split] = {
            "rows": len(corrected_rows),
            "hours": sum(float(row["duration"]) for row in corrected_rows) / 3600,
            "changed_from_v1": changed_rows,
            "ids_reassigned": len(corrected_rows),
            "v1_duplicate_id_groups": sum(count > 1 for count in v1_id_counts.values()),
            "v1_rows_in_duplicate_id_groups": sum(
                count for count in v1_id_counts.values() if count > 1
            ),
            "raw_rows_with_numbers": raw_digit_rows,
            "corrected_rows_with_numbers": corrected_digit_rows,
            "v1_manifest": str(v1_manifest_paths[split]),
            "v1_sha256": sha256_file(v1_manifest_paths[split]),
            "v2_manifest": str(output_path),
            "v2_sha256": sha256_file(output_path),
        }

    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        audio_audits = list(pool.map(decoded_pcm_hash, all_corrected_rows))

    audio_hash_path = output_dir / "audio_hashes.csv"
    audio_columns = list(audio_audits[0])
    with audio_hash_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=audio_columns)
        writer.writeheader()
        writer.writerows(audio_audits)

    if any(row["sample_rate"] != 16000 for row in audio_audits):
        raise ValueError("Corrected FLEURS contains non-16 kHz audio")
    if any(row["channels"] != 1 for row in audio_audits):
        raise ValueError("Corrected FLEURS contains non-mono audio")
    if any(row["duration_error"] > 0.01 for row in audio_audits):
        raise ValueError("Corrected FLEURS contains a duration mismatch above 10 ms")

    hashes_by_split = {
        split: {
            row["pcm_sha256"] for row in audio_audits if row["split"] == split
        }
        for split in SPLITS
    }
    within_split_duplicates = {
        split: len(corrected_rows_by_split[split]) - len(hashes_by_split[split])
        for split in SPLITS
    }
    cross_split_duplicates = {
        f"{left}_{right}": len(hashes_by_split[left] & hashes_by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    waxal_hashes = load_waxal_hashes(waxal_hash_path)
    fleurs_hashes = set().union(*hashes_by_split.values())
    waxal_overlap = len(fleurs_hashes & waxal_hashes)
    if any(within_split_duplicates.values()) or any(cross_split_duplicates.values()):
        raise ValueError("Duplicate FLEURS audio detected")
    if waxal_overlap:
        raise ValueError(f"Detected {waxal_overlap} FLEURS audio duplicates in WAXAL")

    summary = {
        "schema_version": 2,
        "protocol": "fleurs_shona_corrected_v2",
        "normalization": {
            "unicode": "NFC",
            "case": "casefold",
            "apostrophes": "map typographic forms to ASCII apostrophe",
            "dashes": "replace with spaces",
            "retained": ["letters", "combining marks", "Unicode numbers", "apostrophes"],
            "other_punctuation_symbols_controls": "replace with spaces",
            "whitespace": "collapse and strip",
        },
        "fleurs": {
            "dataset": v1_summary["fleurs"]["dataset"],
            "config": v1_summary["fleurs"]["config"],
            "revision": v1_summary["fleurs"]["revision"],
            "fingerprints": v1_summary["fleurs"]["fingerprints"],
        },
        "v1_summary": str(v1_summary_path),
        "v1_summary_sha256": sha256_file(v1_summary_path),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
        "splits": split_audits,
        "audio_audit": {
            "rows": len(audio_audits),
            "hash_file": str(audio_hash_path),
            "hash_file_sha256": sha256_file(audio_hash_path),
            "within_split_duplicate_rows": within_split_duplicates,
            "cross_split_duplicate_hashes": cross_split_duplicates,
            "waxal_hash_file": str(waxal_hash_path),
            "waxal_hash_file_sha256": sha256_file(waxal_hash_path),
            "waxal_duplicate_hashes": waxal_overlap,
            "sample_rate": 16000,
            "channels": 1,
            "maximum_duration_error_seconds": max(
                row["duration_error"] for row in audio_audits
            ),
        },
        "locked_use": {
            "training": False,
            "model_selection": False,
            "full_corpus_evaluation_before_rq1_selection": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()