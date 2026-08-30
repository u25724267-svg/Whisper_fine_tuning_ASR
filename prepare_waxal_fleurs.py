import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

import soundfile as sf
from datasets import DatasetDict, load_dataset


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WAXAL_DIR = Path("/home/casper/Speech/data/waxal/sna_asr")
DEFAULT_OUTPUT_DIR = Path("/ext_data/casper/whisper_data/waxal_fleurs/sna_asr")
DEFAULT_FLEURS_DATASET = "google/fleurs"
DEFAULT_FLEURS_CONFIG = "sn_zw"
DEFAULT_FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible combined WAXAL and FLEURS Shona ASR manifests."
    )
    parser.add_argument("--waxal-dir", type=Path, default=DEFAULT_WAXAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fleurs-dataset", default=DEFAULT_FLEURS_DATASET)
    parser.add_argument("--fleurs-config", default=DEFAULT_FLEURS_CONFIG)
    parser.add_argument(
        "--fleurs-revision",
        default=DEFAULT_FLEURS_REVISION,
        help="Immutable Hugging Face dataset commit SHA used to load FLEURS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_asr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    cleaned_characters = []
    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"P", "N", "S", "C"}:
            cleaned_characters.append(" ")
        else:
            cleaned_characters.append(character)
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
            row = json.loads(line)
            if not Path(row["audio_filepath"]).is_file():
                raise FileNotFoundError(
                    f"Missing WAXAL audio at {path}:{line_number}: {row['audio_filepath']}"
                )
            rows.append(row)
    return rows


def normalize_waxal_rows(rows: Iterable[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    normalized_rows = []
    for row in rows:
        text = normalize_asr_text(row["text"])
        if not text:
            raise ValueError(f"WAXAL row {row.get('id')} has empty normalized text")
        normalized_rows.append(
            {
                **row,
                "id": f"waxal_{row['id']}",
                "text": text,
                "source": "waxal",
                "source_split": split,
                "source_id": str(row["id"]),
            }
        )
    return normalized_rows


def normalize_fleurs_rows(
    dataset: DatasetDict, split: str, audio_dir: Path
) -> List[Dict[str, Any]]:
    split_dataset = dataset[split]
    gender_feature = split_dataset.features.get("gender")
    split_audio_dir = audio_dir / split
    split_audio_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = []

    for index, row in enumerate(split_dataset):
        source_id = str(row["id"])
        text_raw = str(row.get("raw_transcription") or row["transcription"])
        text = normalize_asr_text(text_raw)
        if not text:
            raise ValueError(f"FLEURS row {source_id} has empty normalized text")

        audio = row["audio"]
        audio_path = split_audio_dir / f"fleurs_{split}_{index:06d}_{source_id}.wav"
        sf.write(
            audio_path,
            audio["array"],
            audio["sampling_rate"],
            subtype="PCM_16",
        )
        gender = row.get("gender")
        if gender_feature is not None and hasattr(gender_feature, "int2str"):
            gender = gender_feature.int2str(gender)

        normalized_rows.append(
            {
                "audio_filepath": str(audio_path.resolve()),
                "duration": len(audio["array"]) / audio["sampling_rate"],
                "text": text,
                "id": f"fleurs_{split}_{source_id}",
                "speaker_id": None,
                "language": "sna",
                "gender": gender,
                "text_raw": text_raw,
                "source": "fleurs",
                "source_split": split,
                "source_id": source_id,
                "fleurs_language": row.get("language"),
            }
        )
    return normalized_rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    waxal_dir = args.waxal_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_dir = output_dir / "manifests"
    audio_dir = output_dir / "fleurs_audio"
    summary_path = output_dir / "preparation_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Prepared dataset already exists at {output_dir}; choose a new --output-dir"
        )

    waxal_manifest_paths = {
        split: waxal_dir / split / f"sna_asr_{split}.normalized.json" for split in SPLITS
    }
    missing_manifests = [str(path) for path in waxal_manifest_paths.values() if not path.is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing WAXAL manifests: {missing_manifests}")

    fleurs = load_dataset(
        args.fleurs_dataset,
        args.fleurs_config,
        revision=args.fleurs_revision,
    )
    if set(fleurs) != set(SPLITS):
        raise ValueError(f"Unexpected FLEURS splits: {sorted(fleurs)}")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, Dict[str, int]] = {}
    hours: Dict[str, Dict[str, float]] = {}
    output_manifests: Dict[str, str] = {}

    for split in SPLITS:
        waxal_rows = normalize_waxal_rows(read_jsonl(waxal_manifest_paths[split]), split)
        fleurs_rows = normalize_fleurs_rows(fleurs, split, audio_dir)
        combined_rows = [*waxal_rows, *fleurs_rows]
        if split == "train":
            random.Random(args.seed).shuffle(combined_rows)

        rows_by_source = {
            "waxal": waxal_rows,
            "fleurs": fleurs_rows,
            "combined": combined_rows,
        }
        counts[split] = {name: len(rows) for name, rows in rows_by_source.items()}
        hours[split] = {
            name: sum(float(row["duration"]) for row in rows) / 3600
            for name, rows in rows_by_source.items()
        }
        for source_name, rows in rows_by_source.items():
            manifest_path = manifest_dir / f"{source_name}_{split}.jsonl"
            write_jsonl(manifest_path, rows)
            output_manifests[f"{source_name}_{split}"] = str(manifest_path)

    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "normalization": "NFKC, lowercase, remove Unicode punctuation/numbers/symbols/control, collapse whitespace",
        "waxal": {
            "root": str(waxal_dir),
            "manifests": {split: str(path) for split, path in waxal_manifest_paths.items()},
            "sha256": {split: sha256_file(path) for split, path in waxal_manifest_paths.items()},
        },
        "fleurs": {
            "dataset": args.fleurs_dataset,
            "config": args.fleurs_config,
            "revision": args.fleurs_revision,
            "fingerprints": {split: fleurs[split]._fingerprint for split in SPLITS},
        },
        "counts": counts,
        "hours": hours,
        "output_manifests": output_manifests,
        "output_sha256": {
            name: sha256_file(Path(path)) for name, path in output_manifests.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()