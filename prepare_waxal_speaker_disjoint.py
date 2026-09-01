import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SOURCE_DIR = Path("/home/casper/Speech/data/waxal/sna_asr")
DEFAULT_OUTPUT_DIR = Path("/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v1")
SOURCE_SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic speaker-disjoint WAXAL Shona manifests."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=20000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_rows(source_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Path]]:
    rows = []
    source_paths = {}
    for split in SOURCE_SPLITS:
        path = source_dir / split / f"sna_asr_{split}.normalized.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source manifest: {path}")
        source_paths[split] = path
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                speaker_id = str(row.get("speaker_id") or "").strip()
                if not speaker_id:
                    raise ValueError(f"Missing speaker_id at {path}:{line_number}")
                audio_path = Path(row["audio_filepath"])
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Missing audio at {path}:{line_number}: {audio_path}")
                if not str(row.get("text") or "").strip():
                    raise ValueError(f"Empty text at {path}:{line_number}")
                rows.append({**row, "original_split": split})

    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate utterance IDs found while pooling source splits")
    return rows, source_paths


def split_objective(
    assignments: Dict[str, List[str]],
    speaker_rows: Dict[str, List[Dict[str, Any]]],
    total_rows: int,
    total_hours: float,
) -> float:
    score = 0.0
    for split, target_ratio in TARGET_RATIOS.items():
        split_rows = sum(len(speaker_rows[speaker]) for speaker in assignments[split])
        split_hours = sum(
            float(row["duration"]) / 3600
            for speaker in assignments[split]
            for row in speaker_rows[speaker]
        )
        target_rows = total_rows * target_ratio
        target_hours = total_hours * target_ratio
        score += ((split_rows - target_rows) / target_rows) ** 2
        score += ((split_hours - target_hours) / target_hours) ** 2
    return score


def assign_speakers(
    speaker_rows: Dict[str, List[Dict[str, Any]]], seed: int, search_trials: int
) -> Dict[str, List[str]]:
    speakers = sorted(speaker_rows)
    validation_speakers = round(len(speakers) * TARGET_RATIOS["validation"])
    test_speakers = round(len(speakers) * TARGET_RATIOS["test"])
    total_rows = sum(len(rows) for rows in speaker_rows.values())
    total_hours = sum(
        float(row["duration"]) / 3600 for rows in speaker_rows.values() for row in rows
    )
    random_generator = random.Random(seed)
    best_score = float("inf")
    best_assignments = None

    for _ in range(search_trials):
        candidate = speakers.copy()
        random_generator.shuffle(candidate)
        assignments = {
            "validation": candidate[:validation_speakers],
            "test": candidate[validation_speakers : validation_speakers + test_speakers],
            "train": candidate[validation_speakers + test_speakers :],
        }
        score = split_objective(assignments, speaker_rows, total_rows, total_hours)
        if score < best_score:
            best_score = score
            best_assignments = assignments

    if best_assignments is None:
        raise RuntimeError("No speaker assignment was generated")
    return best_assignments


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = output_dir / "split_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Speaker-disjoint protocol already exists at {output_dir}; "
            "choose a new --output-dir"
        )
    if args.search_trials < 1:
        raise ValueError("--search-trials must be at least 1")

    rows, source_paths = read_source_rows(source_dir)
    speaker_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        speaker_rows[str(row["speaker_id"])].append(row)

    assignments = assign_speakers(speaker_rows, args.seed, args.search_trials)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    split_rows = {}
    for split in TARGET_RATIOS:
        split_speaker_set = set(assignments[split])
        selected_rows = sorted(
            (row for row in rows if str(row["speaker_id"]) in split_speaker_set),
            key=lambda row: str(row["id"]),
        )
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, selected_rows)
        manifest_paths[split] = path
        split_rows[split] = selected_rows

    speaker_sets = {split: set(speakers) for split, speakers in assignments.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = speaker_sets[left] & speaker_sets[right]
        if overlap:
            raise RuntimeError(f"Speaker overlap between {left} and {right}: {sorted(overlap)}")

    summary = {
        "schema_version": 1,
        "protocol": "waxal_shona_speaker_disjoint_v1",
        "seed": args.seed,
        "search_trials": args.search_trials,
        "assignment_uses": ["speaker_id", "row_count", "duration"],
        "source_dir": str(source_dir),
        "source_manifests": {split: str(path) for split, path in source_paths.items()},
        "source_sha256": {split: sha256_file(path) for split, path in source_paths.items()},
        "splits": {
            split: {
                "manifest": str(manifest_paths[split]),
                "sha256": sha256_file(manifest_paths[split]),
                "rows": len(split_rows[split]),
                "hours": sum(float(row["duration"]) for row in split_rows[split]) / 3600,
                "speakers": len(speaker_sets[split]),
                "original_split_rows": {
                    source_split: sum(
                        row["original_split"] == source_split for row in split_rows[split]
                    )
                    for source_split in SOURCE_SPLITS
                },
            }
            for split in TARGET_RATIOS
        },
        "speaker_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
    }
    (output_dir / "speaker_assignments.json").write_text(
        json.dumps({split: sorted(speakers) for split, speakers in assignments.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()