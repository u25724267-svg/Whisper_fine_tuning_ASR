import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prepare_waxal_speaker_disjoint import (
    DEFAULT_SOURCE_DIR,
    SOURCE_SPLITS,
    TARGET_RATIOS,
    read_source_rows,
    sha256_file,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path("/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create balanced deterministic speaker-disjoint WAXAL Shona manifests."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=50000)
    parser.add_argument("--heldout-speakers", type=int, default=24)
    parser.add_argument("--max-heldout-speaker-share", type=float, default=0.20)
    return parser.parse_args()


def group_statistics(
    speakers: List[str], speaker_rows: Dict[str, List[Dict[str, Any]]]
) -> Tuple[int, float, float, float]:
    row_counts = [len(speaker_rows[speaker]) for speaker in speakers]
    hour_counts = [
        sum(float(row["duration"]) for row in speaker_rows[speaker]) / 3600
        for speaker in speakers
    ]
    total_rows = sum(row_counts)
    total_hours = sum(hour_counts)
    return (
        total_rows,
        total_hours,
        max(row_counts) / total_rows,
        max(hour_counts) / total_hours,
    )


def assign_speakers_balanced(
    speaker_rows: Dict[str, List[Dict[str, Any]]],
    seed: int,
    search_trials: int,
    heldout_speakers: int,
    max_heldout_speaker_share: float,
) -> Tuple[Dict[str, List[str]], float]:
    speakers = sorted(speaker_rows)
    if 2 * heldout_speakers >= len(speakers):
        raise ValueError("Held-out speaker count leaves no viable training split")
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
            "validation": candidate[:heldout_speakers],
            "test": candidate[heldout_speakers : 2 * heldout_speakers],
            "train": candidate[2 * heldout_speakers :],
        }
        statistics = {
            split: group_statistics(split_speakers, speaker_rows)
            for split, split_speakers in assignments.items()
        }
        if any(
            statistics[split][2] > max_heldout_speaker_share
            or statistics[split][3] > max_heldout_speaker_share
            for split in ("validation", "test")
        ):
            continue

        score = 0.0
        for split, target_ratio in TARGET_RATIOS.items():
            split_rows, split_hours, _, _ = statistics[split]
            target_rows = total_rows * target_ratio
            target_hours = total_hours * target_ratio
            score += ((split_rows - target_rows) / target_rows) ** 2
            score += ((split_hours - target_hours) / target_hours) ** 2
        if score < best_score:
            best_score = score
            best_assignments = assignments

    if best_assignments is None:
        raise RuntimeError(
            "No assignment satisfied the held-out concentration constraint; "
            "increase --search-trials or revise the preregistered constraint"
        )
    return best_assignments, best_score


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
    if not 0 < args.max_heldout_speaker_share < 1:
        raise ValueError("--max-heldout-speaker-share must be between 0 and 1")

    rows, source_paths = read_source_rows(source_dir)
    speaker_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        speaker_rows[str(row["speaker_id"])].append(row)

    assignments, assignment_score = assign_speakers_balanced(
        speaker_rows,
        seed=args.seed,
        search_trials=args.search_trials,
        heldout_speakers=args.heldout_speakers,
        max_heldout_speaker_share=args.max_heldout_speaker_share,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    selected_rows_by_split = {}
    statistics = {}
    for split in TARGET_RATIOS:
        split_speaker_set = set(assignments[split])
        selected_rows = sorted(
            (row for row in rows if str(row["speaker_id"]) in split_speaker_set),
            key=lambda row: str(row["id"]),
        )
        manifest_path = output_dir / f"{split}.jsonl"
        write_jsonl(manifest_path, selected_rows)
        manifest_paths[split] = manifest_path
        selected_rows_by_split[split] = selected_rows
        statistics[split] = group_statistics(assignments[split], speaker_rows)

    speaker_sets = {split: set(speakers) for split, speakers in assignments.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = speaker_sets[left] & speaker_sets[right]
        if overlap:
            raise RuntimeError(f"Speaker overlap between {left} and {right}: {sorted(overlap)}")

    summary = {
        "schema_version": 2,
        "protocol": "waxal_shona_speaker_disjoint_v2",
        "seed": args.seed,
        "search_trials": args.search_trials,
        "assignment_score": assignment_score,
        "assignment_uses": ["speaker_id", "row_count", "duration"],
        "constraints": {
            "heldout_speakers_per_split": args.heldout_speakers,
            "max_heldout_speaker_row_share": args.max_heldout_speaker_share,
            "max_heldout_speaker_hour_share": args.max_heldout_speaker_share,
        },
        "source_dir": str(source_dir),
        "source_manifests": {split: str(path) for split, path in source_paths.items()},
        "source_sha256": {split: sha256_file(path) for split, path in source_paths.items()},
        "splits": {
            split: {
                "manifest": str(manifest_paths[split]),
                "sha256": sha256_file(manifest_paths[split]),
                "rows": statistics[split][0],
                "hours": statistics[split][1],
                "speakers": len(speaker_sets[split]),
                "largest_speaker_row_share": statistics[split][2],
                "largest_speaker_hour_share": statistics[split][3],
                "original_split_rows": {
                    source_split: sum(
                        row["original_split"] == source_split
                        for row in selected_rows_by_split[split]
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