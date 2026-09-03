import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired speaker-cluster bootstrap comparison of ASR predictions."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--practical-threshold", type=float, default=0.5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.expanduser().resolve().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            utterance_id = str(row["id"])
            if utterance_id in rows:
                raise ValueError(f"Duplicate ID {utterance_id} in {path}:{line_number}")
            if not row.get("speaker_id"):
                raise ValueError(f"Missing speaker ID in {path}:{line_number}")
            rows[utterance_id] = row
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    return rows


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def corpus_counts(rows: Sequence[Dict[str, Any]]) -> tuple[int, int]:
    errors = sum(
        int(row["substitutions"])
        + int(row["deletions"])
        + int(row["insertions"])
        for row in rows
    )
    reference_words = sum(int(row["reference_words"]) for row in rows)
    return errors, reference_words


def main() -> None:
    args = parse_args()
    if args.replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    if args.practical_threshold < 0:
        raise ValueError("Practical threshold must be non-negative")

    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    baseline = load_rows(baseline_path)
    candidate = load_rows(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs do not match")

    by_speaker: Dict[str, list[str]] = defaultdict(list)
    for utterance_id, baseline_row in baseline.items():
        candidate_row = candidate[utterance_id]
        if baseline_row["reference"] != candidate_row["reference"]:
            raise ValueError(f"Reference mismatch for {utterance_id}")
        if baseline_row["speaker_id"] != candidate_row["speaker_id"]:
            raise ValueError(f"Speaker mismatch for {utterance_id}")
        by_speaker[str(baseline_row["speaker_id"])].append(utterance_id)

    speaker_counts: Dict[str, tuple[int, int, int]] = {}
    for speaker, utterance_ids in by_speaker.items():
        baseline_errors, reference_words = corpus_counts(
            [baseline[utterance_id] for utterance_id in utterance_ids]
        )
        candidate_errors, candidate_words = corpus_counts(
            [candidate[utterance_id] for utterance_id in utterance_ids]
        )
        if reference_words != candidate_words:
            raise ValueError(f"Reference word count mismatch for speaker {speaker}")
        speaker_counts[speaker] = (
            baseline_errors,
            candidate_errors,
            reference_words,
        )

    baseline_errors, reference_words = corpus_counts(list(baseline.values()))
    candidate_errors, candidate_words = corpus_counts(list(candidate.values()))
    if reference_words != candidate_words or reference_words <= 0:
        raise ValueError("Invalid paired reference word totals")
    baseline_wer = 100 * baseline_errors / reference_words
    candidate_wer = 100 * candidate_errors / reference_words
    observed_improvement = baseline_wer - candidate_wer

    speakers = sorted(by_speaker)
    generator = random.Random(args.seed)
    improvements = []
    for _ in range(args.replicates):
        sampled = generator.choices(speakers, k=len(speakers))
        sampled_baseline_errors = sum(speaker_counts[speaker][0] for speaker in sampled)
        sampled_candidate_errors = sum(speaker_counts[speaker][1] for speaker in sampled)
        sampled_words = sum(speaker_counts[speaker][2] for speaker in sampled)
        improvements.append(
            100 * (sampled_baseline_errors - sampled_candidate_errors) / sampled_words
        )
    improvements.sort()

    result = {
        "schema_version": 1,
        "baseline": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "rows": len(baseline),
        "speakers": len(speakers),
        "reference_words": reference_words,
        "baseline_wer": baseline_wer,
        "candidate_wer": candidate_wer,
        "observed_improvement": observed_improvement,
        "replicates": args.replicates,
        "seed": args.seed,
        "confidence_interval_95": [
            percentile(improvements, 0.025),
            percentile(improvements, 0.975),
        ],
        "probability_improvement": sum(value > 0 for value in improvements)
        / args.replicates,
        "practical_threshold": args.practical_threshold,
        "probability_practical_improvement": sum(
            value >= args.practical_threshold for value in improvements
        )
        / args.replicates,
    }
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()