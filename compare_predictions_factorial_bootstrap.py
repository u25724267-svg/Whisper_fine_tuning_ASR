import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from compare_predictions_bootstrap import (
    load_cluster_map,
    load_rows,
    percentile,
    sha256_file,
)


CELL_NAMES = ("cell_00", "cell_noise", "cell_rir", "cell_noise_rir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Joint speaker-cluster bootstrap for a 2x2 noise-by-RIR ASR factorial."
        )
    )
    parser.add_argument("--cell-00", type=Path, required=True)
    parser.add_argument("--cell-noise", type=Path, required=True)
    parser.add_argument("--cell-rir", type=Path, required=True)
    parser.add_argument("--cell-noise-rir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-manifest", type=Path)
    parser.add_argument("--cluster-field", default="source_id")
    return parser.parse_args()


def error_count(row: dict[str, Any]) -> int:
    return int(row["substitutions"]) + int(row["deletions"]) + int(
        row["insertions"]
    )


def reference_words(row: dict[str, Any]) -> int:
    return int(row["reference_words"])


def factorial_effects(wers: dict[str, float]) -> dict[str, float]:
    noise_improvement = 0.5 * (
        (wers["cell_00"] - wers["cell_noise"])
        + (wers["cell_rir"] - wers["cell_noise_rir"])
    )
    rir_improvement = 0.5 * (
        (wers["cell_00"] - wers["cell_rir"])
        + (wers["cell_noise"] - wers["cell_noise_rir"])
    )
    interaction_improvement = (
        wers["cell_noise"]
        + wers["cell_rir"]
        - wers["cell_noise_rir"]
        - wers["cell_00"]
    )
    return {
        "noise_main_improvement": noise_improvement,
        "rir_main_improvement": rir_improvement,
        "noise_rir_interaction_improvement": interaction_improvement,
    }


def centered_bootstrap_p_value(samples: Sequence[float], observed: float) -> float:
    exceedances = sum(
        abs(sample - observed) >= abs(observed) for sample in samples
    )
    return (exceedances + 1) / (len(samples) + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_values[name])
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted


def main() -> None:
    args = parse_args()
    if args.replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")

    paths = {
        "cell_00": args.cell_00.expanduser().resolve(),
        "cell_noise": args.cell_noise.expanduser().resolve(),
        "cell_rir": args.cell_rir.expanduser().resolve(),
        "cell_noise_rir": args.cell_noise_rir.expanduser().resolve(),
    }
    require_speaker = args.cluster_manifest is None
    cells = {
        name: load_rows(path, require_speaker=require_speaker)
        for name, path in paths.items()
    }
    reference_ids = set(cells["cell_00"])
    for name in CELL_NAMES[1:]:
        if set(cells[name]) != reference_ids:
            raise ValueError(f"Prediction IDs do not match for {name}")

    cluster_manifest_path = None
    if args.cluster_manifest:
        cluster_manifest_path = args.cluster_manifest.expanduser().resolve()
        cluster_by_id = load_cluster_map(cluster_manifest_path, args.cluster_field)
        missing = reference_ids - set(cluster_by_id)
        if missing:
            raise ValueError(
                f"Cluster manifest is missing {len(missing)} prediction IDs"
            )
    else:
        cluster_by_id = {
            utterance_id: str(row["speaker_id"])
            for utterance_id, row in cells["cell_00"].items()
        }

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for utterance_id in sorted(reference_ids):
        baseline_row = cells["cell_00"][utterance_id]
        for name in CELL_NAMES[1:]:
            candidate_row = cells[name][utterance_id]
            if candidate_row["reference"] != baseline_row["reference"]:
                raise ValueError(f"Reference mismatch for {utterance_id} in {name}")
            if not args.cluster_manifest and (
                candidate_row["speaker_id"] != baseline_row["speaker_id"]
            ):
                raise ValueError(f"Speaker mismatch for {utterance_id} in {name}")
            if reference_words(candidate_row) != reference_words(baseline_row):
                raise ValueError(
                    f"Reference word-count mismatch for {utterance_id} in {name}"
                )
        by_cluster[cluster_by_id[utterance_id]].append(utterance_id)

    cluster_counts: dict[str, dict[str, tuple[int, int]]] = {}
    for cluster, utterance_ids in by_cluster.items():
        cluster_counts[cluster] = {}
        for name in CELL_NAMES:
            rows = [cells[name][utterance_id] for utterance_id in utterance_ids]
            cluster_counts[cluster][name] = (
                sum(error_count(row) for row in rows),
                sum(reference_words(row) for row in rows),
            )

    observed_wers = {}
    for name in CELL_NAMES:
        errors = sum(error_count(row) for row in cells[name].values())
        words = sum(reference_words(row) for row in cells[name].values())
        if words <= 0:
            raise ValueError(f"Invalid reference-word total for {name}")
        observed_wers[name] = 100 * errors / words
    observed_effects = factorial_effects(observed_wers)

    clusters = sorted(by_cluster)
    generator = random.Random(args.seed)
    bootstrap_effects = {name: [] for name in observed_effects}
    for _ in range(args.replicates):
        sampled_clusters = generator.choices(clusters, k=len(clusters))
        sampled_wers = {}
        for name in CELL_NAMES:
            errors = sum(
                cluster_counts[cluster][name][0] for cluster in sampled_clusters
            )
            words = sum(
                cluster_counts[cluster][name][1] for cluster in sampled_clusters
            )
            sampled_wers[name] = 100 * errors / words
        for name, value in factorial_effects(sampled_wers).items():
            bootstrap_effects[name].append(value)

    p_values = {
        name: centered_bootstrap_p_value(values, observed_effects[name])
        for name, values in bootstrap_effects.items()
    }
    adjusted_p_values = holm_adjust(p_values)
    effects = {}
    for name, values in bootstrap_effects.items():
        values.sort()
        effects[name] = {
            "observed": observed_effects[name],
            "confidence_interval_95": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
            "centered_bootstrap_p_value_two_sided": p_values[name],
            "holm_adjusted_p_value": adjusted_p_values[name],
        }

    result = {
        "schema_version": 1,
        "effect_direction": "positive values mean lower WER",
        "formulas": {
            "noise_main_improvement": "0.5*((WER00-WERN)+(WERR-WERNR))",
            "rir_main_improvement": "0.5*((WER00-WERR)+(WERN-WERNR))",
            "noise_rir_interaction_improvement": "WERN+WERR-WERNR-WER00",
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "rows": len(reference_ids),
        "clusters": len(clusters),
        "cluster_field": args.cluster_field if args.cluster_manifest else "speaker_id",
        "replicates": args.replicates,
        "seed": args.seed,
        "observed_wer": observed_wers,
        "effects": effects,
        "multiplicity": {
            "method": "Holm",
            "family": list(observed_effects),
            "note": "P-values use the centered paired cluster-bootstrap distribution.",
        },
    }
    if cluster_manifest_path:
        result["cluster_manifest"] = str(cluster_manifest_path)
        result["cluster_manifest_sha256"] = sha256_file(cluster_manifest_path)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()