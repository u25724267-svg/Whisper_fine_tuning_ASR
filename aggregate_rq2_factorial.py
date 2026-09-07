import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict


SEEDS = (42, 43, 44)
SPLITS = ("validation", "test")
METRICS = ("wer", "cer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the completed RQ2 A0-A3 factorial without making a decision."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/ext_data/casper/asr_experiment_outputs"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-subdir", default="item_predictions")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--report-name", default="rq2_factorial_summary")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_paths(
    output_root: Path, prediction_subdir: str
) -> Dict[str, Dict[int, Path]]:
    return {
        "a0_clean_random": {
            seed: output_root
            / "rq1"
            / f"c0_random_seed{seed}_v2"
            / prediction_subdir
            / "summary.json"
            for seed in SEEDS
        },
        "a1_clean_c3": {
            seed: output_root
            / "rq1"
            / "c3_snr_duration_seed42"
            / prediction_subdir
            / "summary.json"
            for seed in SEEDS
        },
        "a2_waveform_random": {
            seed: output_root
            / "rq2"
            / f"a2_waveform_random_seed{seed}"
            / prediction_subdir
            / "summary.json"
            for seed in SEEDS
        },
        "a3_waveform_c3": {
            seed: output_root
            / "rq2"
            / f"a3_waveform_c3_seed{seed}"
            / prediction_subdir
            / "summary.json"
            for seed in SEEDS
        },
    }


def summarize(values: list[float]) -> Dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = summary_paths(output_root, args.prediction_subdir)
    missing = [str(path) for seeds in paths.values() for path in seeds.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "RQ2 factorial is incomplete; missing summaries:\n" + "\n".join(missing)
        )

    cells: Dict[str, Dict[int, Dict[str, Any]]] = {}
    source_hashes: Dict[str, Dict[str, str]] = {}
    for cell, seed_paths in paths.items():
        cells[cell] = {}
        source_hashes[cell] = {}
        for seed, path in seed_paths.items():
            summary = json.loads(path.read_text(encoding="utf-8"))
            cells[cell][seed] = summary["splits"]
            source_hashes[cell][str(seed)] = sha256_file(path)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "effect_sign": "positive values mean lower error for the named factor or interaction",
        "source_summary_sha256": source_hashes,
        "splits": {},
    }
    if args.prediction_subdir != "item_predictions" or args.splits != list(SPLITS):
        result["prediction_subdir"] = args.prediction_subdir
        result["evaluated_splits"] = args.splits
    for split in args.splits:
        split_result: Dict[str, Any] = {"cells": {}, "effects": {}}
        for cell in cells:
            split_result["cells"][cell] = {
                metric: summarize(
                    [float(cells[cell][seed][split][metric]) for seed in SEEDS]
                )
                for metric in METRICS
            }
        for metric in METRICS:
            values = {
                cell: {
                    seed: float(cells[cell][seed][split][metric])
                    for seed in SEEDS
                }
                for cell in cells
            }
            effects = {
                "augmentation_without_curriculum": [
                    values["a0_clean_random"][seed]
                    - values["a2_waveform_random"][seed]
                    for seed in SEEDS
                ],
                "curriculum_without_augmentation": [
                    values["a0_clean_random"][seed]
                    - values["a1_clean_c3"][seed]
                    for seed in SEEDS
                ],
                "curriculum_with_augmentation": [
                    values["a2_waveform_random"][seed]
                    - values["a3_waveform_c3"][seed]
                    for seed in SEEDS
                ],
                "augmentation_with_curriculum": [
                    values["a1_clean_c3"][seed]
                    - values["a3_waveform_c3"][seed]
                    for seed in SEEDS
                ],
            }
            effects["interaction"] = [
                augmented - clean
                for augmented, clean in zip(
                    effects["curriculum_with_augmentation"],
                    effects["curriculum_without_augmentation"],
                )
            ]
            for name, effect_values in effects.items():
                split_result["effects"].setdefault(name, {})[metric] = summarize(
                    effect_values
                )
        result["splits"][split] = split_result

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.report_name}.json"
    markdown_path = output_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RQ2 Factorial Summary",
        "",
        "Positive effects mean lower error. This report is descriptive and does not make a promotion decision.",
        "",
    ]
    for split in args.splits:
        lines.extend([f"## {split.title()}", "", "### Cell means", "", "| Cell | WER mean +/- SD | CER mean +/- SD |", "|---|---:|---:|"])
        for cell, metrics in result["splits"][split]["cells"].items():
            lines.append(
                f"| {cell} | {metrics['wer']['mean']:.4f} +/- {metrics['wer']['sample_sd']:.4f} | "
                f"{metrics['cer']['mean']:.4f} +/- {metrics['cer']['sample_sd']:.4f} |"
            )
        lines.extend(["", "### Effects", "", "| Effect | WER mean +/- SD | CER mean +/- SD |", "|---|---:|---:|"])
        for name, metrics in result["splits"][split]["effects"].items():
            lines.append(
                f"| {name} | {metrics['wer']['mean']:+.4f} +/- {metrics['wer']['sample_sd']:.4f} | "
                f"{metrics['cer']['mean']:+.4f} +/- {metrics['cer']['sample_sd']:.4f} |"
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()