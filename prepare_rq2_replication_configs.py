import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/ext_data/casper/asr_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hash-pinned A2/A3 configs for one materialized RQ2 seed."
    )
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exact(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(f"Refusing to overwrite differing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_if_absent(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def configure_common(
    config: Dict[str, Any], seed: int, summary: Dict[str, Any]
) -> None:
    data_dir = DATA_ROOT / f"rq2_waveform_mild_seed{seed}"
    config["training"]["seed"] = seed
    config["data"]["protocol"] = (
        f"waxal_shona_speaker_disjoint_v2_rq2_waveform_mild_seed{seed}"
    )
    config["data"]["protocol_summary"] = str(data_dir / "summary.json")
    config["data"]["train_manifest"] = str(data_dir / "train.jsonl")
    waveform = config["waveform_augmentation"]
    waveform["materialization_seed"] = seed
    waveform["policy_sha256"] = summary["policy_sha256"]
    waveform["asset_archive_sha256"] = summary["asset_archive_sha256"]
    waveform["train_manifest_sha256"] = summary["train_manifest_sha256"]
    waveform["parameters_sha256"] = summary["parameters_sha256"]


def render_run_script(experiment_id: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
ROOT_DIR=\"$(cd \"$EXPERIMENT_DIR/../../..\" && pwd)\"
exec \"$ROOT_DIR/scripts/run_experiment.sh\" \"$EXPERIMENT_DIR/config.json\" \"{experiment_id}\"
"""


def render_record(condition: str, seed: int, summary: Dict[str, Any]) -> str:
    if condition == "a2":
        title = f"# RQ2-A2: Waveform Augmentation, Random Training, Seed {seed}"
        purpose = "Measure the waveform-augmentation main effect against the matched C0 control."
        order = "Standard seeded random order"
        control = f"C0 seed {seed}"
    else:
        title = f"# RQ2-A3: Waveform Augmentation with C3, Seed {seed}"
        purpose = "Measure the C3-by-waveform interaction against the matched A2 condition."
        order = "Frozen clean-data C3 order `86ff768edcc8`"
        control = f"A2 seed {seed}"
    return f"""{title}

## Purpose

{purpose}

## Contract

- Control: {control}
- Materialization seed: {seed}
- {order}
- Frozen waveform policy `rq2-waveform-mild-v1`
- SpecAugment disabled
- Clean validation and test audio
- Training manifest SHA-256: `{summary['train_manifest_sha256']}`
- Parameter audit SHA-256: `{summary['parameters_sha256']}`
- Hashed item-level validation/test predictions required

## Status

Generated from the completed seed-{seed} materialization summary. Training not yet launched.
"""


def generate(seed: int) -> None:
    if seed not in {43, 44}:
        raise ValueError("Replication config generation is restricted to seeds 43 and 44")
    data_dir = DATA_ROOT / f"rq2_waveform_mild_seed{seed}"
    summary_path = data_dir / "summary.json"
    manifest_path = data_dir / "train.jsonl"
    parameters_path = data_dir / "augmentation_parameters.jsonl"
    if not all(path.is_file() for path in (summary_path, manifest_path, parameters_path)):
        raise FileNotFoundError(f"Incomplete RQ2 materialization for seed {seed}")
    summary = load_json(summary_path)
    if summary["seed"] != seed or summary["rows"] != 13_807:
        raise ValueError(f"Invalid materialization summary for seed {seed}")

    specifications = {
        "a2": {
            "template": ROOT_DIR / "experiments/rq2/a2_waveform_random_seed42/config.json",
            "directory": ROOT_DIR / f"experiments/rq2/a2_waveform_random_seed{seed}",
            "experiment_id": f"rq2-a2-waveform-random-seed{seed}",
            "output_dir": f"rq2/a2_waveform_random_seed{seed}",
            "control": f"rq1-c0-random-seed{seed}-v2",
            "job_type": "waveform-augmentation-replication",
            "tags": ["rq2", "a2", "waveform-augmentation", "random"],
        },
        "a3": {
            "template": ROOT_DIR / "experiments/rq2/a3_waveform_c3_seed42/config.json",
            "directory": ROOT_DIR / f"experiments/rq2/a3_waveform_c3_seed{seed}",
            "experiment_id": f"rq2-a3-waveform-c3-seed{seed}",
            "output_dir": f"rq2/a3_waveform_c3_seed{seed}",
            "control": f"rq2-a2-waveform-random-seed{seed}",
            "job_type": "waveform-curriculum-replication",
            "tags": ["rq2", "a3", "waveform-augmentation", "c3"],
        },
    }
    for condition, specification in specifications.items():
        config = copy.deepcopy(load_json(specification["template"]))
        configure_common(config, seed, summary)
        experiment_id = specification["experiment_id"]
        config["experiment_id"] = experiment_id
        config["experiment_name"] = experiment_id
        config["control_experiment"] = specification["control"]
        config["output_dir"] = specification["output_dir"]
        config["wandb"]["run_name"] = experiment_id
        config["wandb"]["job_type"] = specification["job_type"]
        config["wandb"]["tags"] = specification["tags"] + [
            "whisper-base",
            "shona",
            "speaker-disjoint-v2",
            f"seed-{seed}",
            "replication",
        ]
        directory = specification["directory"]
        write_exact(
            directory / "config.json",
            json.dumps(config, indent=2) + "\n",
        )
        write_exact(directory / "run.sh", render_run_script(experiment_id))
        (directory / "run.sh").chmod(0o775)
        write_if_absent(
            directory / "experiment.md",
            render_record(condition, seed, summary),
        )
        print(f"Prepared {directory.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    generate(parse_args().seed)