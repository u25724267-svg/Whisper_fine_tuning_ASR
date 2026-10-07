"""Prepare the frozen Medium extension without launching training."""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from asr_experiments.config import PROJECT_ROOT
from asr_experiments.provenance import sha256_file


SEEDS = (42, 43, 44)
MODEL_ID = "openai/whisper-medium"
MODEL_REVISION = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
PROJECT = "whisper-shona-multilingual"
FAMILIES = (
    ("rq1", "c0_random_seed{seed}_v2"),
    ("rq1", "c1_sortagrad_seed{seed}"),
    ("rq1", "c3_snr_duration_seed{seed}"),
    ("rq2", "a2_waveform_random_seed{seed}"),
    ("rq2", "a3_waveform_c3_seed{seed}"),
    ("rq2", "s2_specaug_lb_random_seed{seed}"),
    ("rq2", "s3_specaug_lb_sortagrad_seed{seed}"),
    ("rq2", "s4_specaug_ld_random_seed{seed}"),
    ("rq2", "s5_specaug_ld_sortagrad_seed{seed}"),
)
QUEUE_PATH = PROJECT_ROOT / "experiments/medium/queue.json"


def specifications(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Order clean controls before treatments, matching seeds within each cell."""
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for question, template in FAMILIES
        for seed in SEEDS
        for path in [root / "experiments" / question / template.format(seed=seed) / "config.json"]
    ]


def derive_config(
    source: dict[str, Any], source_path: Path, root: Path, identities: dict[str, str]
) -> dict[str, Any]:
    """Preserve Base scientific settings; change only the pinned model."""
    config = copy.deepcopy(source)
    base_id = source["experiment_id"]
    experiment_id = identities[base_id]
    control = source.get("control_experiment")
    if control is not None and control not in identities:
        raise ValueError(f"Unmapped Base control for {base_id}: {control}")
    config.update(
        experiment_id=experiment_id,
        experiment_name=experiment_id,
        control_experiment=identities[control] if control else None,
        runner=source.get("runner", "train_full.py"),
        output_dir=f"medium/{source_path.parent.parent.name}/{source_path.parent.name}",
    )
    config["model"].update(id=MODEL_ID, revision=MODEL_REVISION)
    config["resources"].update(min_free_gpu_mb=20000, min_free_disk_gb=32)
    config["wandb"].update(
        project=PROJECT,
        run_name=experiment_id,
        job_type="medium-extension-training",
        resume_run_id=None,
        tags=["whisper-medium" if tag == "whisper-base" else tag for tag in source["wandb"]["tags"]]
        + ["medium-extension"],
    )
    config["medium_extension"] = {
        "source_config": source_path.relative_to(root).as_posix(),
        "source_config_sha256": sha256_file(source_path),
        "source_experiment_id": base_id,
        "base_control_experiment": control,
        "batch_adaptation": "none; exact Base training and evaluation batch settings",
    }
    return config


def expected_files(root: Path) -> dict[Path, str]:
    sources = specifications(root)
    identities = {config["experiment_id"]: f"medium-{config['experiment_id']}" for _, config in sources}
    files: dict[Path, str] = {}
    entries = []
    for source_path, source in sources:
        config = derive_config(source, source_path, root, identities)
        relative_dir = Path("experiments") / config["output_dir"]
        config_path = root / relative_dir / "config.json"
        files[config_path] = json.dumps(config, indent=2) + "\n"
        files[config_path.with_name("run.sh")] = (
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'ROOT_DIR="$(cd "$EXPERIMENT_DIR/../../../.." && pwd)"\n'
            'exec "$ROOT_DIR/scripts/run_experiment.sh" "$EXPERIMENT_DIR"\n'
        )
        files[config_path.with_name("experiment.md")] = (
            f"# {config['experiment_id']}\n\n"
            f"Medium extension of `{source_path.relative_to(root).as_posix()}`.\n"
            "Seeds, data, curriculum, augmentation, optimizer settings, and three-epoch\n"
            "schedule are inherited, including exact training/evaluation batch settings.\n"
            "Only the pinned model differs scientifically. No time warping is used.\n\n"
            "Protocol: `documents/data/whisper_medium_rq1_rq2_protocol.md`.\n"
        )
        entries.append({
            "experiment_dir": relative_dir.as_posix(),
            "experiment_id": config["experiment_id"],
            "control_experiment": config["control_experiment"],
            "runner": config["runner"],
            "output_dir": config["output_dir"],
            "source_config": config["medium_extension"]["source_config"],
        })
    files[root / "experiments/medium/queue.json"] = json.dumps({
        "schema_version": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "seeds": list(SEEDS),
        "runs": entries,
        "estimated_disk_reserve_gib": 32 * len(entries),
        "training_launched": False,
    }, indent=2) + "\n"
    return files


def materialize(root: Path, check_only: bool = False) -> None:
    files = expected_files(root)
    for path, content in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace differing queue definition: {path}")
        if check_only and not path.is_file():
            raise FileNotFoundError(f"Missing prepared queue file: {path}")
    if check_only:
        return
    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if path.name == "run.sh":
            path.chmod(0o775)


def validate_inputs(root: Path, output_root: Path) -> list[dict[str, Any]]:
    materialize(root, check_only=True)
    queue = json.loads((root / "experiments/medium/queue.json").read_text(encoding="utf-8"))
    for entry in queue["runs"]:
        config_path = root / entry["experiment_dir"] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        required = [Path(config["data"][f"{split}_manifest"]) for split in ("train", "validation", "test")]
        required.append(Path(config["data"]["protocol_summary"]))
        if "metadata_path" in config["curriculum"]:
            required.append(Path(config["curriculum"]["metadata_path"]))
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"{entry['experiment_id']}: missing {path}")
        waveform = config.get("waveform_augmentation", {})
        if waveform.get("enabled"):
            summary = json.loads(Path(config["data"]["protocol_summary"]).read_text(encoding="utf-8"))
            train_path = Path(config["data"]["train_manifest"])
            if sha256_file(train_path) != waveform["train_manifest_sha256"]:
                raise ValueError(f"Waveform manifest changed: {train_path}")
            if summary["seed"] != config["training"]["seed"]:
                raise ValueError(f"Waveform seed mismatch: {config_path}")
        for destination in (output_root / config["output_dir"], root / "artifacts/experiment_outputs" / config["output_dir"]):
            if destination.exists():
                raise FileExistsError(f"Medium output already exists; inspect before launch: {destination}")
    return queue["runs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate prepared queue without writing files.")
    parser.add_argument("--list", action="store_true", help="Validate and print ordered experiment directories.")
    parser.add_argument("--dry-run-runners", action="store_true", help="Validate all 27 actual runner dry-runs, never train.")
    args = parser.parse_args()
    materialize(PROJECT_ROOT, check_only=args.check or args.list or args.dry_run_runners)
    entries = validate_inputs(PROJECT_ROOT, Path(os.environ.get("ASR_OUTPUT_ROOT", "/ext_data/casper/asr_experiment_outputs")))
    if args.list:
        print("\n".join(entry["experiment_dir"] for entry in entries))
        return
    if args.dry_run_runners:
        for entry in entries:
            print(f"Dry-run: {entry['experiment_id']}", flush=True)
            subprocess.run([
                sys.executable, str(PROJECT_ROOT / "asr.py"), Path(entry["runner"]).stem,
                "--config", str(PROJECT_ROOT / entry["experiment_dir"] / "config.json"),
                "--dry-run",
            ], cwd=PROJECT_ROOT, check=True)
    print(f"Prepared/validated {len(entries)} Medium runs. Training has NOT been launched.")


if __name__ == "__main__":
    main()