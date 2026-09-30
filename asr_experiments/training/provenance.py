"""Curriculum-specific additions to a completed training run manifest."""

import json
from pathlib import Path
from typing import Any, Mapping

from asr_experiments.provenance import sha256_file


def add_curriculum_provenance(
    output_dir: Path,
    curriculum: Mapping[str, Any],
    runner_path: Path,
    component_paths: Mapping[str, Path],
) -> None:
    """Attach curriculum configuration and implementation hashes to a run manifest."""
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["curriculum"] = dict(curriculum)
    manifest["sha256"]["curriculum_runner"] = sha256_file(runner_path)
    for component_name, component_path in component_paths.items():
        manifest["sha256"][component_name] = sha256_file(component_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")