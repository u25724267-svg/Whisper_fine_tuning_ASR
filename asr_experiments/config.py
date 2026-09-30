"""Project-relative configuration loading and path resolution."""

import json
import os
from pathlib import Path
from typing import Any, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Union[str, Path]) -> Path:
    """Expand environment/user markers and resolve relative paths from the project."""
    resolved = Path(os.path.expandvars(str(path))).expanduser()
    return resolved if resolved.is_absolute() else (PROJECT_ROOT / resolved).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a JSON experiment config and record its resolved source path."""
    resolved_path = resolve_path(config_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing experiment configuration: {resolved_path}")
    with resolved_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["config_path"] = str(resolved_path)
    return config