"""Hashing helpers used to record experiment and artifact provenance."""

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using stable key and separator formatting."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_indices_sha256(indices: list[int]) -> str:
    """Hash an ordered integer sequence using the established CSV encoding."""
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()