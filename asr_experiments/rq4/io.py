"""Durable atomic persistence for resumable RQ4 pipelines."""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file after flushing it to disk."""
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary_path.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file after flushing it to disk."""
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary_path.replace(path)