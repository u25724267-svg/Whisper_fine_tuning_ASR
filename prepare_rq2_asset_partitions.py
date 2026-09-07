import argparse
import hashlib
import json
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


DEFAULT_SALT = "rq2-followup-v1-assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create content-disjoint SLR28 train/validation/test inventories."
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-glob", default="pointsource_noises/*.wav")
    parser.add_argument("--rir-glob", default="simulated_rirs/**/*.wav")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--train-weight", type=int, default=8)
    parser.add_argument("--validation-weight", type=int, default=1)
    parser.add_argument("--test-weight", type=int, default=1)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_audio_identity(path: Path) -> tuple[str, int, int, int]:
    waveform, sampling_rate = sf.read(
        path, dtype="float32", always_2d=True
    )
    waveform = np.asarray(waveform, dtype="<f4", order="C")
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio asset: {path}")
    frames, channels = waveform.shape
    digest = hashlib.sha256()
    digest.update(struct.pack("<IQQ", int(sampling_rate), frames, channels))
    digest.update(waveform.tobytes(order="C"))
    return digest.hexdigest(), int(sampling_rate), frames, channels


def assign_partition(
    content_sha256: str,
    salt: str,
    train_weight: int,
    validation_weight: int,
    test_weight: int,
) -> str:
    weights = (train_weight, validation_weight, test_weight)
    if any(weight <= 0 for weight in weights):
        raise ValueError("Partition weights must be positive")
    bucket_digest = hashlib.sha256(
        f"{salt}:{content_sha256}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(bucket_digest[:8], "big") % sum(weights)
    if bucket < train_weight:
        return "train"
    if bucket < train_weight + validation_weight:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def inventory_rows(
    asset_root: Path,
    kind: str,
    pattern: str,
    salt: str,
    train_weight: int,
    validation_weight: int,
    test_weight: int,
) -> list[dict[str, Any]]:
    paths = sorted(path for path in asset_root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No {kind} assets match {pattern} under {asset_root}")
    rows = []
    for path in paths:
        content_sha256, sampling_rate, frames, channels = canonical_audio_identity(path)
        rows.append(
            {
                "kind": kind,
                "relative_path": str(path.relative_to(asset_root)),
                "file_sha256": sha256_file(path),
                "content_sha256": content_sha256,
                "partition": assign_partition(
                    content_sha256,
                    salt,
                    train_weight,
                    validation_weight,
                    test_weight,
                ),
                "sampling_rate": sampling_rate,
                "frames": frames,
                "channels": channels,
                "duration_seconds": frames / sampling_rate,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if not asset_root.is_dir():
        raise FileNotFoundError(f"Missing asset root: {asset_root}")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Output or staging path already exists: {output_dir}")
    if not args.salt.strip():
        raise ValueError("Partition salt must not be empty")

    archive_path = (
        args.archive.expanduser().resolve()
        if args.archive
        else asset_root.parent / "rirs_noises.zip"
    )
    archive_sha256 = None
    if args.expected_archive_sha256:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing SLR28 archive: {archive_path}")
        archive_sha256 = sha256_file(archive_path)
        if archive_sha256 != args.expected_archive_sha256:
            raise ValueError("SLR28 archive hash does not match the expected value")
    elif archive_path.is_file():
        archive_sha256 = sha256_file(archive_path)

    try:
        staging_dir.mkdir(parents=True)
        rows = inventory_rows(
            asset_root,
            "noise",
            args.noise_glob,
            args.salt,
            args.train_weight,
            args.validation_weight,
            args.test_weight,
        )
        rows.extend(
            inventory_rows(
                asset_root,
                "rir",
                args.rir_glob,
                args.salt,
                args.train_weight,
                args.validation_weight,
                args.test_weight,
            )
        )
        rows.sort(key=lambda row: (row["kind"], row["relative_path"]))

        partitions_by_content: dict[str, set[str]] = {}
        for row in rows:
            partitions_by_content.setdefault(row["content_sha256"], set()).add(
                row["partition"]
            )
        crossing = [
            content_sha256
            for content_sha256, partitions in partitions_by_content.items()
            if len(partitions) != 1
        ]
        if crossing:
            raise RuntimeError(
                f"Content leakage across partitions for {len(crossing)} assets"
            )

        inventory_path = staging_dir / "inventory.jsonl"
        write_jsonl(inventory_path, rows)
        for partition in ("train", "validation", "test"):
            write_jsonl(
                staging_dir / f"{partition}.jsonl",
                (row for row in rows if row["partition"] == partition),
            )

        count_by_kind_partition = Counter(
            (row["kind"], row["partition"]) for row in rows
        )
        unique_content = {row["content_sha256"] for row in rows}
        summary = {
            "schema_version": 1,
            "protocol": "rq2-followup-v1",
            "asset_root": str(asset_root),
            "noise_glob": args.noise_glob,
            "rir_glob": args.rir_glob,
            "salt": args.salt,
            "weights": {
                "train": args.train_weight,
                "validation": args.validation_weight,
                "test": args.test_weight,
            },
            "asset_archive": str(archive_path) if archive_path.is_file() else None,
            "asset_archive_sha256": archive_sha256,
            "rows": len(rows),
            "unique_content": len(unique_content),
            "duplicate_content_rows": len(rows) - len(unique_content),
            "counts": {
                kind: {
                    partition: count_by_kind_partition[(kind, partition)]
                    for partition in ("train", "validation", "test")
                }
                for kind in ("noise", "rir")
            },
            "inventory_sha256": sha256_file(inventory_path),
            "partition_manifest_sha256": {
                partition: sha256_file(staging_dir / f"{partition}.jsonl")
                for partition in ("train", "validation", "test")
            },
        }
        if any(
            summary["counts"][kind][partition] == 0
            for kind in ("noise", "rir")
            for partition in ("train", "validation", "test")
        ):
            raise RuntimeError("Every asset kind must be represented in every partition")
        (staging_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        staging_dir.replace(output_dir)
        print(json.dumps(summary, indent=2))
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


if __name__ == "__main__":
    main()