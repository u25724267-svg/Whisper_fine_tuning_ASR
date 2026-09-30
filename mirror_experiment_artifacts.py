import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT_FILES = {
    "all_results.json",
    "experiment_config.json",
    "run_manifest.json",
    "test_results.json",
    "train_results.json",
    "trainer_state.json",
    "validation_results.json",
}
ARTIFACT_DIRECTORIES = {
    "curriculum_orders",
    "curriculum_scores",
    "fleurs_corrected_v2",
    "item_predictions",
    "logs",
}
ALLOWED_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md"}
PRESERVED_LEGACY_PREFIXES = ("wandb/",)
COMPLETION_FILES = {
    "train_results.json",
    "validation_results.json",
    "item_predictions/summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically mirror non-model experiment results into the repository."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Atomically refresh an existing mirror from the same source directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(source_dir: Path) -> list[Path]:
    files = [
        source_dir / name
        for name in sorted(ROOT_FILES)
        if (source_dir / name).is_file()
    ]
    for directory_name in sorted(ARTIFACT_DIRECTORIES):
        directory = source_dir / directory_name
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
        )
    return files


def build_entries(source_dir: Path, files: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(source_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def verify_existing_mirror(
    source_dir: Path, output_dir: Path, expected_entries: list[dict[str, object]]
) -> bool:
    manifest_path = output_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_dir") != str(source_dir):
        return False
    return manifest.get("entries") == expected_entries


def verify_unmanaged_existing_mirror(
    source_dir: Path,
    output_dir: Path,
    expected_entries: list[dict[str, object]],
) -> bool:
    expected_by_path = {str(entry["path"]): entry for entry in expected_entries}
    existing_files = [path for path in output_dir.rglob("*") if path.is_file()]
    if not existing_files:
        return False
    for path in existing_files:
        relative_path = path.relative_to(output_dir).as_posix()
        expected = expected_by_path.get(relative_path)
        if expected is None:
            if relative_path.startswith(PRESERVED_LEGACY_PREFIXES):
                continue
            return False
        if path.stat().st_size != expected["bytes"]:
            return False
        if sha256_file(path) != expected["sha256"]:
            return False
        if not (source_dir / relative_path).is_file():
            return False
    return True


def mirror_experiment(
    source_dir: Path, output_dir: Path, *, refresh: bool = False
) -> dict[str, object]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing experiment output: {source_dir}")
    missing = [
        relative_path
        for relative_path in sorted(COMPLETION_FILES)
        if not (source_dir / relative_path).is_file()
    ]
    if missing:
        raise ValueError(f"Experiment output is incomplete: {', '.join(missing)}")

    files = selected_files(source_dir)
    entries = build_entries(source_dir, files)
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    if output_dir.exists():
        if verify_existing_mirror(source_dir, output_dir, entries):
            return {
                "status": "already-mirrored",
                "output_dir": str(output_dir),
                "files": len(entries),
            }
        if not refresh:
            raise FileExistsError(f"Artifact destination already exists: {output_dir}")
        manifest_path = output_dir / "artifact_manifest.json"
        if manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest.get("source_dir") != str(source_dir):
                raise ValueError(f"Existing mirror source does not match: {output_dir}")
        elif not verify_unmanaged_existing_mirror(source_dir, output_dir, entries):
            raise ValueError(
                f"Existing unmanaged mirror does not match its source: {output_dir}"
            )
    if staging_dir.exists():
        raise FileExistsError(f"Artifact staging destination exists: {staging_dir}")
    if backup_dir.exists():
        raise FileExistsError(f"Artifact backup destination exists: {backup_dir}")

    try:
        if output_dir.exists():
            shutil.copytree(output_dir, staging_dir)
        else:
            staging_dir.mkdir(parents=True)
        for source_path, entry in zip(files, entries):
            destination = staging_dir / str(entry["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            if sha256_file(destination) != entry["sha256"]:
                raise ValueError(f"Artifact hash mismatch after copy: {destination}")
        manifest = {
            "schema_version": 1,
            "selection": "non-model result and audit artifacts",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "files": len(entries),
            "bytes": sum(int(entry["bytes"]) for entry in entries),
            "entries": entries,
        }
        (staging_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        status = "mirrored"
        if output_dir.exists():
            output_dir.replace(backup_dir)
            try:
                staging_dir.replace(output_dir)
            except BaseException:
                backup_dir.replace(output_dir)
                raise
            shutil.rmtree(backup_dir)
            status = "refreshed"
        else:
            staging_dir.replace(output_dir)
        return {"status": status, **manifest}
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def main() -> None:
    args = parse_args()
    result = mirror_experiment(args.source_dir, args.output_dir, refresh=args.refresh)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()