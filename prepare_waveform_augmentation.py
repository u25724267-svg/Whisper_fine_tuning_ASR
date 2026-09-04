import argparse
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


_POLICY: Dict[str, Any] = {}
_ASSET_ROOT = Path()
_NOISE_FILES: list[Path] = []
_RIR_FILES: list[Path] = []
_AUDIO_DIR = Path()
_FINAL_AUDIO_DIR = Path()
_GLOBAL_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize deterministic waveform augmentation for RQ2."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(global_seed: int, utterance_id: str) -> int:
    payload = f"rq2-waveform-mild-v1:{global_seed}:{utterance_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_audio(path: Path, sampling_rate: int) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sampling_rate:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=sampling_rate
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    return waveform


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def fit_noise(
    noise: np.ndarray, target_length: int, generator: np.random.Generator
) -> tuple[np.ndarray, int]:
    if len(noise) < target_length:
        repeats = math.ceil(target_length / len(noise))
        noise = np.tile(noise, repeats)
    maximum_start = len(noise) - target_length
    start = int(generator.integers(0, maximum_start + 1)) if maximum_start else 0
    return noise[start : start + target_length], start


def initialize_worker(
    policy: Dict[str, Any],
    asset_root: str,
    noise_files: list[str],
    rir_files: list[str],
    audio_dir: str,
    final_audio_dir: str,
    global_seed: int,
) -> None:
    global _POLICY, _ASSET_ROOT, _NOISE_FILES, _RIR_FILES
    global _AUDIO_DIR, _FINAL_AUDIO_DIR, _GLOBAL_SEED
    _POLICY = policy
    _ASSET_ROOT = Path(asset_root)
    _NOISE_FILES = [Path(value) for value in noise_files]
    _RIR_FILES = [Path(value) for value in rir_files]
    _AUDIO_DIR = Path(audio_dir)
    _FINAL_AUDIO_DIR = Path(final_audio_dir)
    _GLOBAL_SEED = global_seed


def augment_row(index_and_row: tuple[int, Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    index, row = index_and_row
    utterance_id = str(row["id"])
    source_path = Path(row["audio_filepath"]).expanduser().resolve()
    sampling_rate = int(_POLICY["sampling_rate"])
    generator = np.random.default_rng(stable_seed(_GLOBAL_SEED, utterance_id))
    waveform = load_audio(source_path, sampling_rate)
    input_samples = len(waveform)

    speed_applied = bool(generator.random() < _POLICY["speed"]["probability"])
    speed_factor = (
        float(generator.choice(_POLICY["speed"]["factors"]))
        if speed_applied
        else 1.0
    )
    if speed_applied:
        waveform = librosa.effects.time_stretch(waveform, rate=speed_factor)
        waveform = np.asarray(waveform, dtype=np.float32)

    rir_applied = bool(generator.random() < _POLICY["rir"]["probability"])
    rir_path = None
    if rir_applied:
        rir_path = _RIR_FILES[int(generator.integers(0, len(_RIR_FILES)))]
        impulse = load_audio(rir_path, sampling_rate)
        impulse_energy = float(np.linalg.norm(impulse))
        if not math.isfinite(impulse_energy) or impulse_energy <= 0:
            raise ValueError(f"Invalid RIR: {rir_path}")
        impulse = impulse / impulse_energy
        target_length = len(waveform)
        waveform = fftconvolve(waveform, impulse, mode="full")[:target_length]
        waveform = np.asarray(waveform, dtype=np.float32)

    noise_applied = bool(generator.random() < _POLICY["noise"]["probability"])
    noise_path = None
    noise_start = None
    target_snr_db = None
    achieved_snr_db = None
    if noise_applied:
        noise_path = _NOISE_FILES[int(generator.integers(0, len(_NOISE_FILES)))]
        noise = load_audio(noise_path, sampling_rate)
        noise, noise_start = fit_noise(noise, len(waveform), generator)
        target_snr_db = float(
            generator.uniform(
                _POLICY["noise"]["snr_db_min"],
                _POLICY["noise"]["snr_db_max"],
            )
        )
        clean_rms = rms(waveform)
        noise_rms = rms(noise)
        if clean_rms <= 0 or noise_rms <= 0:
            raise ValueError(f"Zero-RMS input for noise mixing: {utterance_id}")
        noise_gain = clean_rms / (noise_rms * 10 ** (target_snr_db / 20))
        scaled_noise = noise * noise_gain
        achieved_snr_db = 20 * math.log10(clean_rms / rms(scaled_noise))
        waveform = waveform + scaled_noise

    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite augmented audio: {utterance_id}")
    peak_before_gain = float(np.max(np.abs(waveform)))
    anti_clipping_gain = (
        min(1.0, float(_POLICY["peak_limit"]) / peak_before_gain)
        if peak_before_gain > 0
        else 1.0
    )
    waveform = np.asarray(waveform * anti_clipping_gain, dtype=np.float32)
    peak_after_gain = float(np.max(np.abs(waveform)))
    if peak_after_gain > float(_POLICY["peak_limit"]) + 1e-6:
        raise RuntimeError(f"Peak limit failed for {utterance_id}")

    output_path = _AUDIO_DIR / f"{utterance_id}.flac"
    final_output_path = _FINAL_AUDIO_DIR / output_path.name
    temporary_path = output_path.with_suffix(".flac.tmp")
    sf.write(temporary_path, waveform, sampling_rate, format="FLAC", subtype="PCM_16")
    temporary_path.replace(output_path)

    augmented_row = {
        **row,
        "audio_filepath": str(final_output_path),
        "duration": len(waveform) / sampling_rate,
    }
    parameters = {
        "index": index,
        "id": utterance_id,
        "source_audio": str(source_path),
        "output_audio": str(final_output_path),
        "per_utterance_seed": stable_seed(_GLOBAL_SEED, utterance_id),
        "input_samples": input_samples,
        "output_samples": len(waveform),
        "speed_applied": speed_applied,
        "speed_factor": speed_factor,
        "rir_applied": rir_applied,
        "rir_asset": None if rir_path is None else str(rir_path.relative_to(_ASSET_ROOT)),
        "noise_applied": noise_applied,
        "noise_asset": None if noise_path is None else str(noise_path.relative_to(_ASSET_ROOT)),
        "noise_start_sample": noise_start,
        "target_snr_db": target_snr_db,
        "achieved_snr_db": achieved_snr_db,
        "peak_before_gain": peak_before_gain,
        "anti_clipping_gain": anti_clipping_gain,
        "peak_after_gain": peak_after_gain,
        "output_sha256": sha256_file(output_path),
    }
    return augmented_row, parameters


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = args.train_manifest.expanduser().resolve()
    policy_path = args.policy.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Output or staging directory already exists: {output_dir}")
    if args.workers <= 0:
        raise ValueError("Workers must be positive")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    archive_path = asset_root.parent / "rirs_noises.zip"
    if sha256_file(archive_path) != policy["asset_archive_sha256"]:
        raise ValueError("SLR28 archive hash does not match the frozen policy")
    noise_files = sorted(asset_root.glob(policy["noise"]["asset_glob"]))
    rir_files = sorted(asset_root.glob(policy["rir"]["asset_glob"]))
    if not noise_files or not rir_files:
        raise FileNotFoundError("Frozen noise or RIR asset pool is empty")

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Training manifest must contain unique non-empty rows")

    try:
        audio_dir = staging_dir / "audio"
        audio_dir.mkdir(parents=True)
        initializer_args = (
            policy,
            str(asset_root),
            [str(path) for path in noise_files],
            [str(path) for path in rir_files],
            str(audio_dir),
            str(output_dir / "audio"),
            args.seed,
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=initializer_args,
        ) as executor:
            results = list(executor.map(augment_row, enumerate(rows), chunksize=8))
        augmented_rows = [result[0] for result in results]
        parameters = [result[1] for result in results]
        write_jsonl(staging_dir / "train.jsonl", augmented_rows)
        write_jsonl(staging_dir / "augmentation_parameters.jsonl", parameters)
        summary = {
            "schema_version": 1,
            "seed": args.seed,
            "rows": len(rows),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "policy": str(policy_path),
            "policy_sha256": sha256_file(policy_path),
            "asset_archive": str(archive_path),
            "asset_archive_sha256": sha256_file(archive_path),
            "noise_pool_files": len(noise_files),
            "rir_pool_files": len(rir_files),
            "speed_applied_rows": sum(row["speed_applied"] for row in parameters),
            "rir_applied_rows": sum(row["rir_applied"] for row in parameters),
            "noise_applied_rows": sum(row["noise_applied"] for row in parameters),
            "clean_rows": sum(
                not row["speed_applied"]
                and not row["rir_applied"]
                and not row["noise_applied"]
                for row in parameters
            ),
            "anti_clipping_rows": sum(
                row["anti_clipping_gain"] < 1.0 for row in parameters
            ),
            "train_manifest_sha256": sha256_file(staging_dir / "train.jsonl"),
            "parameters_sha256": sha256_file(
                staging_dir / "augmentation_parameters.jsonl"
            ),
        }
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