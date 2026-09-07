import argparse
import hashlib
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


CONDITIONS = {"clean", "noise", "rir", "noise-rir", "speed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize leakage-controlled RQ2 follow-up audio and manifests."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--clean-probability", type=float, default=0.0)
    parser.add_argument("--snr-db", type=float, action="append", default=[])
    parser.add_argument("--speed-factor", type=float, action="append", default=[])
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument(
        "--asset-partition", choices=["train", "validation", "test"]
    )
    parser.add_argument("--allow-test-partition", action="store_true")
    parser.add_argument("--peak-limit", type=float, default=0.99)
    parser.add_argument("--max-output-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, utterance_id: str, condition: str) -> int:
    payload = f"rq2-followup-v1:{condition}:{seed}:{utterance_id}".encode("utf-8")
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


def root_mean_square(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def fit_noise(
    noise: np.ndarray, target_length: int, generator: np.random.Generator
) -> tuple[np.ndarray, int, int]:
    if len(noise) < target_length:
        raise ValueError("Noise must be at least as long as the target waveform")
    maximum_start = len(noise) - target_length
    start = int(generator.integers(0, maximum_start + 1)) if maximum_start else 0
    return noise[start : start + target_length], start, 1


def add_noise(
    waveform: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    fitted_noise, start, repeats = fit_noise(noise, len(waveform), generator)
    speech_rms = root_mean_square(waveform)
    noise_rms = root_mean_square(fitted_noise)
    if speech_rms <= 0 or noise_rms <= 0:
        raise ValueError("Speech and noise must have positive RMS")
    noise_gain = speech_rms / (noise_rms * 10 ** (snr_db / 20))
    scaled_noise = fitted_noise * noise_gain
    achieved_snr_db = 20 * math.log10(speech_rms / root_mean_square(scaled_noise))
    return np.asarray(waveform + scaled_noise, dtype=np.float32), {
        "noise_start_sample": start,
        "noise_repeats": repeats,
        "noise_gain": noise_gain,
        "target_snr_db": snr_db,
        "achieved_snr_db": achieved_snr_db,
    }


def apply_rir(
    waveform: np.ndarray,
    impulse: np.ndarray,
    maximum_samples: int,
) -> tuple[np.ndarray, bool]:
    impulse_energy = float(np.linalg.norm(impulse))
    if not math.isfinite(impulse_energy) or impulse_energy <= 0:
        raise ValueError("RIR must have positive finite energy")
    reverberant = fftconvolve(waveform, impulse / impulse_energy, mode="full")
    truncated = len(reverberant) > maximum_samples
    return np.asarray(reverberant[:maximum_samples], dtype=np.float32), truncated


def apply_speed(waveform: np.ndarray, factor: float) -> np.ndarray:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Speed factors must be positive and finite")
    if math.isclose(factor, 1.0):
        return waveform.copy()
    ratio = Fraction(1 / factor).limit_denominator(1000)
    return np.asarray(
        resample_poly(waveform, up=ratio.numerator, down=ratio.denominator),
        dtype=np.float32,
    )


def load_asset_rows(
    manifest_path: Path,
    asset_root: Path,
    expected_partition: str,
) -> dict[str, list[dict[str, Any]]]:
    assets: dict[str, list[dict[str, Any]]] = {"noise": [], "rir": []}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        row = json.loads(line)
        kind = row.get("kind")
        if kind not in assets:
            raise ValueError(f"Invalid asset kind at {manifest_path}:{line_number}")
        if row.get("partition") != expected_partition:
            raise ValueError(
                f"Asset partition mismatch at {manifest_path}:{line_number}"
            )
        path = asset_root / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing asset: {path}")
        assets[kind].append({**row, "path": path})
    return assets


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.clean_probability <= 1.0:
        raise ValueError("Clean probability must be between zero and one")
    if args.sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")
    if not 0 < args.peak_limit <= 1:
        raise ValueError("Peak limit must be in (0, 1]")
    if args.max_output_seconds <= 0:
        raise ValueError("Maximum output duration must be positive")
    if args.condition in {"noise", "noise-rir"} and not args.snr_db:
        raise ValueError("Noise conditions require at least one --snr-db")
    if args.condition == "speed" and not args.speed_factor:
        raise ValueError("Speed condition requires at least one --speed-factor")
    if any(not math.isfinite(value) for value in args.snr_db):
        raise ValueError("SNR values must be finite")
    if any(
        not math.isfinite(value) or value <= 0 for value in args.speed_factor
    ):
        raise ValueError("Speed factors must be positive and finite")
    if args.asset_partition == "test" and not args.allow_test_partition:
        raise PermissionError("Test assets require --allow-test-partition")
    needs_assets = args.condition in {"noise", "rir", "noise-rir"}
    if needs_assets and not (
        args.asset_root and args.asset_manifest and args.asset_partition
    ):
        raise ValueError(
            "Noise/RIR conditions require --asset-root, --asset-manifest, "
            "and --asset-partition"
        )
    supplied_asset_arguments = (
        args.asset_root is not None,
        args.asset_manifest is not None,
        args.asset_partition is not None,
    )
    if any(supplied_asset_arguments) and not all(supplied_asset_arguments):
        raise ValueError("Asset root, manifest, and partition must be supplied together")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source_manifest = args.source_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest}")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Output or staging path already exists: {output_dir}")

    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(row.get("id", "")) for row in rows]
    if not rows or any(not utterance_id for utterance_id in ids):
        raise ValueError("Source manifest must contain non-empty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest IDs must be unique")

    asset_manifest = None
    assets: dict[str, list[dict[str, Any]]] = {"noise": [], "rir": []}
    if args.asset_manifest:
        asset_manifest = args.asset_manifest.expanduser().resolve()
        asset_root = args.asset_root.expanduser().resolve()
        if not asset_manifest.is_file():
            raise FileNotFoundError(f"Missing asset manifest: {asset_manifest}")
        if not asset_root.is_dir():
            raise FileNotFoundError(f"Missing asset root: {asset_root}")
        assets = load_asset_rows(asset_manifest, asset_root, args.asset_partition)
        if args.condition in {"noise", "noise-rir"} and not assets["noise"]:
            raise ValueError("Selected partition contains no noise assets")
        if args.condition in {"rir", "noise-rir"} and not assets["rir"]:
            raise ValueError("Selected partition contains no RIR assets")

    try:
        audio_dir = staging_dir / "audio"
        audio_dir.mkdir(parents=True)
        output_rows = []
        parameter_rows = []
        verified_assets: set[Path] = set()
        maximum_samples = round(args.max_output_seconds * args.sampling_rate)
        for index, row in enumerate(rows):
            utterance_id = str(row["id"])
            source_path = Path(row["audio_filepath"]).expanduser().resolve()
            generator = np.random.default_rng(
                stable_seed(args.seed, utterance_id, args.condition)
            )
            use_clean = bool(generator.random() < args.clean_probability)
            waveform = load_audio(source_path, args.sampling_rate)
            input_samples = len(waveform)
            parameters: dict[str, Any] = {
                "index": index,
                "id": utterance_id,
                "condition": args.condition,
                "seed": stable_seed(args.seed, utterance_id, args.condition),
                "clean_presentation": use_clean or args.condition == "clean",
                "source_audio": str(source_path),
                "input_samples": input_samples,
                "noise_asset": None,
                "rir_asset": None,
                "speed_factor": 1.0,
            }

            if not use_clean and args.condition != "clean":
                if args.condition in {"rir", "noise-rir"}:
                    rir_row = assets["rir"][
                        int(generator.integers(0, len(assets["rir"])))
                    ]
                    if rir_row["path"] not in verified_assets:
                        if sha256_file(rir_row["path"]) != rir_row["file_sha256"]:
                            raise ValueError(f"RIR asset hash mismatch: {rir_row['path']}")
                        verified_assets.add(rir_row["path"])
                    impulse = load_audio(rir_row["path"], args.sampling_rate)
                    waveform, rir_truncated = apply_rir(
                        waveform, impulse, maximum_samples
                    )
                    parameters.update(
                        {
                            "rir_asset": rir_row["relative_path"],
                            "rir_content_sha256": rir_row["content_sha256"],
                            "rir_tail_truncated": rir_truncated,
                        }
                    )
                if args.condition in {"noise", "noise-rir"}:
                    minimum_noise_seconds = len(waveform) / args.sampling_rate
                    eligible_noise_rows = [
                        row
                        for row in assets["noise"]
                        if float(row["duration_seconds"]) >= minimum_noise_seconds
                    ]
                    if not eligible_noise_rows:
                        raise ValueError(
                            f"No duration-compatible noise for {utterance_id} "
                            f"({minimum_noise_seconds:.3f} seconds)"
                        )
                    noise_row = eligible_noise_rows[
                        int(generator.integers(0, len(eligible_noise_rows)))
                    ]
                    if noise_row["path"] not in verified_assets:
                        if sha256_file(noise_row["path"]) != noise_row["file_sha256"]:
                            raise ValueError(
                                f"Noise asset hash mismatch: {noise_row['path']}"
                            )
                        verified_assets.add(noise_row["path"])
                    snr_db = float(generator.choice(args.snr_db))
                    noise = load_audio(noise_row["path"], args.sampling_rate)
                    waveform, noise_parameters = add_noise(
                        waveform, noise, snr_db, generator
                    )
                    parameters.update(
                        {
                            "noise_asset": noise_row["relative_path"],
                            "noise_content_sha256": noise_row["content_sha256"],
                            **noise_parameters,
                        }
                    )
                if args.condition == "speed":
                    speed_factor = float(generator.choice(args.speed_factor))
                    waveform = apply_speed(waveform, speed_factor)
                    parameters["speed_factor"] = speed_factor
                    parameters["clean_presentation"] = math.isclose(
                        speed_factor, 1.0
                    )

            if parameters["clean_presentation"]:
                parameters.update(
                    {
                        "output_audio": str(source_path),
                        "output_samples": input_samples,
                        "peak_before_gain": float(np.max(np.abs(waveform))),
                        "anti_clipping_gain": 1.0,
                        "maximum_duration_truncated": False,
                        "output_sha256": sha256_file(source_path),
                    }
                )
                output_rows.append(row)
                parameter_rows.append(parameters)
                continue

            if len(waveform) > maximum_samples:
                waveform = waveform[:maximum_samples]
                parameters["maximum_duration_truncated"] = True
            else:
                parameters["maximum_duration_truncated"] = False
            if not np.isfinite(waveform).all():
                raise ValueError(f"Non-finite output for {utterance_id}")
            peak_before_gain = float(np.max(np.abs(waveform)))
            gain = (
                min(1.0, args.peak_limit / peak_before_gain)
                if peak_before_gain > 0
                else 1.0
            )
            waveform = np.asarray(waveform * gain, dtype=np.float32)
            output_path = audio_dir / f"{utterance_id}.flac"
            sf.write(output_path, waveform, args.sampling_rate, subtype="PCM_16")
            final_path = output_dir / "audio" / output_path.name
            output_rows.append(
                {
                    **row,
                    "audio_filepath": str(final_path),
                    "duration": len(waveform) / args.sampling_rate,
                }
            )
            parameters.update(
                {
                    "output_audio": str(final_path),
                    "output_samples": len(waveform),
                    "peak_before_gain": peak_before_gain,
                    "anti_clipping_gain": gain,
                    "output_sha256": sha256_file(output_path),
                }
            )
            parameter_rows.append(parameters)

        manifest_path = staging_dir / "manifest.jsonl"
        parameters_path = staging_dir / "parameters.jsonl"
        write_jsonl(manifest_path, output_rows)
        write_jsonl(parameters_path, parameter_rows)
        transformed_rows = sum(
            not row["clean_presentation"] for row in parameter_rows
        )
        summary = {
            "schema_version": 1,
            "protocol": "rq2-followup-v1",
            "condition": args.condition,
            "seed": args.seed,
            "rows": len(rows),
            "clean_probability": args.clean_probability,
            "clean_rows": len(rows) - transformed_rows,
            "transformed_rows": transformed_rows,
            "snr_db": args.snr_db,
            "speed_factors": args.speed_factor,
            "sampling_rate": args.sampling_rate,
            "peak_limit": args.peak_limit,
            "max_output_seconds": args.max_output_seconds,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "asset_partition": args.asset_partition,
            "asset_manifest": str(asset_manifest) if asset_manifest else None,
            "asset_manifest_sha256": (
                sha256_file(asset_manifest) if asset_manifest else None
            ),
            "output_manifest_sha256": sha256_file(manifest_path),
            "parameters_sha256": sha256_file(parameters_path),
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