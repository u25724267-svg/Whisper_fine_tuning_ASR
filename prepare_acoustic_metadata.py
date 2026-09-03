import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import soundfile as sf


DEFAULT_PROTOCOL_DIR = Path("/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2")
DEFAULT_WAXAL_HASHES = Path(
    "/home/casper/Speech/data/waxal/sna_asr/sna_asr_audio_hashes.csv"
)
SPLITS = ("train", "validation", "test")
FRAME_SECONDS = 0.1
ACTIVITY_DB_BELOW_PEAK = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute audited acoustic metadata for speaker-disjoint WAXAL."
    )
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--waxal-hashes", type=Path, default=DEFAULT_WAXAL_HASHES)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, split: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({**row, "split": split, "manifest_line": line_number})
    return rows


def amplitude_dbfs(value: float) -> float:
    return 20 * math.log10(max(float(value), np.finfo(np.float64).tiny))


def analyze_audio(row: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = Path(row["audio_filepath"])
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing audio: {audio_path}")
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    measured_duration = len(audio) / sample_rate
    mono = audio.mean(axis=1)
    absolute_audio = np.abs(audio)
    sample_peak = float(absolute_audio.max(initial=0.0))
    global_rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64, copy=False)))))

    frame_size = max(1, round(sample_rate * FRAME_SECONDS))
    energy = np.square(mono.astype(np.float64, copy=False))
    if padding := (-len(energy)) % frame_size:
        energy = np.pad(energy, (0, padding))
    frame_rms = np.sqrt(energy.reshape(-1, frame_size).mean(axis=1))
    peak_frame_rms = float(frame_rms.max(initial=0.0))
    activity_threshold = peak_frame_rms * 10 ** (-ACTIVITY_DB_BELOW_PEAK / 20)
    active_windows = (
        np.flatnonzero(frame_rms >= activity_threshold)
        if peak_frame_rms
        else np.array([], dtype=int)
    )

    if len(active_windows):
        leading_silence = min(measured_duration, int(active_windows[0]) * FRAME_SECONDS)
        active_end = min(
            measured_duration, (int(active_windows[-1]) + 1) * FRAME_SECONDS
        )
        trailing_silence = max(0.0, measured_duration - active_end)
        speech_ratio = len(active_windows) / len(frame_rms)
        speech_rms = float(np.percentile(frame_rms[active_windows], 75))
        noise_rms = float(np.percentile(frame_rms, 20))
        snr_proxy_db = min(
            60.0, amplitude_dbfs(speech_rms) - amplitude_dbfs(noise_rms)
        )
    else:
        leading_silence = measured_duration
        trailing_silence = measured_duration
        speech_ratio = 0.0
        speech_rms = 0.0
        noise_rms = 0.0
        snr_proxy_db = 0.0

    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2", copy=False)
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{audio.shape[1]}:".encode())
    digest.update(pcm.tobytes())

    return {
        "split": row["split"],
        "id": str(row["id"]),
        "speaker_id": str(row["speaker_id"]),
        "audio_filepath": str(audio_path),
        "manifest_duration": float(row["duration"]),
        "measured_duration": measured_duration,
        "duration_error": abs(measured_duration - float(row["duration"])),
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "peak_dbfs": amplitude_dbfs(sample_peak),
        "rms_dbfs": amplitude_dbfs(global_rms),
        "speech_ratio": speech_ratio,
        "speech_rms": speech_rms,
        "noise_rms": noise_rms,
        "leading_silence": leading_silence,
        "trailing_silence": trailing_silence,
        "snr_proxy_db": snr_proxy_db,
        "pcm_sha256": digest.hexdigest(),
    }


def load_reference_hashes(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return {str(row["id"]): row for row in csv.DictReader(source)}


def percentile_summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    waxal_hash_path = args.waxal_hashes.expanduser().resolve()
    output_path = protocol_dir / "acoustic_metadata.csv"
    summary_path = protocol_dir / "acoustic_metadata_summary.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Acoustic metadata already exists under {protocol_dir}; refusing to overwrite"
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not waxal_hash_path.is_file():
        raise FileNotFoundError(f"Missing WAXAL reference hashes: {waxal_hash_path}")

    manifest_paths = {split: protocol_dir / f"{split}.jsonl" for split in SPLITS}
    missing_manifests = [str(path) for path in manifest_paths.values() if not path.is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing protocol manifests: {missing_manifests}")
    rows = [
        row
        for split in SPLITS
        for row in read_jsonl(manifest_paths[split], split)
    ]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Protocol contains duplicate utterance IDs")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metadata = list(pool.map(analyze_audio, rows))

    reference_hashes = load_reference_hashes(waxal_hash_path)
    missing_hashes = [row["id"] for row in metadata if row["id"] not in reference_hashes]
    matching_reference_paths = [
        row
        for row in metadata
        if row["id"] in reference_hashes
        and row["audio_filepath"] == reference_hashes[row["id"]]["audit_audio_filepath"]
    ]
    changed_reference_paths = [
        row
        for row in metadata
        if row["id"] in reference_hashes
        and row["audio_filepath"] != reference_hashes[row["id"]]["audit_audio_filepath"]
    ]
    mismatched_hashes = [
        row["id"]
        for row in matching_reference_paths
        if row["pcm_sha256"] != reference_hashes[row["id"]]["pcm_sha256"]
    ]
    if missing_hashes or mismatched_hashes:
        raise ValueError(
            f"PCM hash validation failed: {len(missing_hashes)} missing, "
            f"{len(mismatched_hashes)} mismatched"
        )
    if any(row["channels"] != 1 for row in metadata):
        raise ValueError("Protocol contains non-mono audio")
    if any(not math.isfinite(row["snr_proxy_db"]) for row in metadata):
        raise ValueError("Protocol contains non-finite SNR values")
    if any(row["duration_error"] > 0.05 for row in metadata):
        raise ValueError("Protocol contains duration errors above 50 ms")

    columns = list(metadata[0])
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metadata)
    temporary_path.replace(output_path)

    summary = {
        "schema_version": 1,
        "protocol": "waxal_shona_speaker_disjoint_v2",
        "method": {
            "frame_seconds": FRAME_SECONDS,
            "activity_db_below_peak": ACTIVITY_DB_BELOW_PEAK,
            "speech_rms": "75th percentile of active frame RMS",
            "noise_rms": "20th percentile of all frame RMS",
            "snr_proxy_db": "min(60, dbfs(speech_rms) - dbfs(noise_rms))",
        },
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
        "manifests": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": sum(row["split"] == split for row in metadata),
            }
            for split, path in manifest_paths.items()
        },
        "metadata": str(output_path),
        "metadata_sha256": sha256_file(output_path),
        "reference_hashes": str(waxal_hash_path),
        "reference_hashes_sha256": sha256_file(waxal_hash_path),
        "pcm_hashes_missing": 0,
        "same_path_hashes_verified": len(matching_reference_paths),
        "same_path_hashes_mismatched": 0,
        "changed_trimmed_audio_paths": len(changed_reference_paths),
        "native_sample_rates": dict(
            sorted(Counter(row["sample_rate"] for row in metadata).items())
        ),
        "native_channels": dict(
            sorted(Counter(row["channels"] for row in metadata).items())
        ),
        "maximum_duration_error_seconds": max(row["duration_error"] for row in metadata),
        "splits": {
            split: {
                "snr_proxy_db": percentile_summary(
                    row["snr_proxy_db"] for row in metadata if row["split"] == split
                ),
                "speech_ratio": percentile_summary(
                    row["speech_ratio"] for row in metadata if row["split"] == split
                ),
                "duration": percentile_summary(
                    row["measured_duration"] for row in metadata if row["split"] == split
                ),
            }
            for split in SPLITS
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()