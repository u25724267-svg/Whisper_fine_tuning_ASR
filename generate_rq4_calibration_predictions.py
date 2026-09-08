import argparse
import importlib.metadata
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from generate_rq4_teacher_labels import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_sha256,
    inference_batch,
    percentile,
    validate_existing_chunk,
)
from inventory_waxal_unlabeled import sha256_file


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path(
    "/ext_data/casper/asr_data/waxal_shona_speaker_disjoint_v2/train.jsonl"
)
DEFAULT_TEACHER = Path(
    "/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2"
)
DEFAULT_OUTPUT = Path("/ext_data/casper/asr_data/rq4_calibration_c0_train_v1")
AMENDMENT_PATH = (
    ROOT_DIR / "documents" / "data" / "rq2_rq4_parameter_sweep_amendment_v2.md"
)
EXPECTED_MANIFEST_SHA256 = (
    "38bb70aa18bb0ebd8d00321ebf9b463f21a35f5507f74e2b2279bdb3caf56b4c"
)
EXPECTED_MODEL_SHA256 = (
    "a3e4df1999147a9170078cdde23d2c340f85b67f13f650aff7210d47fcec7e9c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped labelled-training predictions for RQ4 calibration."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=225)
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def load_audio(path: Path, sampling_rate: int) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sampling_rate:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=sampling_rate
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid calibration audio: {path}")
    return waveform


def error_record(reference: str, prediction: str) -> dict[str, Any]:
    word_output = jiwer.process_words(reference, prediction)
    character_output = jiwer.process_characters(reference, prediction)
    reference_words = (
        word_output.hits + word_output.substitutions + word_output.deletions
    )
    reference_characters = (
        character_output.hits
        + character_output.substitutions
        + character_output.deletions
    )
    return {
        "reference_normalized": reference,
        "teacher_wer": float(word_output.wer),
        "teacher_cer": float(character_output.cer),
        "word_substitutions": int(word_output.substitutions),
        "word_deletions": int(word_output.deletions),
        "word_insertions": int(word_output.insertions),
        "word_hits": int(word_output.hits),
        "reference_words": int(reference_words),
        "character_substitutions": int(character_output.substitutions),
        "character_deletions": int(character_output.deletions),
        "character_insertions": int(character_output.insertions),
        "character_hits": int(character_output.hits),
        "reference_characters": int(reference_characters),
    }


def build_provenance(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    teacher_dir = args.teacher_dir.expanduser().resolve()
    paths = {
        "manifest": manifest,
        "teacher_model": teacher_dir / "model.safetensors",
        "teacher_config": teacher_dir / "config.json",
        "teacher_generation_config": teacher_dir / "generation_config.json",
        "amendment": AMENDMENT_PATH,
        "teacher_inference": ROOT_DIR / "generate_rq4_teacher_labels.py",
        "script": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RQ4 calibration prerequisites: {missing}")
    if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Calibration manifest does not match frozen training data")
    if sha256_file(teacher_dir / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("Calibration teacher does not match frozen weights")
    return {
        "schema_version": 1,
        "protocol": "rq4-calibration-c0-train-v1",
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "parameters": {
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "max_length": args.max_length,
            "sampling_rate": args.sampling_rate,
            "seed": args.seed,
            "limit": args.limit,
            "device": args.device,
            "normalizer": "BasicTextNormalizer(remove_diacritics=False)",
            "decoding": {
                "language": "shona",
                "task": "transcribe",
                "do_sample": False,
                "num_beams": 1,
                "return_timestamps": False,
            },
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jiwer", "librosa", "numpy", "soundfile", "torch", "transformers")
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be positive")
    if args.chunk_size % args.batch_size != 0:
        raise ValueError("Chunk size must be divisible by batch size")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration requested but CUDA is unavailable")

    provenance = build_provenance(args)
    provenance_sha256 = canonical_json_sha256(provenance)
    output_dir = args.output_dir.expanduser().resolve()
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(existing) != provenance_sha256:
            raise ValueError("Existing calibration provenance differs")
    else:
        atomic_write_json(provenance_path, provenance)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"Calibration inference already complete: {output_dir}")

    manifest_path = args.manifest.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Calibration manifest IDs must be unique and non-empty")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    teacher_dir = args.teacher_dir.expanduser().resolve()
    processor = WhisperProcessor.from_pretrained(teacher_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        teacher_dir, torch_dtype=dtype
    ).to(device)
    model.eval()
    no_timestamps_token_id = model.generation_config.no_timestamps_token_id
    if no_timestamps_token_id is None:
        raise ValueError("Teacher has no no-timestamps token ID")
    no_speech_token_id = int(no_timestamps_token_id) - 1
    normalizer = BasicTextNormalizer(remove_diacritics=False)

    chunk_paths = []
    for chunk_start in range(0, len(rows), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(rows))
        chunk_index = chunk_start // args.chunk_size
        part_path = chunks_dir / f"part-{chunk_index:06d}.jsonl"
        marker_path = chunks_dir / f"part-{chunk_index:06d}.complete.json"
        if validate_existing_chunk(
            part_path,
            marker_path,
            chunk_start,
            chunk_end,
            provenance_sha256,
        ):
            chunk_paths.append(part_path)
            print(f"[{chunk_end}/{len(rows)}] verified {part_path.name}", flush=True)
            continue

        chunk_rows = rows[chunk_start:chunk_end]
        predictions = []
        for batch_start in range(0, len(chunk_rows), args.batch_size):
            batch_rows = chunk_rows[batch_start : batch_start + args.batch_size]
            waveforms = []
            metadata = []
            for index, row in enumerate(batch_rows):
                audio_path = Path(row["audio_filepath"]).expanduser().resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Missing calibration audio: {audio_path}")
                waveform = load_audio(audio_path, args.sampling_rate)
                measured_duration = len(waveform) / args.sampling_rate
                if not math.isclose(
                    measured_duration, float(row["duration"]), abs_tol=0.05
                ):
                    raise ValueError(f"Calibration duration mismatch for {row['id']}")
                reference = normalizer(str(row["text"])).strip()
                if not reference:
                    raise ValueError(f"Empty calibration reference for {row['id']}")
                waveforms.append(waveform)
                metadata.append(
                    {
                        "manifest_index": chunk_start + batch_start + index,
                        "id": str(row["id"]),
                        "speaker_id": str(row["speaker_id"]),
                        "language": str(row.get("language") or "sna"),
                        "gender": str(row.get("gender") or ""),
                        "duration": float(row["duration"]),
                        "audio_filepath": str(audio_path),
                        "audio_file_sha256": sha256_file(audio_path),
                        "reference": str(row["text"]),
                        "reference_normalized": reference,
                    }
                )
            batch_predictions = inference_batch(
                model,
                processor,
                normalizer,
                waveforms,
                metadata,
                device,
                dtype,
                args.sampling_rate,
                args.max_length,
                no_speech_token_id,
            )
            for prediction in batch_predictions:
                prediction.update(
                    error_record(
                        prediction["reference_normalized"],
                        prediction["text_normalized"],
                    )
                )
            predictions.extend(batch_predictions)
        atomic_write_jsonl(part_path, predictions)
        marker = {
            "schema_version": 1,
            "chunk_index": chunk_index,
            "start_index": chunk_start,
            "end_index_exclusive": chunk_end,
            "rows": len(predictions),
            "first_id": predictions[0]["id"],
            "last_id": predictions[-1]["id"],
            "part_sha256": sha256_file(part_path),
            "provenance_sha256": provenance_sha256,
        }
        atomic_write_json(marker_path, marker)
        chunk_paths.append(part_path)
        print(f"[{chunk_end}/{len(rows)}] wrote {part_path.name}", flush=True)

    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"Calibration predictions already exist: {predictions_path}")
    all_predictions = []
    for path in chunk_paths:
        all_predictions.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if [row["id"] for row in all_predictions] != ids:
        raise RuntimeError("Calibration predictions do not match manifest order")
    atomic_write_jsonl(predictions_path, all_predictions)

    word_errors = sum(
        row["word_substitutions"]
        + row["word_deletions"]
        + row["word_insertions"]
        for row in all_predictions
    )
    reference_words = sum(row["reference_words"] for row in all_predictions)
    character_errors = sum(
        row["character_substitutions"]
        + row["character_deletions"]
        + row["character_insertions"]
        for row in all_predictions
    )
    reference_characters = sum(
        row["reference_characters"] for row in all_predictions
    )
    wers = sorted(float(row["teacher_wer"]) for row in all_predictions)
    summary = {
        "schema_version": 1,
        "protocol": "rq4-calibration-c0-train-v1",
        "rows": len(all_predictions),
        "speakers": len({row["speaker_id"] for row in all_predictions}),
        "hours": sum(float(row["duration"]) for row in all_predictions) / 3600,
        "corpus_wer": word_errors / reference_words,
        "corpus_cer": character_errors / reference_characters,
        "utterance_wer": {
            "median": percentile(wers, 0.5),
            "p90": percentile(wers, 0.9),
            "above_40_percent": sum(value > 0.4 for value in wers),
            "above_50_percent": sum(value > 0.5 for value in wers),
            "above_80_percent": sum(value > 0.8 for value in wers),
        },
        "empty_outputs": sum(bool(row["empty_text"]) for row in all_predictions),
        "hit_max_length": sum(bool(row["hit_max_length"]) for row in all_predictions),
        "manifest_sha256": sha256_file(manifest_path),
        "teacher_weights_sha256": sha256_file(teacher_dir / "model.safetensors"),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "chunks": len(chunk_paths),
        "provenance_sha256": provenance_sha256,
        "speaker_rows": dict(sorted(Counter(row["speaker_id"] for row in all_predictions).items())),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()