import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from generate_rq4_teacher_labels import (
    ParquetAudioReader,
    atomic_write_json,
    atomic_write_jsonl,
)
from inventory_waxal_unlabeled import sha256_file


ROOT_DIR = Path(__file__).resolve().parent
MODEL_ID = "facebook/seamless-m4t-v2-large"
MODEL_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
DEFAULT_INPUT = Path(
    "/ext_data/casper/asr_data/rq4_calibration_c0_train_v1/predictions.jsonl"
)
DEFAULT_OUTPUT = Path("/ext_data/casper/asr_data/rq4_proxy_calibration_v1")
AUDIT_ROOT = Path("/ext_data/casper/asr_data/waxal_shona_unlabeled_v1")
REVIEW_PATH = ROOT_DIR / "documents" / "data" / "rq4_seamless_proxy_review_v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate license-gated SeamlessM4T-v2 Shona proxy predictions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified-calibration-limit", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--confirm-noncommercial-license", action="store_true")
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_character_disagreement(first: str, second: str) -> float:
    if not first or not second:
        return 0.0 if first == second else 1.0
    output = jiwer.process_characters(first, second)
    edits = output.substitutions + output.deletions + output.insertions
    return edits / max(1, len(first), len(second))


def select_stratified_calibration_rows(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count < 4 or count % 4:
        raise ValueError("Stratified calibration count must be divisible by four")
    bins = (
        ("wer_le_40", lambda value: value <= 0.4),
        ("wer_40_to_50", lambda value: 0.4 < value <= 0.5),
        ("wer_50_to_80", lambda value: 0.5 < value <= 0.8),
        ("wer_gt_80", lambda value: value > 0.8),
    )
    per_bin = count // len(bins)
    selected = []
    for name, predicate in bins:
        candidates = [
            (index, row)
            for index, row in enumerate(rows)
            if predicate(float(row["teacher_wer"]))
        ]
        candidates.sort(
            key=lambda item: (
                float(item[1]["duration"]),
                str(item[1]["speaker_id"]),
                str(item[1]["id"]),
            )
        )
        if len(candidates) < per_bin:
            raise ValueError(f"Not enough rows in pilot stratum {name}")
        positions = [
            round(index * (len(candidates) - 1) / (per_bin - 1))
            if per_bin > 1
            else len(candidates) // 2
            for index in range(per_bin)
        ]
        selected.extend(
            (original_index, {**row, "proxy_pilot_stratum": name})
            for original_index, row in (candidates[position] for position in positions)
        )
    selected.sort(key=lambda item: item[0])
    return [
        {**row, "proxy_input_index": original_index}
        for original_index, row in selected
    ]


def load_audio(path: Path, sampling_rate: int) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sampling_rate:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=sampling_rate
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid proxy audio: {path}")
    return waveform


def proxy_chunk_valid(
    part_path: Path,
    marker_path: Path,
    start: int,
    end: int,
    provenance_sha256: str,
) -> bool:
    if not part_path.exists() and not marker_path.exists():
        return False
    if not part_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"Incomplete proxy chunk: {part_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("start_index") != start or marker.get("end_index_exclusive") != end:
        raise ValueError(f"Proxy chunk bounds differ: {marker_path}")
    if marker.get("rows") != end - start:
        raise ValueError(f"Proxy chunk count differs: {marker_path}")
    if marker.get("provenance_sha256") != provenance_sha256:
        raise ValueError(f"Proxy chunk provenance differs: {marker_path}")
    if marker.get("part_sha256") != sha256_file(part_path):
        raise ValueError(f"Proxy chunk hash differs: {part_path}")
    rows = [
        json.loads(line)
        for line in part_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row["proxy_index"] for row in rows] != list(range(start, end)):
        raise ValueError(f"Proxy chunk indices differ: {part_path}")
    return True


def trim_token_ids(token_ids: Sequence[int], eos_token_id: int, pad_token_id: int) -> list[int]:
    retained = []
    for index, token_id in enumerate(token_ids):
        value = int(token_id)
        if value == pad_token_id and index > 0:
            break
        retained.append(value)
        if value == eos_token_id and index > 0:
            break
    return retained


def inference_batch(
    model: SeamlessM4Tv2ForSpeechToText,
    processor: Any,
    waveforms: Sequence[np.ndarray],
    rows: Sequence[dict[str, Any]],
    start_index: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling_rate: int,
    max_new_tokens: int,
    normalizer: BasicTextNormalizer,
) -> list[dict[str, Any]]:
    inputs = processor(
        audios=list(waveforms),
        sampling_rate=sampling_rate,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        name: tensor.to(
            device=device,
            dtype=dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            tgt_lang="sna",
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
        )
    texts = processor.batch_decode(generated.detach().cpu(), skip_special_tokens=True)
    eos_token_id = int(model.generation_config.eos_token_id)
    pad_token_id = int(model.generation_config.pad_token_id)
    results = []
    for index, (row, raw_text, tokens) in enumerate(zip(rows, texts, generated)):
        proxy_text = raw_text.strip()
        proxy_normalized = normalizer(proxy_text).strip()
        teacher_normalized = str(row["text_normalized"])
        retained_tokens = trim_token_ids(
            tokens.detach().cpu().tolist(), eos_token_id, pad_token_id
        )
        results.append(
            {
                **row,
                "proxy_index": start_index + index,
                "proxy_text_raw": proxy_text,
                "proxy_text_normalized": proxy_normalized,
                "proxy_token_ids": retained_tokens,
                "proxy_empty_text": not bool(proxy_normalized),
                "proxy_hit_max_length": eos_token_id not in retained_tokens,
                "teacher_proxy_character_disagreement": (
                    normalized_character_disagreement(
                        teacher_normalized, proxy_normalized
                    )
                ),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if not args.confirm_noncommercial_license:
        raise PermissionError(
            "Proxy inference requires --confirm-noncommercial-license"
        )
    if args.model_id != MODEL_ID or args.revision != MODEL_REVISION:
        raise ValueError("Proxy model ID and revision must match the frozen protocol")
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be positive")
    if args.chunk_size % args.batch_size != 0:
        raise ValueError("Chunk size must be divisible by batch size")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive")
    if (
        args.stratified_calibration_limit is not None
        and args.stratified_calibration_limit <= 0
    ):
        raise ValueError("Stratified calibration limit must be positive")
    if args.limit is not None and args.stratified_calibration_limit is not None:
        raise ValueError("Use only one of --limit and --stratified-calibration-limit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA proxy inference requested but unavailable")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    chunks_dir = output_dir / "chunks"
    if not input_path.is_file() or not REVIEW_PATH.is_file():
        raise FileNotFoundError("Missing proxy input or frozen proxy review")
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.stratified_calibration_limit is not None:
        rows = select_stratified_calibration_rows(
            rows, args.stratified_calibration_limit
        )
    elif args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Proxy input requires unique non-empty IDs")

    provenance = {
        "schema_version": 1,
        "protocol": "rq4-seamless-proxy-v1",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "tgt_lang": "sna",
        "license": "CC-BY-NC-4.0",
        "noncommercial_license_confirmed": True,
        "review": str(REVIEW_PATH),
        "review_sha256": sha256_file(REVIEW_PATH),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "parameters": {
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "max_new_tokens": args.max_new_tokens,
            "sampling_rate": args.sampling_rate,
            "seed": args.seed,
            "limit": args.limit,
            "stratified_calibration_limit": args.stratified_calibration_limit,
            "device": args.device,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("sentencepiece", "torch", "transformers")
        },
        "runtime": {"python": sys.version, "torch": torch.__version__},
    }
    provenance_sha256 = canonical_json_sha256(provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(existing) != provenance_sha256:
            raise ValueError("Existing proxy provenance differs")
    else:
        atomic_write_json(provenance_path, provenance)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"Proxy inference already complete: {output_dir}")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    language_token_id = processor.tokenizer.convert_tokens_to_ids("__sna__")
    if language_token_id != 256078:
        raise ValueError("Pinned processor Shona token ID differs from review")
    normalizer = BasicTextNormalizer(remove_diacritics=False)
    parquet_reader = ParquetAudioReader(AUDIT_ROOT, args.sampling_rate)

    chunk_paths = []
    for chunk_start in range(0, len(rows), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(rows))
        chunk_index = chunk_start // args.chunk_size
        part_path = chunks_dir / f"part-{chunk_index:06d}.jsonl"
        marker_path = chunks_dir / f"part-{chunk_index:06d}.complete.json"
        if proxy_chunk_valid(
            part_path, marker_path, chunk_start, chunk_end, provenance_sha256
        ):
            chunk_paths.append(part_path)
            print(f"[{chunk_end}/{len(rows)}] verified {part_path.name}", flush=True)
            continue
        chunk_rows = rows[chunk_start:chunk_end]
        predictions = []
        for batch_start in range(0, len(chunk_rows), args.batch_size):
            batch_rows = chunk_rows[batch_start : batch_start + args.batch_size]
            waveforms = []
            for row in batch_rows:
                if row.get("audio_filepath"):
                    waveforms.append(
                        load_audio(
                            Path(row["audio_filepath"]).expanduser().resolve(),
                            args.sampling_rate,
                        )
                    )
                else:
                    waveforms.append(parquet_reader.read(row))
            predictions.extend(
                inference_batch(
                    model,
                    processor,
                    waveforms,
                    batch_rows,
                    chunk_start + batch_start,
                    device,
                    dtype,
                    args.sampling_rate,
                    args.max_new_tokens,
                    normalizer,
                )
            )
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
        raise FileExistsError(f"Proxy predictions already exist: {predictions_path}")
    consolidated = []
    for path in chunk_paths:
        consolidated.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if [row["id"] for row in consolidated] != ids:
        raise RuntimeError("Proxy predictions do not match input order")
    atomic_write_jsonl(predictions_path, consolidated)

    total_memory = (
        torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    disagreements = sorted(
        float(row["teacher_proxy_character_disagreement"]) for row in consolidated
    )
    summary = {
        "schema_version": 1,
        "protocol": "rq4-seamless-proxy-v1",
        "rows": len(consolidated),
        "hours": sum(float(row["duration"]) for row in consolidated) / 3600,
        "speakers": len({str(row["speaker_id"]) for row in consolidated}),
        "proxy_empty_outputs": sum(bool(row["proxy_empty_text"]) for row in consolidated),
        "proxy_hit_max_length": sum(
            bool(row["proxy_hit_max_length"]) for row in consolidated
        ),
        "character_disagreement": {
            "p05": float(np.quantile(disagreements, 0.05)),
            "median": float(np.quantile(disagreements, 0.5)),
            "p95": float(np.quantile(disagreements, 0.95)),
        },
        "model_id": args.model_id,
        "model_revision": args.revision,
        "license": "CC-BY-NC-4.0",
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "chunks": len(chunk_paths),
        "provenance_sha256": provenance_sha256,
        "resource_usage": {
            "total_memory_bytes": total_memory,
            "peak_reserved_bytes": peak_reserved,
            "reserved_headroom_fraction": (
                1 - peak_reserved / total_memory
                if peak_reserved is not None and total_memory is not None
                else None
            ),
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()