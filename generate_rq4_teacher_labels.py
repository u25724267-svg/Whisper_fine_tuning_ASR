import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
import uuid
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from inventory_waxal_unlabeled import sha256_file
from prepare_rq4_admission import canonical_pcm_hash


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path(
    "/ext_data/casper/asr_data/waxal_shona_unlabeled_admission_v1/eligible.jsonl"
)
DEFAULT_ADMISSION_SUMMARY = Path(
    "/ext_data/casper/asr_data/waxal_shona_unlabeled_admission_v1/summary.json"
)
DEFAULT_AUDIT_ROOT = Path("/ext_data/casper/asr_data/waxal_shona_unlabeled_v1")
DEFAULT_TEACHER = Path(
    "/ext_data/casper/asr_experiment_outputs/rq1/c0_random_seed42_v2"
)
DEFAULT_OUTPUT = Path("/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1")
PROTOCOL_PATH = ROOT_DIR / "documents" / "data" / "rq4_pseudolabel_protocol_v1.md"
EXPECTED_MANIFEST_SHA256 = (
    "2305614f4215463e5a93cead01c9e1e58a6cd7dc3efd0fd70c934895968af92a"
)
EXPECTED_MODEL_SHA256 = (
    "a3e4df1999147a9170078cdde23d2c340f85b67f13f650aff7210d47fcec7e9c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate resumable deterministic RQ4 Whisper teacher labels."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--admission-summary", type=Path, default=DEFAULT_ADMISSION_SUMMARY
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=225)
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified-limit", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary_path.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary_path.replace(path)


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def compression_ratio(text: str) -> float | None:
    payload = text.encode("utf-8")
    if not payload:
        return None
    return len(payload) / len(zlib.compress(payload))


def repeated_ngram_fraction(words: Sequence[str], size: int) -> float:
    count = len(words) - size + 1
    if count <= 0:
        return 0.0
    ngrams = [tuple(words[index : index + size]) for index in range(count)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def maximum_word_run(words: Sequence[str]) -> int:
    maximum = 0
    current = 0
    previous = None
    for word in words:
        if word == previous:
            current += 1
        else:
            previous = word
            current = 1
        maximum = max(maximum, current)
    return maximum


def select_duration_stratified_rows(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if not 1 <= count <= len(rows):
        raise ValueError("Stratified count must be between one and the row count")
    indexed_rows = list(enumerate(rows))
    ordered = sorted(
        indexed_rows,
        key=lambda item: (float(item[1]["duration"]), str(item[1]["id"])),
    )
    if count == 1:
        selected_positions = [len(ordered) // 2]
    else:
        selected_positions = [
            round(index * (len(ordered) - 1) / (count - 1))
            for index in range(count)
        ]
    selected = [ordered[position] for position in selected_positions]
    selected.sort(key=lambda item: item[0])
    return [
        {**row, "_admission_manifest_index": original_index}
        for original_index, row in selected
    ]


def text_diagnostics(text: str, normalized_text: str, duration: float) -> dict[str, Any]:
    words = normalized_text.split()
    characters = len(normalized_text.replace(" ", ""))
    non_letter_characters = sum(
        not character.isalpha() and not character.isspace() for character in text
    )
    return {
        "empty_text": not bool(normalized_text),
        "utf8_compression_ratio": compression_ratio(text),
        "word_count": len(words),
        "character_count": characters,
        "words_per_second": len(words) / duration,
        "characters_per_second": characters / duration,
        "unique_word_ratio": len(set(words)) / len(words) if words else None,
        "maximum_consecutive_word_run": maximum_word_run(words),
        "repeated_bigram_fraction": repeated_ngram_fraction(words, 2),
        "repeated_trigram_fraction": repeated_ngram_fraction(words, 3),
        "non_letter_fraction_raw": (
            non_letter_characters / len(text) if text else None
        ),
    }


class ParquetAudioReader:
    def __init__(self, audit_root: Path, sampling_rate: int) -> None:
        self.audit_root = audit_root
        self.sampling_rate = sampling_rate
        self.current_path: Path | None = None
        self.current_table: Any = None
        self.verified_paths: set[Path] = set()

    def _load_shard(self, manifest_row: dict[str, Any]) -> None:
        shard_index = int(manifest_row["source_shard_index"])
        parquet_path = Path(manifest_row["source_parquet"]).expanduser().resolve()
        report_path = (
            self.audit_root / "shard_audits" / f"shard-{shard_index:05d}.json"
        )
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing source audit: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(report["parquet_path"]).resolve() != parquet_path:
            raise ValueError(f"Source path changed for shard {shard_index}")
        if parquet_path not in self.verified_paths:
            if sha256_file(parquet_path) != report["parquet_sha256"]:
                raise ValueError(f"Source Parquet hash mismatch: {parquet_path}")
            self.verified_paths.add(parquet_path)
        self.current_table = pq.read_table(
            parquet_path, columns=["id", "speaker_id", "audio"]
        )
        self.current_path = parquet_path

    def read(self, manifest_row: dict[str, Any]) -> np.ndarray:
        parquet_path = Path(manifest_row["source_parquet"]).expanduser().resolve()
        if self.current_path != parquet_path:
            self._load_shard(manifest_row)
        row_index = int(manifest_row["source_row_index"])
        if not 0 <= row_index < self.current_table.num_rows:
            raise IndexError(f"Source row index out of range: {row_index}")
        source_row = self.current_table.slice(row_index, 1).to_pylist()[0]
        if str(source_row.get("id") or "") != str(manifest_row["id"]):
            raise ValueError(f"Source ID mismatch for {manifest_row['id']}")
        if str(source_row.get("speaker_id") or "") != str(
            manifest_row["speaker_id"]
        ):
            raise ValueError(f"Source speaker mismatch for {manifest_row['id']}")
        audio = source_row.get("audio")
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if audio_bytes is None:
            raise ValueError(f"Missing embedded audio for {manifest_row['id']}")
        decoded = canonical_pcm_hash(audio_bytes)
        if decoded["pcm_sha256"] != manifest_row["pcm_sha256"]:
            raise ValueError(f"PCM hash mismatch for {manifest_row['id']}")
        if decoded["channels"] != 1:
            raise ValueError(f"Non-mono admitted audio for {manifest_row['id']}")
        if not math.isclose(
            decoded["duration"], float(manifest_row["duration"]), abs_tol=1e-6
        ):
            raise ValueError(f"Duration mismatch for {manifest_row['id']}")
        waveform, source_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
        waveform = waveform[:, 0]
        if source_rate != self.sampling_rate:
            waveform = librosa.resample(
                waveform, orig_sr=source_rate, target_sr=self.sampling_rate
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid decoded waveform for {manifest_row['id']}")
        return waveform


def validate_existing_chunk(
    part_path: Path,
    marker_path: Path,
    expected_start: int,
    expected_end: int,
    provenance_sha256: str,
) -> bool:
    if not part_path.exists() and not marker_path.exists():
        return False
    if not part_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"Incomplete chunk artifact: {part_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_count = expected_end - expected_start
    if marker.get("start_index") != expected_start:
        raise ValueError(f"Chunk start mismatch: {marker_path}")
    if marker.get("end_index_exclusive") != expected_end:
        raise ValueError(f"Chunk end mismatch: {marker_path}")
    if marker.get("rows") != expected_count:
        raise ValueError(f"Chunk count mismatch: {marker_path}")
    if marker.get("provenance_sha256") != provenance_sha256:
        raise ValueError(f"Chunk provenance mismatch: {marker_path}")
    if marker.get("part_sha256") != sha256_file(part_path):
        raise ValueError(f"Chunk hash mismatch: {part_path}")
    with part_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) != expected_count:
        raise ValueError(f"Chunk JSONL row mismatch: {part_path}")
    if [row["manifest_index"] for row in rows] != list(
        range(expected_start, expected_end)
    ):
        raise ValueError(f"Chunk manifest indices are not contiguous: {part_path}")
    return True


def token_statistics(
    generated_ids: Sequence[int],
    transition_scores: Sequence[float],
    special_ids: set[int],
    eos_token_id: int,
) -> dict[str, Any]:
    if len(generated_ids) != len(transition_scores):
        raise ValueError("Generated IDs and transition scores must align")
    eos_positions = [
        index for index, token_id in enumerate(generated_ids) if token_id == eos_token_id
    ]
    eos_position = eos_positions[0] if eos_positions else None
    retained_length = eos_position + 1 if eos_position is not None else len(generated_ids)
    generated_ids = generated_ids[:retained_length]
    transition_scores = transition_scores[:retained_length]
    content_log_probabilities = [
        float(score)
        for token_id, score in zip(generated_ids, transition_scores)
        if int(token_id) not in special_ids
    ]
    content_probabilities = sorted(math.exp(value) for value in content_log_probabilities)
    eos_log_probability = (
        float(transition_scores[eos_position]) if eos_position is not None else None
    )
    mean_log_probability = (
        sum(content_log_probabilities) / len(content_log_probabilities)
        if content_log_probabilities
        else None
    )
    return {
        "generated_token_ids": [int(value) for value in generated_ids],
        "generated_token_log_probabilities": [
            float(value) for value in transition_scores
        ],
        "content_token_count": len(content_log_probabilities),
        "content_token_log_probability_sum": (
            sum(content_log_probabilities) if content_log_probabilities else None
        ),
        "content_token_log_probability_mean": mean_log_probability,
        "geometric_content_token_confidence": (
            math.exp(mean_log_probability)
            if mean_log_probability is not None
            else None
        ),
        "minimum_content_token_probability": (
            content_probabilities[0] if content_probabilities else None
        ),
        "p10_content_token_probability": (
            percentile(content_probabilities, 0.10)
            if content_probabilities
            else None
        ),
        "eos_generated": eos_position is not None,
        "eos_generated_position": eos_position,
        "eos_log_probability": eos_log_probability,
    }


def inference_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    normalizer: BasicTextNormalizer,
    waveforms: Sequence[np.ndarray],
    metadata: Sequence[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    sampling_rate: int,
    max_length: int,
    no_speech_token_id: int,
) -> list[dict[str, Any]]:
    features = processor.feature_extractor(
        list(waveforms),
        sampling_rate=sampling_rate,
        padding="max_length",
        max_length=30 * sampling_rate,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = features.input_features.to(device=device, dtype=dtype)
    attention_mask = features.attention_mask.to(device=device)
    decoder_start_token_id = model.generation_config.decoder_start_token_id
    if decoder_start_token_id is None:
        raise ValueError("Teacher has no decoder start token ID")

    with torch.inference_mode():
        encoder_outputs = model.model.encoder(
            input_features, attention_mask=attention_mask, return_dict=True
        )
        decoder_input_ids = torch.full(
            (len(waveforms), 1),
            int(decoder_start_token_id),
            dtype=torch.long,
            device=device,
        )
        initial_outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        no_speech_probabilities = torch.softmax(
            initial_outputs.logits[:, 0, :].float(), dim=-1
        )[:, no_speech_token_id]
        generated = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            language="shona",
            task="transcribe",
            do_sample=False,
            num_beams=1,
            return_timestamps=False,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
        transition_scores = model.compute_transition_scores(
            generated.sequences, generated.scores, normalize_logits=True
        )

    score_steps = len(generated.scores)
    if score_steps <= 0:
        raise RuntimeError("Teacher generation returned no score steps")
    generated_ids = generated.sequences[:, -score_steps:]
    if generated_ids.shape != transition_scores.shape:
        raise RuntimeError("Generated token and transition-score shapes differ")
    texts = processor.tokenizer.batch_decode(
        generated.sequences, skip_special_tokens=True
    )
    special_ids = {int(value) for value in processor.tokenizer.all_special_ids}
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Teacher tokenizer has no EOS token ID")

    results = []
    for index, item_metadata in enumerate(metadata):
        raw_text = texts[index].strip()
        normalized_text = normalizer(raw_text).strip()
        item_token_statistics = token_statistics(
            generated_ids[index].detach().cpu().tolist(),
            transition_scores[index].detach().float().cpu().tolist(),
            special_ids,
            int(eos_token_id),
        )
        results.append(
            {
                **item_metadata,
                "text_raw": raw_text,
                "text_normalized": normalized_text,
                **item_token_statistics,
                "raw_no_speech_token_probability": float(
                    no_speech_probabilities[index].detach().cpu()
                ),
                "hit_max_length": not item_token_statistics["eos_generated"],
                **text_diagnostics(
                    raw_text, normalized_text, float(item_metadata["duration"])
                ),
            }
        )
    return results


def build_provenance(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    admission_summary = args.admission_summary.expanduser().resolve()
    audit_root = args.audit_root.expanduser().resolve()
    teacher_dir = args.teacher_dir.expanduser().resolve()
    required_teacher_files = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    paths = {
        "manifest": manifest,
        "admission_summary": admission_summary,
        "source_audit": audit_root / "inventory_full.json",
        "protocol": PROTOCOL_PATH,
        "script": Path(__file__).resolve(),
        **{
            f"teacher_{name}": teacher_dir / name for name in required_teacher_files
        },
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RQ4 inference prerequisites: {missing}")
    if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Eligible manifest does not match the frozen SHA-256")
    if sha256_file(teacher_dir / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("Teacher weights do not match the frozen SHA-256")
    summary = json.loads(admission_summary.read_text(encoding="utf-8"))
    if summary.get("manifests", {}).get("eligible", {}).get("sha256") != (
        EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError("Admission summary does not identify the frozen manifest")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("librosa", "numpy", "pyarrow", "soundfile", "torch", "transformers")
    }
    return {
        "schema_version": 1,
        "protocol": "rq4-pseudolabel-v1",
        "teacher": "rq1-c0-random-seed42-v2",
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "parameters": {
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "max_length": args.max_length,
            "sampling_rate": args.sampling_rate,
            "seed": args.seed,
            "limit": args.limit,
            "stratified_limit": args.stratified_limit,
            "device": args.device,
            "decoding": {
                "language": "shona",
                "task": "transcribe",
                "do_sample": False,
                "num_beams": 1,
                "return_timestamps": False,
            },
        },
        "packages": packages,
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
    if args.max_length <= 4:
        raise ValueError("Maximum generation length is too small")
    if args.sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive")
    if args.stratified_limit is not None and args.stratified_limit <= 0:
        raise ValueError("Stratified limit must be positive")
    if args.limit is not None and args.stratified_limit is not None:
        raise ValueError("Use only one of --limit and --stratified-limit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")

    provenance = build_provenance(args)
    provenance_sha256 = canonical_json_sha256(provenance)
    output_dir = args.output_dir.expanduser().resolve()
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "provenance.json"
    if provenance_path.exists():
        existing_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(existing_provenance) != provenance_sha256:
            raise ValueError("Existing output provenance differs from this invocation")
    else:
        atomic_write_json(provenance_path, provenance)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"Teacher inference already complete: {output_dir}")

    manifest_path = args.manifest.expanduser().resolve()
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_rows = [
        {**row, "_admission_manifest_index": index}
        for index, row in enumerate(manifest_rows)
    ]
    if args.stratified_limit is not None:
        manifest_rows = select_duration_stratified_rows(
            manifest_rows, args.stratified_limit
        )
    elif args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    utterance_ids = [str(row["id"]) for row in manifest_rows]
    if not manifest_rows or len(utterance_ids) != len(set(utterance_ids)):
        raise ValueError("Inference manifest must contain unique non-empty rows")

    device = torch.device(args.device)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    teacher_dir = args.teacher_dir.expanduser().resolve()
    processor = WhisperProcessor.from_pretrained(teacher_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        teacher_dir, torch_dtype=dtype
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    no_timestamps_token_id = model.generation_config.no_timestamps_token_id
    if no_timestamps_token_id is None:
        raise ValueError("Teacher generation config has no no-timestamps token ID")
    no_speech_token_id = int(no_timestamps_token_id) - 1
    if not 0 <= no_speech_token_id < model.config.vocab_size:
        raise ValueError("Derived no-speech token ID is outside the teacher vocabulary")
    normalizer = BasicTextNormalizer(remove_diacritics=False)
    reader = ParquetAudioReader(
        args.audit_root.expanduser().resolve(), args.sampling_rate
    )

    chunk_paths = []
    total_rows = len(manifest_rows)
    for chunk_start in range(0, total_rows, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, total_rows)
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
            print(
                f"[{chunk_end}/{total_rows}] verified existing {part_path.name}",
                flush=True,
            )
            chunk_paths.append(part_path)
            continue

        chunk_rows = manifest_rows[chunk_start:chunk_end]
        predictions = []
        for batch_start in range(0, len(chunk_rows), args.batch_size):
            batch_rows = chunk_rows[batch_start : batch_start + args.batch_size]
            waveforms = [reader.read(row) for row in batch_rows]
            batch_metadata = [
                {
                    "manifest_index": chunk_start + batch_start + index,
                    "admission_manifest_index": int(
                        row["_admission_manifest_index"]
                    ),
                    "id": str(row["id"]),
                    "speaker_id": str(row["speaker_id"]),
                    "language": str(row["language"]),
                    "gender": str(row["gender"]),
                    "duration": float(row["duration"]),
                    "pcm_sha256": str(row["pcm_sha256"]),
                    "source_shard_index": int(row["source_shard_index"]),
                    "source_row_index": int(row["source_row_index"]),
                    "source_parquet": str(row["source_parquet"]),
                }
                for index, row in enumerate(batch_rows)
            ]
            predictions.extend(
                inference_batch(
                    model,
                    processor,
                    normalizer,
                    waveforms,
                    batch_metadata,
                    device,
                    dtype,
                    args.sampling_rate,
                    args.max_length,
                    int(no_speech_token_id),
                )
            )
        if [row["id"] for row in predictions] != [
            str(row["id"]) for row in chunk_rows
        ]:
            raise RuntimeError("Prediction order differs from manifest order")
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
        print(f"[{chunk_end}/{total_rows}] wrote {part_path.name}", flush=True)

    prediction_path = output_dir / "predictions.jsonl"
    if prediction_path.exists():
        raise FileExistsError(f"Consolidated predictions already exist: {prediction_path}")
    all_predictions = []
    for part_path in chunk_paths:
        with part_path.open(encoding="utf-8") as source:
            all_predictions.extend(json.loads(line) for line in source if line.strip())
    if len(all_predictions) != total_rows:
        raise RuntimeError("Consolidated prediction count does not match manifest")
    if [row["id"] for row in all_predictions] != utterance_ids:
        raise RuntimeError("Consolidated prediction IDs do not match manifest order")
    atomic_write_jsonl(prediction_path, all_predictions)

    confidences = sorted(
        float(row["geometric_content_token_confidence"])
        for row in all_predictions
        if row["geometric_content_token_confidence"] is not None
    )
    no_speech_probabilities = sorted(
        float(row["raw_no_speech_token_probability"]) for row in all_predictions
    )
    speakers = Counter(str(row["speaker_id"]) for row in all_predictions)
    resource_usage = None
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        peak_reserved = torch.cuda.max_memory_reserved(device)
        resource_usage = {
            "device": torch.cuda.get_device_name(device),
            "total_memory_bytes": total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": peak_reserved,
            "reserved_headroom_fraction": 1 - peak_reserved / total_memory,
        }
    summary = {
        "schema_version": 1,
        "protocol": "rq4-pseudolabel-v1",
        "rows": total_rows,
        "hours": sum(float(row["duration"]) for row in all_predictions) / 3600,
        "speakers": len(speakers),
        "empty_outputs": sum(bool(row["empty_text"]) for row in all_predictions),
        "hit_max_length": sum(bool(row["hit_max_length"]) for row in all_predictions),
        "confidence": {
            "nonempty_rows": len(confidences),
            "p05": percentile(confidences, 0.05) if confidences else None,
            "median": percentile(confidences, 0.5) if confidences else None,
            "p95": percentile(confidences, 0.95) if confidences else None,
        },
        "raw_no_speech_token_probability": {
            "p05": percentile(no_speech_probabilities, 0.05),
            "median": percentile(no_speech_probabilities, 0.5),
            "p95": percentile(no_speech_probabilities, 0.95),
            "calibrated": False,
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "teacher_dir": str(teacher_dir),
        "teacher_weights_sha256": sha256_file(teacher_dir / "model.safetensors"),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "chunks": len(chunk_paths),
        "provenance_sha256": provenance_sha256,
        "resource_usage": resource_usage,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()