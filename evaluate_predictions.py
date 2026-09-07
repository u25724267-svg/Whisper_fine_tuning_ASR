import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import jiwer
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from train_full import DataCollatorSpeechSeq2SeqWithPadding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export item-level predictions and ASR errors for a trained Whisper run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Prediction artifact directory; defaults to MODEL_DIR/item_predictions.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=["validation", "test"],
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Evaluate an arbitrary named manifest. Repeat for multiple manifests. "
            "When supplied without --splits, only named manifests are evaluated."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    word_output = jiwer.process_words(references, predictions)
    character_output = jiwer.process_characters(references, predictions)
    return {
        "wer": 100 * word_output.wer,
        "cer": 100 * character_output.cer,
        "substitutions": word_output.substitutions,
        "deletions": word_output.deletions,
        "insertions": word_output.insertions,
        "hits": word_output.hits,
        "reference_words": word_output.substitutions
        + word_output.deletions
        + word_output.hits,
    }


def evaluation_manifests(
    split_names: List[str] | None,
    manifest_arguments: List[str],
    data_config: Dict[str, Any],
) -> List[tuple[str, Path]]:
    selected_splits = (
        split_names
        if split_names is not None
        else ([] if manifest_arguments else ["validation", "test"])
    )
    manifests = [
        (split_name, Path(data_config[f"{split_name}_manifest"]).expanduser().resolve())
        for split_name in selected_splits
    ]
    for argument in manifest_arguments:
        if "=" not in argument:
            raise ValueError(f"Manifest must use NAME=PATH syntax: {argument}")
        name, value = argument.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid manifest name: {name}")
        manifests.append((name, Path(value).expanduser().resolve()))
    names = [name for name, _ in manifests]
    if len(names) != len(set(names)):
        raise ValueError("Evaluation manifest names must be unique")
    for name, path in manifests:
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation manifest '{name}': {path}")
    if not manifests:
        raise ValueError("At least one split or named manifest is required")
    return manifests


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing trained model: {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    manifests = evaluation_manifests(
        args.splits, args.manifest, data_config
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else model_dir / "item_predictions"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Prediction output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        processor = WhisperProcessor.from_pretrained(
            model_dir,
            language=model_config["language"],
            task=model_config["task"],
        )
    except OSError:
        processor = WhisperProcessor.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            language=model_config["language"],
            task=model_config["task"],
        )
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.generation_config.language = model_config["language"]
    model.generation_config.task = model_config["task"]
    if model_config.get("clear_forced_decoder_ids", False):
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
    if model_config.get("clear_suppress_tokens", False):
        model.config.suppress_tokens = []
        model.generation_config.suppress_tokens = []

    evaluation_args = Seq2SeqTrainingArguments(
        output_dir=str(model_dir / "prediction_evaluation"),
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        fp16=training_config["fp16"],
        predict_with_generate=True,
        generation_max_length=training_config["generation_max_length"],
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=evaluation_args,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        processing_class=processor.feature_extractor,
    )

    def prepare_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example[data_config["audio_column"]]
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        return {
            "input_features": inputs.input_features[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": processor.tokenizer(example[data_config["text_column"]]).input_ids,
        }

    summary = {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id", config["experiment_name"]),
        "model_dir": str(model_dir),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "splits": {},
    }

    for split_name, manifest_path in manifests:
        dataset = load_dataset(
            "json", data_files={split_name: str(manifest_path)}
        )[split_name]
        dataset = dataset.cast_column(
            data_config["audio_column"], Audio(sampling_rate=data_config["sampling_rate"])
        )
        metadata = [
            {
                "id": row.get("id"),
                "speaker_id": row.get("speaker_id"),
                "duration": row.get("duration"),
                "original_split": row.get("original_split"),
            }
            for row in dataset
        ]
        prepared_dataset = dataset.map(
            prepare_dataset,
            remove_columns=dataset.column_names,
            num_proc=1,
            keep_in_memory=data_config.get("keep_preprocessed_in_memory", False),
            desc=f"Extracting {split_name} features",
        )
        prediction_output = trainer.predict(
            prepared_dataset, metric_key_prefix=f"item_{split_name}"
        )
        prediction_ids = prediction_output.predictions
        label_ids = prediction_output.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        prediction_path = output_dir / f"{split_name}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as destination:
            for item_metadata, reference, prediction in zip(
                metadata, references, predictions
            ):
                item_errors = error_metrics([reference], [prediction])
                destination.write(
                    json.dumps(
                        {
                            **item_metadata,
                            "reference": reference,
                            "prediction": prediction,
                            **item_errors,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        summary["splits"][split_name] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "rows": len(references),
            **error_metrics(references, predictions),
            "predictions": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()