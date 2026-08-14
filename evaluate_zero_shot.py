import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import jiwer
import torch
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from train_full import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_config,
    load_environment,
    load_local_data,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained Whisper checkpoint without fine-tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/whisper-medium-shona-3epochs-eval1000.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def compute_wer(references: List[str], predictions: List[str]) -> float:
    return 100 * jiwer.wer(references, predictions)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    evaluation_wandb = {
        **config["wandb"],
        "run_name": f"{config['experiment_name']}-zero-shot",
        "job_type": "evaluation",
        "tags": [*config["wandb"].get("tags", []), "zero-shot"],
        "resume_run_id": None,
    }
    output_dir = resolve_path(args.output_dir or f"zero_shot_{config['experiment_name']}")

    load_environment(require_api_key=not args.dry_run, wandb_config=evaluation_wandb)
    dataset = load_local_data(data_config)
    train_speakers = set(dataset["train"]["speaker_id"])
    test_speaker_ids = dataset["test"]["speaker_id"]
    unseen_mask = [speaker_id not in train_speakers for speaker_id in test_speaker_ids]
    unseen_count = sum(unseen_mask)

    processor = WhisperProcessor.from_pretrained(
        model_config["id"], language=model_config["language"], task=model_config["task"]
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

    if args.dry_run:
        prepared = prepare_dataset(dataset["test"][0])
        print(
            f"Zero-shot dry run passed: model {model_config['id']}, {len(dataset['test'])} test rows, "
            f"{unseen_count} unseen-speaker rows, {len(prepared['input_features'])} mel bins"
        )
        return

    wandb.login()
    run = wandb.init(
        project=evaluation_wandb["project"],
        name=evaluation_wandb["run_name"],
        job_type=evaluation_wandb["job_type"],
        tags=evaluation_wandb["tags"],
        config={"source_experiment": config, "evaluation": "zero-shot"},
    )

    prepared_test = dataset["test"].map(
        prepare_dataset,
        remove_columns=dataset["test"].column_names,
        keep_in_memory=True,
        num_proc=1,
        desc="Extracting zero-shot test features",
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_config["id"])
    if model_config.get("clear_forced_decoder_ids", False):
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
    if model_config.get("clear_suppress_tokens", False):
        model.config.suppress_tokens = []
        model.generation_config.suppress_tokens = []
    model.generation_config.language = model_config["language"]
    model.generation_config.task = model_config["task"]

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        fp16=training_config["fp16"],
        predict_with_generate=True,
        generation_max_length=training_config["generation_max_length"],
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        processing_class=processor.feature_extractor,
    )
    prediction_output = trainer.predict(prepared_test, metric_key_prefix="zero_shot")

    prediction_ids = prediction_output.predictions
    label_ids = prediction_output.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    predictions = processor.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
    references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(text) for text in predictions]
    normalized_references = [normalizer(text) for text in references]
    unseen_predictions = [text for text, unseen in zip(predictions, unseen_mask) if unseen]
    unseen_references = [text for text, unseen in zip(references, unseen_mask) if unseen]
    unseen_normalized_predictions = [text for text, unseen in zip(normalized_predictions, unseen_mask) if unseen]
    unseen_normalized_references = [text for text, unseen in zip(normalized_references, unseen_mask) if unseen]

    metrics = {
        "model_id": model_config["id"],
        "test_rows": len(references),
        "unseen_speaker_rows": unseen_count,
        "test_wer_raw": compute_wer(references, predictions),
        "test_wer_normalized": compute_wer(normalized_references, normalized_predictions),
        "unseen_speaker_wer_raw": compute_wer(unseen_references, unseen_predictions),
        "unseen_speaker_wer_normalized": compute_wer(
            unseen_normalized_references, unseen_normalized_predictions
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file:
        for row_id, speaker_id, unseen, reference, prediction in zip(
            dataset["test"]["id"], test_speaker_ids, unseen_mask, references, predictions
        ):
            predictions_file.write(
                json.dumps(
                    {
                        "id": row_id,
                        "speaker_id": speaker_id,
                        "speaker_seen_in_train": not unseen,
                        "reference": reference,
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    run.log(metrics)
    run.summary.update(metrics)
    wandb.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()