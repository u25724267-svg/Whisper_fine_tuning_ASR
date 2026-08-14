import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


MODEL_ID = "openai/whisper-base"
LANGUAGE = "shona"
DATA_DIR = Path("/home/casper/Speech/data/waxal/sna_asr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Whisper Base pilot on cleaned WAXAL Shona data.")
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output_dir_pilot"))
    parser.add_argument("--dry-run", action="store_true", help="Validate local data and preprocess one example only.")
    return parser.parse_args()


def load_local_data(train_samples: int, eval_samples: int, seed: int) -> DatasetDict:
    data_files = {
        "train": str(DATA_DIR / "train" / "sna_asr_train.normalized.json"),
        "validation": str(DATA_DIR / "validation" / "sna_asr_validation.normalized.json"),
    }
    missing_manifests = [path for path in data_files.values() if not Path(path).is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing manifests: {missing_manifests}")

    dataset = load_dataset("json", data_files=data_files)
    dataset = DatasetDict(
        {
            "train": dataset["train"].shuffle(seed=seed).select(range(min(train_samples, len(dataset["train"])))),
            "validation": dataset["validation"]
            .shuffle(seed=seed)
            .select(range(min(eval_samples, len(dataset["validation"])))),
        }
    )

    for split in dataset:
        missing_audio = [path for path in dataset[split]["audio_filepath"] if not Path(path).is_file()]
        empty_text = [text for text in dataset[split]["text"] if not str(text).strip()]
        if missing_audio or empty_text:
            raise ValueError(
                f"Invalid {split} data: {len(missing_audio)} missing audio files, {len(empty_text)} empty transcripts"
            )
        dataset[split] = dataset[split].cast_column("audio_filepath", Audio(sampling_rate=16_000))

    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    dataset = load_local_data(args.train_samples, args.eval_samples, args.seed)
    print(f"Loaded {len(dataset['train'])} train and {len(dataset['validation'])} validation examples")

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task="transcribe")

    def prepare_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example["audio_filepath"]
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        return {
            "input_features": inputs.input_features[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": processor.tokenizer(example["text"]).input_ids,
        }

    if args.dry_run:
        prepared = prepare_dataset(dataset["train"][0])
        print(
            f"Dry run passed: {len(prepared['input_features'])} mel bins, "
            f"{len(prepared['attention_mask'])} mask frames, {len(prepared['labels'])} label tokens"
        )
        return

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    metric = evaluate.load("wer")

    def compute_metrics(prediction: Any) -> Dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        prediction_text = processor.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        label_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=prediction_text, references=label_text)}

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="no",
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    print(f"Pilot complete. Outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()