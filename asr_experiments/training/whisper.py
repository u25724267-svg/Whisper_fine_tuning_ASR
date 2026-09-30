"""Shared Whisper construction, preprocessing, and metric callbacks."""

from typing import Any, Callable, Mapping

from pathlib import Path

from transformers import (
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def load_processor(model_config: Mapping[str, Any]) -> WhisperProcessor:
    """Load the pinned processor with the configured language and task."""
    return WhisperProcessor.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        language=model_config["language"],
        task=model_config["task"],
    )


def build_prepare_dataset(
    processor: WhisperProcessor, data_config: Mapping[str, Any]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the dataset mapping callback used by every training strategy."""

    def prepare_dataset(example: dict[str, Any]) -> dict[str, Any]:
        audio = example[data_config["audio_column"]]
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        return {
            "input_features": inputs.input_features[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": processor.tokenizer(
                example[data_config["text_column"]]
            ).input_ids,
        }

    return prepare_dataset


def build_compute_metrics(
    processor: WhisperProcessor, metric: Any
) -> Callable[[Any], dict[str, float]]:
    """Build the generated-text WER callback used by every training strategy."""

    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        prediction_text = processor.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        label_text = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        return {
            "wer": 100
            * metric.compute(predictions=prediction_text, references=label_text)
        }

    return compute_metrics


def build_training_arguments(
    output_dir: Path,
    training_config: Mapping[str, Any],
    epochs: float,
    seed: int,
    run_name: str,
    *,
    preserve_curriculum_index: bool = False,
) -> Seq2SeqTrainingArguments:
    """Build the shared trainer arguments with the dynamic-curriculum exception."""
    arguments = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "learning_rate": training_config["learning_rate"],
        "warmup_steps": training_config["warmup_steps"],
        "num_train_epochs": epochs,
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        "fp16": training_config["fp16"],
        "eval_strategy": training_config["eval_strategy"],
        "predict_with_generate": training_config["predict_with_generate"],
        "generation_max_length": training_config["generation_max_length"],
        "save_strategy": training_config["save_strategy"],
        "save_total_limit": training_config["save_total_limit"],
        "logging_steps": training_config["logging_steps"],
        "optim": training_config.get("optim", "adamw_torch"),
        "report_to": ["wandb"],
        "run_name": run_name,
        "load_best_model_at_end": training_config["load_best_model_at_end"],
        "metric_for_best_model": training_config["metric_for_best_model"],
        "greater_is_better": training_config["greater_is_better"],
        "seed": seed,
        "data_seed": seed,
        "full_determinism": training_config.get("full_determinism", False),
        "gradient_checkpointing_kwargs": training_config.get(
            "gradient_checkpointing_kwargs"
        ),
    }
    if preserve_curriculum_index:
        arguments["remove_unused_columns"] = False
    else:
        arguments.update(
            {
                "max_steps": training_config.get("max_steps", -1),
                "eval_steps": training_config.get("eval_steps"),
                "save_steps": training_config.get("save_steps", 500),
                "save_only_model": training_config.get("save_only_model", False),
            }
        )
    return Seq2SeqTrainingArguments(**arguments)


def load_model(model_config: Mapping[str, Any]) -> WhisperForConditionalGeneration:
    """Load a pinned Whisper model and apply the common generation contract."""
    model = WhisperForConditionalGeneration.from_pretrained(
        model_config["id"], revision=model_config["revision"]
    )
    if model_config.get("clear_forced_decoder_ids", False):
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
    if model_config.get("clear_suppress_tokens", False):
        model.config.suppress_tokens = []
        model.generation_config.suppress_tokens = []
    model.generation_config.language = model_config["language"]
    model.generation_config.task = model_config["task"]
    return model