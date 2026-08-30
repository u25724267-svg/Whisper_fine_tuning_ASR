import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Union

import evaluate
import torch
import wandb
from datasets import Audio, DatasetDict, load_dataset
from dotenv import load_dotenv
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = ROOT_DIR / "configs" / "whisper-base-shona-3epochs.json"
ENV_FILE = ROOT_DIR / ".env"
WANDB_PROJECTS_FILE = ROOT_DIR / "configs" / "wandb-projects.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper from a versioned experiment configuration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate data, preprocessing, and W&B configuration.")
    return parser.parse_args()


def resolve_path(path: Union[str, Path]) -> Path:
    resolved = Path(os.path.expandvars(str(path))).expanduser()
    return resolved if resolved.is_absolute() else (ROOT_DIR / resolved).resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    resolved_path = resolve_path(config_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing experiment configuration: {resolved_path}")
    with resolved_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["config_path"] = str(resolved_path)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    config: Dict[str, Any], output_dir: Path, epochs: float, seed: int
) -> None:
    package_names = [
        "accelerate",
        "bitsandbytes",
        "datasets",
        "evaluate",
        "jiwer",
        "numpy",
        "python-dotenv",
        "torch",
        "transformers",
        "wandb",
    ]
    data_config = config["data"]
    source_paths = {
        "trainer": Path(__file__).resolve(),
        "config": resolve_path(config["config_path"]),
        "dependencies": ROOT_DIR / "requirements-training.txt",
        "train_manifest": resolve_path(data_config["train_manifest"]),
        "validation_manifest": resolve_path(data_config["validation_manifest"]),
        "test_manifest": resolve_path(data_config["test_manifest"]),
    }
    for name, path in data_config.get("evaluation_manifests", {}).items():
        source_paths[f"evaluation_manifest_{name}"] = resolve_path(path)
    local_model_path = resolve_path(config["model"]["id"])
    if local_model_path.is_dir():
        for filename in ("model.safetensors", "config.json", "generation_config.json"):
            model_file = local_model_path / filename
            if model_file.is_file():
                source_paths[f"parent_model_{filename}"] = model_file
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "augmentation": config.get("augmentation", {"type": "none", "enabled": False}),
        "effective_num_train_epochs": epochs,
        "effective_max_steps": config["training"].get("max_steps", -1),
        "effective_seed": seed,
        "effective_output_dir": str(output_dir),
        "sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_environment(require_api_key: bool, wandb_config: Dict[str, Any]) -> None:
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Missing environment file: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)

    with WANDB_PROJECTS_FILE.open(encoding="utf-8") as projects_file:
        allowed_projects = set(json.load(projects_file)["allowed_projects"])
    if wandb_config["project"] not in allowed_projects:
        raise ValueError(
            f"W&B project '{wandb_config['project']}' is not approved. "
            f"Allowed projects: {sorted(allowed_projects)}"
        )

    os.environ["WANDB_PROJECT"] = wandb_config["project"]
    os.environ["WANDB_NAME"] = wandb_config["run_name"]
    os.environ["WANDB_JOB_TYPE"] = wandb_config.get("job_type", "training")
    os.environ["WANDB_TAGS"] = ",".join(wandb_config.get("tags", []))
    os.environ["WANDB_WATCH"] = str(wandb_config.get("watch", "false"))
    os.environ["WANDB_LOG_MODEL"] = str(wandb_config.get("log_model", "false"))

    resume_run_id = wandb_config.get("resume_run_id")
    if resume_run_id:
        os.environ["WANDB_RUN_ID"] = str(resume_run_id)
        os.environ["WANDB_RESUME"] = "allow"
    else:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)

    required = ["WANDB_PROJECT", "WANDB_NAME"]
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if require_api_key and not os.getenv("WANDB_API_KEY", "").strip():
        missing.append("WANDB_API_KEY")
    if missing:
        raise RuntimeError(f"Missing W&B settings in {ENV_FILE}: {', '.join(missing)}")


def load_local_data(data_config: Dict[str, Any]) -> DatasetDict:
    data_files = {
        "train": str(resolve_path(data_config["train_manifest"])),
        "validation": str(resolve_path(data_config["validation_manifest"])),
        "test": str(resolve_path(data_config["test_manifest"])),
    }
    missing_manifests = [path for path in data_files.values() if not Path(path).is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing manifests: {missing_manifests}")

    audio_column = data_config["audio_column"]
    text_column = data_config["text_column"]
    dataset = load_dataset("json", data_files=data_files)
    for split in dataset:
        missing_audio = [path for path in dataset[split][audio_column] if not Path(path).is_file()]
        empty_text = [text for text in dataset[split][text_column] if not str(text).strip()]
        if missing_audio or empty_text:
            raise ValueError(
                f"Invalid {split} data: {len(missing_audio)} missing audio files, {len(empty_text)} empty transcripts"
            )
        dataset[split] = dataset[split].cast_column(
            audio_column, Audio(sampling_rate=data_config["sampling_rate"])
        )
    return dataset


def load_evaluation_data(data_config: Dict[str, Any]) -> DatasetDict:
    evaluation_manifests = data_config.get("evaluation_manifests", {})
    if not evaluation_manifests:
        return DatasetDict()

    manifest_paths = {
        name: resolve_path(path) for name, path in evaluation_manifests.items()
    }
    missing_manifests = [str(path) for path in manifest_paths.values() if not path.is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing evaluation manifests: {missing_manifests}")

    audio_column = data_config["audio_column"]
    text_column = data_config["text_column"]
    evaluation_splits = {}
    for split_name, manifest_path in manifest_paths.items():
        split_dataset = load_dataset(
            "json", data_files={split_name: str(manifest_path)}
        )[split_name]
        missing_audio = [path for path in split_dataset[audio_column] if not Path(path).is_file()]
        empty_text = [text for text in split_dataset[text_column] if not str(text).strip()]
        if missing_audio or empty_text:
            raise ValueError(
                f"Invalid {split_name} data: {len(missing_audio)} missing audio files, "
                f"{len(empty_text)} empty transcripts"
            )
        evaluation_splits[split_name] = split_dataset.cast_column(
            audio_column, Audio(sampling_rate=data_config["sampling_rate"])
        )
    return DatasetDict(evaluation_splits)


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


def configure_augmentation(model: Any, augmentation_config: Dict[str, Any]) -> None:
    if augmentation_config["type"] not in {"none", "specaugment"}:
        raise ValueError(f"Unsupported augmentation type: {augmentation_config['type']}")

    model.config.apply_spec_augment = bool(augmentation_config.get("enabled", False))
    if not model.config.apply_spec_augment:
        return

    application_probability = float(augmentation_config.get("application_probability", 1.0))
    if not 0.0 <= application_probability <= 1.0:
        raise ValueError("SpecAugment application_probability must be between 0 and 1")

    model.config.mask_time_prob = augmentation_config["mask_time_prob"]
    model.config.mask_time_length = augmentation_config["mask_time_length"]
    model.config.mask_time_min_masks = augmentation_config["mask_time_min_masks"]
    model.config.mask_feature_prob = augmentation_config["mask_feature_prob"]
    model.config.mask_feature_length = augmentation_config["mask_feature_length"]
    model.config.mask_feature_min_masks = augmentation_config["mask_feature_min_masks"]
    model.config.specaugment_application_probability = application_probability

    if application_probability == 1.0:
        return

    original_mask_input_features = model.model._mask_input_features

    def probabilistic_mask_input_features(
        whisper_model: Any,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if not whisper_model.training:
            return input_features
        clean_features = input_features.clone()
        augmented_features = original_mask_input_features(input_features, attention_mask)
        apply_mask = torch.rand(input_features.shape[0], device=input_features.device) < application_probability
        return torch.where(apply_mask[:, None, None], augmented_features, clean_features)

    model.model._mask_input_features = MethodType(probabilistic_mask_input_features, model.model)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    wandb_config = config["wandb"]
    output_dir = resolve_path(args.output_dir or config["output_dir"])
    epochs = args.epochs if args.epochs is not None else training_config["num_train_epochs"]
    seed = args.seed if args.seed is not None else training_config["seed"]

    load_environment(require_api_key=not args.dry_run, wandb_config=wandb_config)
    dataset = load_local_data(data_config)
    evaluation_dataset = load_evaluation_data(data_config)
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['validation'])} validation, "
        f"and {len(dataset['test'])} test examples"
    )
    if evaluation_dataset:
        print(
            "Additional evaluation splits: "
            + ", ".join(f"{name}={len(split)}" for name, split in evaluation_dataset.items())
        )

    processor = WhisperProcessor.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        language=model_config["language"],
        task=model_config["task"],
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
        prepared = prepare_dataset(dataset["train"][0])
        for split in evaluation_dataset.values():
            prepare_dataset(split[0])
        print(
            f"Dry run passed: {len(prepared['input_features'])} mel bins, "
            f"{len(prepared['attention_mask'])} mask frames, {len(prepared['labels'])} label tokens, "
            f"config {config['experiment_name']}, W&B project {os.environ['WANDB_PROJECT']}"
        )
        return

    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "effective_num_train_epochs": epochs,
        "effective_seed": seed,
        "effective_output_dir": str(output_dir),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    write_run_manifest(config, output_dir, epochs, seed)
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        keep_in_memory=data_config.get("keep_preprocessed_in_memory", False),
        desc="Extracting Whisper features",
    )
    if evaluation_dataset:
        evaluation_dataset = evaluation_dataset.map(
            prepare_dataset,
            remove_columns=next(iter(evaluation_dataset.values())).column_names,
            num_proc=1,
            keep_in_memory=data_config.get("keep_preprocessed_in_memory", False),
            desc="Extracting additional evaluation features",
        )

    metric = evaluate.load("wer")

    def compute_metrics(prediction: Any) -> Dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        prediction_text = processor.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        label_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=prediction_text, references=label_text)}

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

    augmentation_config = config.get("augmentation", {"type": "none", "enabled": False})
    configure_augmentation(model, augmentation_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        num_train_epochs=epochs,
        max_steps=training_config.get("max_steps", -1),
        gradient_checkpointing=training_config["gradient_checkpointing"],
        fp16=training_config["fp16"],
        eval_strategy=training_config["eval_strategy"],
        eval_steps=training_config.get("eval_steps"),
        predict_with_generate=training_config["predict_with_generate"],
        generation_max_length=training_config["generation_max_length"],
        save_strategy=training_config["save_strategy"],
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config["save_total_limit"],
        save_only_model=training_config.get("save_only_model", False),
        logging_steps=training_config["logging_steps"],
        optim=training_config.get("optim", "adamw_torch"),
        report_to=["wandb"],
        run_name=wandb_config["run_name"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        seed=seed,
        data_seed=seed,
        full_determinism=training_config.get("full_determinism", False),
        gradient_checkpointing_kwargs=training_config.get("gradient_checkpointing_kwargs"),
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

    checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    for split_name, split_dataset in evaluation_dataset.items():
        split_metrics = trainer.evaluate(split_dataset, metric_key_prefix=split_name)
        trainer.log_metrics(split_name, split_metrics)
        trainer.save_metrics(split_name, split_metrics)

    trainer.save_model()
    processor.save_pretrained(output_dir)
    trainer.save_state()
    wandb.finish()
    print(f"Full run complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()