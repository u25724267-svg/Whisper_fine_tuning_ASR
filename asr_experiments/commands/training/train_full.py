import argparse
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
)

from asr_experiments.config import PROJECT_ROOT, load_config, resolve_path
from asr_experiments.provenance import sha256_file
from asr_experiments.training.whisper import (
    build_compute_metrics,
    build_prepare_dataset,
    build_training_arguments,
    load_model,
    load_processor,
)


ROOT_DIR = PROJECT_ROOT
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


def _required_integer(
    augmentation_config: Dict[str, Any], name: str, minimum: int
) -> int:
    value = augmentation_config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"SpecAugment {name} must be an integer >= {minimum}")
    return value


def validate_augmentation_config(
    augmentation_config: Dict[str, Any], num_mel_bins: int = 80
) -> None:
    augmentation_type = augmentation_config.get("type")
    supported_types = {"none", "specaugment", "specaugment_paper_masks"}
    if augmentation_type not in supported_types:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
    if augmentation_type == "none":
        if augmentation_config.get("enabled", False):
            raise ValueError("Augmentation type 'none' cannot be enabled")
        return
    if not augmentation_config.get("enabled", False):
        return

    application_probability = float(
        augmentation_config.get("application_probability", 1.0)
    )
    if not 0.0 <= application_probability <= 1.0:
        raise ValueError("SpecAugment application_probability must be between 0 and 1")

    if augmentation_type == "specaugment":
        for name in ("mask_time_prob", "mask_feature_prob"):
            value = float(augmentation_config[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"SpecAugment {name} must be between 0 and 1")
        _required_integer(augmentation_config, "mask_time_length", 1)
        _required_integer(augmentation_config, "mask_time_min_masks", 0)
        feature_length = _required_integer(
            augmentation_config, "mask_feature_length", 1
        )
        _required_integer(augmentation_config, "mask_feature_min_masks", 0)
        if feature_length > num_mel_bins:
            raise ValueError("SpecAugment mask_feature_length exceeds Mel bins")
        return

    if augmentation_type == "specaugment_paper_masks":
        frequency_count = _required_integer(
            augmentation_config, "frequency_mask_count", 0
        )
        frequency_width = _required_integer(
            augmentation_config, "frequency_mask_max_width", 0
        )
        time_count = _required_integer(augmentation_config, "time_mask_count", 0)
        time_width = _required_integer(
            augmentation_config, "time_mask_max_width", 0
        )
        time_proportion = float(augmentation_config["time_mask_max_proportion"])
        if not 0.0 < time_proportion <= 1.0:
            raise ValueError(
                "SpecAugment time_mask_max_proportion must be in (0, 1]"
            )
        if frequency_count and not frequency_width:
            raise ValueError("Frequency masks require a positive maximum width")
        if time_count and not time_width:
            raise ValueError("Time masks require a positive maximum width")
        if frequency_width > num_mel_bins:
            raise ValueError("Frequency mask maximum width exceeds Mel bins")


def apply_paper_specaugment_masks(
    input_features: torch.Tensor,
    attention_mask: torch.Tensor = None,
    *,
    application_probability: float,
    frequency_mask_count: int,
    frequency_mask_max_width: int,
    time_mask_count: int,
    time_mask_max_width: int,
    time_mask_max_proportion: float,
) -> torch.Tensor:
    masked_features = input_features.clone()
    batch_size, mel_bins, sequence_length = masked_features.shape
    if application_probability == 0.0:
        return masked_features
    if attention_mask is not None and attention_mask.shape != (
        batch_size,
        sequence_length,
    ):
        raise ValueError("SpecAugment attention mask is not aligned with features")

    if application_probability == 1.0:
        apply_to_example = torch.ones(
            batch_size, dtype=torch.bool, device=masked_features.device
        )
    else:
        apply_to_example = (
            torch.rand(batch_size, device=masked_features.device)
            < application_probability
        )

    for batch_index in range(batch_size):
        if not bool(apply_to_example[batch_index]):
            continue
        valid_frames = (
            int(attention_mask[batch_index].sum().item())
            if attention_mask is not None
            else sequence_length
        )
        valid_frames = max(0, min(sequence_length, valid_frames))

        for _ in range(frequency_mask_count):
            width = int(
                torch.randint(
                    0,
                    frequency_mask_max_width + 1,
                    (1,),
                    device=masked_features.device,
                ).item()
            )
            if width == 0:
                continue
            start = int(
                torch.randint(
                    0,
                    mel_bins - width + 1,
                    (1,),
                    device=masked_features.device,
                ).item()
            )
            masked_features[batch_index, start : start + width, :] = 0

        maximum_time_width = min(
            time_mask_max_width,
            int(time_mask_max_proportion * valid_frames),
            valid_frames,
        )
        for _ in range(time_mask_count):
            if maximum_time_width == 0:
                continue
            width = int(
                torch.randint(
                    0,
                    maximum_time_width + 1,
                    (1,),
                    device=masked_features.device,
                ).item()
            )
            if width == 0:
                continue
            start = int(
                torch.randint(
                    0,
                    valid_frames - width + 1,
                    (1,),
                    device=masked_features.device,
                ).item()
            )
            masked_features[batch_index, :, start : start + width] = 0

    return masked_features


def configure_augmentation(model: Any, augmentation_config: Dict[str, Any]) -> None:
    validate_augmentation_config(
        augmentation_config, num_mel_bins=int(model.config.num_mel_bins)
    )

    model.config.apply_spec_augment = bool(augmentation_config.get("enabled", False))
    if not model.config.apply_spec_augment:
        return

    application_probability = float(augmentation_config.get("application_probability", 1.0))
    if augmentation_config["type"] == "specaugment_paper_masks":
        model.config.specaugment_implementation = "paper_masks_without_time_warp"
        model.config.specaugment_application_probability = application_probability
        model.config.frequency_mask_count = augmentation_config[
            "frequency_mask_count"
        ]
        model.config.frequency_mask_max_width = augmentation_config[
            "frequency_mask_max_width"
        ]
        model.config.time_mask_count = augmentation_config["time_mask_count"]
        model.config.time_mask_max_width = augmentation_config[
            "time_mask_max_width"
        ]
        model.config.time_mask_max_proportion = augmentation_config[
            "time_mask_max_proportion"
        ]

        def paper_mask_input_features(
            whisper_model: Any,
            input_features: torch.Tensor,
            attention_mask: torch.Tensor = None,
        ) -> torch.Tensor:
            if not whisper_model.training:
                return input_features
            return apply_paper_specaugment_masks(
                input_features,
                attention_mask,
                application_probability=application_probability,
                frequency_mask_count=augmentation_config[
                    "frequency_mask_count"
                ],
                frequency_mask_max_width=augmentation_config[
                    "frequency_mask_max_width"
                ],
                time_mask_count=augmentation_config["time_mask_count"],
                time_mask_max_width=augmentation_config["time_mask_max_width"],
                time_mask_max_proportion=augmentation_config[
                    "time_mask_max_proportion"
                ],
            )

        model.model._mask_input_features = MethodType(
            paper_mask_input_features, model.model
        )
        return

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

    processor = load_processor(model_config)
    validate_augmentation_config(
        config.get("augmentation", {"type": "none", "enabled": False}),
        num_mel_bins=int(processor.feature_extractor.feature_size),
    )

    prepare_dataset = build_prepare_dataset(processor, data_config)

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
    compute_metrics = build_compute_metrics(processor, metric)

    model = load_model(model_config)

    augmentation_config = config.get("augmentation", {"type": "none", "enabled": False})
    configure_augmentation(model, augmentation_config)

    training_args = build_training_arguments(
        output_dir,
        training_config,
        epochs,
        seed,
        wandb_config["run_name"],
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