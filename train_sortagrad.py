import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import evaluate
import wandb
from torch.utils.data import Sampler
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from experiment_core.samplers import AuditedSortaGradSampler
from train_full import (
    DataCollatorSpeechSeq2SeqWithPadding,
    Seq2SeqTrainer,
    configure_augmentation,
    load_config,
    load_environment,
    load_evaluation_data,
    load_local_data,
    parse_args,
    resolve_path,
    sha256_file,
    write_run_manifest,
)


ROOT_DIR = Path(__file__).resolve().parent
SAMPLER_FILE = ROOT_DIR / "experiment_core" / "samplers.py"


class SortaGradTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        sortagrad_ids: list[str],
        sortagrad_durations: list[float],
        sortagrad_seed: int,
        sortagrad_artifact_dir: Path,
        **kwargs: Any,
    ) -> None:
        self.sortagrad_ids = sortagrad_ids
        self.sortagrad_durations = sortagrad_durations
        self.sortagrad_seed = sortagrad_seed
        self.sortagrad_artifact_dir = sortagrad_artifact_dir
        self.sortagrad_sampler: Optional[AuditedSortaGradSampler] = None
        super().__init__(*args, **kwargs)
        if self.train_dataset is None:
            raise ValueError("SortaGrad requires a training dataset")
        if len(self.train_dataset) != len(self.sortagrad_ids):
            raise ValueError("SortaGrad metadata must align one-to-one with training rows")

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None:
            return None
        if self.sortagrad_sampler is None:
            self.sortagrad_sampler = AuditedSortaGradSampler(
                utterance_ids=self.sortagrad_ids,
                durations=self.sortagrad_durations,
                seed=self.sortagrad_seed,
                artifact_dir=self.sortagrad_artifact_dir,
            )
        return self.sortagrad_sampler


def add_curriculum_provenance(output_dir: Path, config: Dict[str, Any]) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["curriculum"] = config["curriculum"]
    manifest["sha256"]["curriculum_runner"] = sha256_file(Path(__file__).resolve())
    manifest["sha256"]["curriculum_sampler"] = sha256_file(SAMPLER_FILE)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def order_digest(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("curriculum", {}).get("type") != "sortagrad":
        raise ValueError("train_sortagrad.py requires curriculum.type='sortagrad'")

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
    train_ids = [str(utterance_id) for utterance_id in dataset["train"]["id"]]
    train_durations = [float(duration) for duration in dataset["train"]["duration"]]
    print(
        f"Loaded {len(dataset['train'])} train, {len(dataset['validation'])} validation, "
        f"and {len(dataset['test'])} test examples"
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

    sampler_probe = AuditedSortaGradSampler(train_ids, train_durations, seed)
    first_order = sampler_probe.indices_for_epoch(0)
    second_order = sampler_probe.indices_for_epoch(1)
    third_order = sampler_probe.indices_for_epoch(2)
    if any(
        train_durations[left] > train_durations[right]
        for left, right in zip(first_order, first_order[1:])
    ):
        raise RuntimeError("SortaGrad epoch 1 is not duration-monotonic")
    if any(sorted(order) != list(range(len(train_ids))) for order in (first_order, second_order, third_order)):
        raise RuntimeError("SortaGrad preflight found an incomplete epoch order")
    if second_order == third_order:
        raise RuntimeError("SortaGrad random epochs unexpectedly have identical order")

    if args.dry_run:
        prepared = prepare_dataset(dataset["train"][first_order[0]])
        print(
            f"SortaGrad dry run passed: {len(train_ids)} rows, duration range "
            f"{min(train_durations):.3f}-{max(train_durations):.3f}s, "
            f"epoch order hashes {order_digest(first_order)[:12]}, "
            f"{order_digest(second_order)[:12]}, {order_digest(third_order)[:12]}, "
            f"{len(prepared['input_features'])} mel bins, W&B project "
            f"{os.environ['WANDB_PROJECT']}"
        )
        return

    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "effective_num_train_epochs": epochs,
        "effective_seed": seed,
        "effective_output_dir": str(output_dir),
        "sortagrad_order_sha256": {
            "epoch_1": order_digest(first_order),
            "epoch_2": order_digest(second_order),
            "epoch_3": order_digest(third_order),
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    write_run_manifest(config, output_dir, epochs, seed)
    add_curriculum_provenance(output_dir, config)

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
        prediction_text = processor.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
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
    configure_augmentation(
        model, config.get("augmentation", {"type": "none", "enabled": False})
    )

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

    trainer = SortaGradTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        sortagrad_ids=train_ids,
        sortagrad_durations=train_durations,
        sortagrad_seed=seed,
        sortagrad_artifact_dir=output_dir / "curriculum_orders",
    )

    checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    expected_order_files = {
        output_dir / "curriculum_orders" / f"epoch_{epoch:02d}_order.jsonl"
        for epoch in range(1, int(epochs) + 1)
    }
    missing_order_files = [str(path) for path in expected_order_files if not path.is_file()]
    if missing_order_files:
        raise RuntimeError(f"Missing SortaGrad epoch-order artifacts: {missing_order_files}")

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
    print(f"SortaGrad run complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()