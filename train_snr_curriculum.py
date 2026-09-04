import csv
import hashlib
import json
import os
import random
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

from experiment_core.samplers import (
    AuditedCumulativePacingSampler,
    AuditedStaticScoreSampler,
    percentile_ranks,
)
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


class StaticScoreTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        curriculum_ids: list[str],
        difficulty_scores: list[float],
        score_metadata: Dict[str, list[float]],
        curriculum_strategy: str,
        curriculum_artifact_dir: Path,
        **kwargs: Any,
    ) -> None:
        self.curriculum_ids = curriculum_ids
        self.difficulty_scores = difficulty_scores
        self.score_metadata = score_metadata
        self.curriculum_strategy = curriculum_strategy
        self.curriculum_artifact_dir = curriculum_artifact_dir
        self.curriculum_sampler: Optional[AuditedStaticScoreSampler] = None
        super().__init__(*args, **kwargs)
        if self.train_dataset is None:
            raise ValueError("Static curriculum requires a training dataset")
        if len(self.train_dataset) != len(self.curriculum_ids):
            raise ValueError("Static curriculum metadata is not aligned with training rows")

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None:
            return None
        if self.curriculum_sampler is None:
            self.curriculum_sampler = AuditedStaticScoreSampler(
                utterance_ids=self.curriculum_ids,
                difficulty_scores=self.difficulty_scores,
                strategy=self.curriculum_strategy,
                score_metadata=self.score_metadata,
                artifact_dir=self.curriculum_artifact_dir,
            )
        return self.curriculum_sampler


class CumulativePacingTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        curriculum_ids: list[str],
        priority_scores: list[float],
        score_metadata: Dict[str, list[float]],
        curriculum_strategy: str,
        curriculum_seed: int,
        eligible_fractions: list[float],
        curriculum_artifact_dir: Path,
        **kwargs: Any,
    ) -> None:
        self.curriculum_ids = curriculum_ids
        self.priority_scores = priority_scores
        self.score_metadata = score_metadata
        self.curriculum_strategy = curriculum_strategy
        self.curriculum_seed = curriculum_seed
        self.eligible_fractions = eligible_fractions
        self.curriculum_artifact_dir = curriculum_artifact_dir
        self.curriculum_sampler: Optional[AuditedCumulativePacingSampler] = None
        super().__init__(*args, **kwargs)
        if self.train_dataset is None:
            raise ValueError("Cumulative pacing requires a training dataset")
        if len(self.train_dataset) != len(self.curriculum_ids):
            raise ValueError("Pacing metadata is not aligned with training rows")

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None:
            return None
        if self.curriculum_sampler is None:
            self.curriculum_sampler = AuditedCumulativePacingSampler(
                utterance_ids=self.curriculum_ids,
                priority_scores=self.priority_scores,
                strategy=self.curriculum_strategy,
                seed=self.curriculum_seed,
                eligible_fractions=self.eligible_fractions,
                score_metadata=self.score_metadata,
                artifact_dir=self.curriculum_artifact_dir,
            )
        return self.curriculum_sampler


def load_train_snr(path: Path, train_ids: list[str]) -> list[float]:
    return load_train_metadata_field(path, train_ids, "snr_proxy_db")


def load_train_metadata_field(
    path: Path, train_ids: list[str], field_name: str
) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing acoustic metadata: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "train"]
    by_id = {str(row["id"]): float(row[field_name]) for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Acoustic metadata contains duplicate training IDs")
    missing = [utterance_id for utterance_id in train_ids if utterance_id not in by_id]
    extras = sorted(set(by_id) - set(train_ids))
    if missing or extras:
        raise ValueError(
            f"Acoustic metadata alignment failed: {len(missing)} missing, "
            f"{len(extras)} extra IDs"
        )
    return [by_id[utterance_id] for utterance_id in train_ids]


def order_digest(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def add_curriculum_provenance(
    output_dir: Path, config: Dict[str, Any], metadata_path: Path
) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["curriculum"] = config["curriculum"]
    manifest["sha256"]["curriculum_runner"] = sha256_file(Path(__file__).resolve())
    manifest["sha256"]["curriculum_sampler"] = sha256_file(SAMPLER_FILE)
    manifest["sha256"]["acoustic_metadata"] = sha256_file(metadata_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    curriculum_type = config.get("curriculum", {}).get("type")
    if curriculum_type not in {
        "snr",
        "snr_duration",
        "acoustic_cumulative",
        "random_pacing",
    }:
        raise ValueError(
            "Unsupported static acoustic curriculum type: "
            f"{curriculum_type}"
        )

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    wandb_config = config["wandb"]
    curriculum_config = config["curriculum"]
    metadata_path = resolve_path(curriculum_config["metadata_path"])
    output_dir = resolve_path(args.output_dir or config["output_dir"])
    epochs = args.epochs if args.epochs is not None else training_config["num_train_epochs"]
    seed = args.seed if args.seed is not None else training_config["seed"]

    load_environment(require_api_key=not args.dry_run, wandb_config=wandb_config)
    dataset = load_local_data(data_config)
    evaluation_dataset = load_evaluation_data(data_config)
    train_ids = [str(utterance_id) for utterance_id in dataset["train"]["id"]]
    duration_source = curriculum_config.get("duration_source", "train_manifest")
    if duration_source == "train_manifest":
        train_durations = [
            float(duration) for duration in dataset["train"]["duration"]
        ]
    elif duration_source == "acoustic_metadata_manifest_duration":
        train_durations = load_train_metadata_field(
            metadata_path, train_ids, "manifest_duration"
        )
    else:
        raise ValueError(f"Unsupported curriculum duration source: {duration_source}")
    snr_values = load_train_snr(metadata_path, train_ids)
    snr_percentiles = percentile_ranks(snr_values)
    duration_percentiles = percentile_ranks(train_durations)
    is_pacing = curriculum_type in {"acoustic_cumulative", "random_pacing"}
    eligible_fractions = [
        float(value)
        for value in curriculum_config.get(
            "eligible_fractions", [1 / 3, 2 / 3, 1.0]
        )
    ]
    if is_pacing and eligible_fractions != [1 / 3, 2 / 3, 1.0]:
        raise ValueError("C4/C4R require eligible fractions [1/3, 2/3, 1]")
    if curriculum_type == "snr":
        difficulty_scores = [1.0 - percentile for percentile in snr_percentiles]
        strategy = "snr_high_to_low"
        score_metadata = {
            "snr_proxy_db": snr_values,
            "snr_percentile": snr_percentiles,
        }
    else:
        difficulty_scores = [
            0.5 * duration_percentile + 0.5 * (1.0 - snr_percentile)
            for duration_percentile, snr_percentile in zip(
                duration_percentiles, snr_percentiles
            )
        ]
        strategy = "joint_snr_duration_easy_to_hard"
        score_metadata = {
            "duration": train_durations,
            "duration_percentile": duration_percentiles,
            "snr_proxy_db": snr_values,
            "snr_percentile": snr_percentiles,
        }
        if curriculum_type == "acoustic_cumulative":
            strategy = "acoustic_cumulative_pacing"
        elif curriculum_type == "random_pacing":
            random_priority = list(range(len(train_ids)))
            random.Random(seed).shuffle(random_priority)
            difficulty_scores = [0.0] * len(train_ids)
            for priority_rank, index in enumerate(random_priority):
                difficulty_scores[index] = priority_rank / (len(train_ids) - 1)
            strategy = "random_priority_cumulative_pacing"
            score_metadata = {"random_priority_score": difficulty_scores}
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

    if is_pacing:
        sampler_probe = AuditedCumulativePacingSampler(
            train_ids,
            difficulty_scores,
            strategy=strategy,
            seed=seed,
            eligible_fractions=eligible_fractions,
            score_metadata=score_metadata,
        )
        orders = [sampler_probe.indices_for_epoch(epoch) for epoch in range(3)]
        expected_unique_counts = [
            sampler_probe.eligible_count(epoch) for epoch in range(3)
        ]
        if [len(set(order)) for order in orders] != expected_unique_counts:
            raise RuntimeError("Pacing eligible-pool sizes do not match the contract")
        if any(len(order) != len(train_ids) for order in orders):
            raise RuntimeError("Pacing epochs do not match the C0 presentation budget")
        if not set(orders[0]) < set(orders[1]) < set(orders[2]):
            raise RuntimeError("Pacing eligible pools are not strictly nested")
    else:
        sampler_probe = AuditedStaticScoreSampler(
            train_ids,
            difficulty_scores,
            strategy=strategy,
            score_metadata=score_metadata,
        )
        order = sampler_probe.indices_for_epoch()
        orders = [order, order, order]
        ordered_difficulty = [difficulty_scores[index] for index in order]
        if any(
            left > right
            for left, right in zip(ordered_difficulty, ordered_difficulty[1:])
        ):
            raise RuntimeError("Static acoustic order is not difficulty-monotonic")
        ordered_snr = [snr_values[index] for index in order]
        if curriculum_type == "snr" and any(
            left < right for left, right in zip(ordered_snr, ordered_snr[1:])
        ):
            raise RuntimeError("C2 order is not monotonically high-to-low SNR")
        if sorted(order) != list(range(len(train_ids))):
            raise RuntimeError("Static order is not a complete training permutation")

    if args.dry_run:
        prepared = prepare_dataset(dataset["train"][orders[0][0]])
        print(
            f"{curriculum_type} curriculum dry run passed: {len(train_ids)} "
            f"aligned rows, SNR range "
            f"{min(snr_values):.3f}-{max(snr_values):.3f} dB, order hash "
            f"{order_digest(orders[0])[:12]}, {len(prepared['input_features'])} mel bins, "
            f"W&B project {os.environ['WANDB_PROJECT']}"
        )
        return

    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "effective_num_train_epochs": epochs,
        "effective_seed": seed,
        "effective_output_dir": str(output_dir),
        "epoch_order_sha256": {
            f"epoch_{epoch + 1}": order_digest(order)
            for epoch, order in enumerate(orders)
        },
        "snr_summary": {
            "minimum_db": min(snr_values),
            "maximum_db": max(snr_values),
            "capped_at_60_db_rows": sum(value == 60.0 for value in snr_values),
        },
        "duration_summary": {
            "minimum_seconds": min(train_durations),
            "maximum_seconds": max(train_durations),
        },
        "pacing_unique_rows": (
            [len(set(order)) for order in orders] if is_pacing else None
        ),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    write_run_manifest(config, output_dir, epochs, seed)
    add_curriculum_provenance(output_dir, config, metadata_path)

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

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        "compute_metrics": compute_metrics,
        "processing_class": processor.feature_extractor,
    }
    if is_pacing:
        trainer = CumulativePacingTrainer(
            **trainer_kwargs,
            curriculum_ids=train_ids,
            priority_scores=difficulty_scores,
            score_metadata=score_metadata,
            curriculum_strategy=strategy,
            curriculum_seed=seed,
            eligible_fractions=eligible_fractions,
            curriculum_artifact_dir=output_dir / "curriculum_orders",
        )
    else:
        trainer = StaticScoreTrainer(
            **trainer_kwargs,
            curriculum_ids=train_ids,
            difficulty_scores=difficulty_scores,
            score_metadata=score_metadata,
            curriculum_strategy=strategy,
            curriculum_artifact_dir=output_dir / "curriculum_orders",
        )

    checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    expected_order_files = {
        output_dir / "curriculum_orders" / f"epoch_{epoch:02d}_order.jsonl"
        for epoch in range(1, int(epochs) + 1)
    }
    if is_pacing:
        expected_order_files.add(
            output_dir / "curriculum_orders" / "priority_ranking.jsonl"
        )
    missing_order_files = [str(path) for path in expected_order_files if not path.is_file()]
    if missing_order_files:
        raise RuntimeError(
            f"Missing static curriculum epoch-order artifacts: {missing_order_files}"
        )

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
    print(f"Static acoustic curriculum run complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()