import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import jiwer
import torch
import torch.nn.functional as functional
import wandb
from torch.utils.data import Sampler
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from experiment_core.dynamic_curriculum import (
    AuditedS2SLossSampler,
    AuditedWERMarginSampler,
    HybridS2SCurriculumState,
    S2SLossCurriculumState,
    WERMarginCurriculumState,
)
from experiment_core.samplers import percentile_ranks
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
from train_snr_curriculum import load_train_snr


ROOT_DIR = Path(__file__).resolve().parent
STATE_FILE = ROOT_DIR / "experiment_core" / "dynamic_curriculum.py"


@dataclass
class S2SCurriculumCollator:
    processor: Any

    def __post_init__(self) -> None:
        self.base_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, int]]]
    ) -> Dict[str, torch.Tensor]:
        indices = [feature.get("_curriculum_index") for feature in features]
        if any(index is not None for index in indices) and not all(
            index is not None for index in indices
        ):
            raise RuntimeError("Partially missing curriculum indices in a batch")
        batch = self.base_collator(features)
        if all(index is not None for index in indices):
            batch["_curriculum_index"] = torch.tensor(indices, dtype=torch.long)
        return batch


class S2SLossCurriculumTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        curriculum_ids: list[str],
        curriculum_durations: list[float],
        curriculum_seed: int,
        curriculum_artifact_dir: Path,
        curriculum_acoustic_scores: Optional[list[float]] = None,
        acoustic_weight: float = 0.5,
        s2s_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if curriculum_acoustic_scores is None:
            self.curriculum_state = S2SLossCurriculumState(
                curriculum_ids,
                curriculum_durations,
                curriculum_seed,
                curriculum_artifact_dir,
            )
        else:
            self.curriculum_state = HybridS2SCurriculumState(
                curriculum_ids,
                curriculum_durations,
                curriculum_seed,
                curriculum_artifact_dir,
                curriculum_acoustic_scores,
                acoustic_weight,
                s2s_weight,
            )
        self.curriculum_sampler: Optional[AuditedS2SLossSampler] = None
        super().__init__(*args, **kwargs)
        if self.train_dataset is None:
            raise ValueError("S2S curriculum requires a training dataset")
        if len(self.train_dataset) != len(self.curriculum_state):
            raise ValueError("S2S curriculum metadata is not aligned with training rows")

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None:
            return None
        if self.curriculum_sampler is None:
            self.curriculum_sampler = AuditedS2SLossSampler(self.curriculum_state)
        return self.curriculum_sampler

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        curriculum_indices = inputs.pop("_curriculum_index", None)
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if model.training and curriculum_indices is not None:
            if labels is None:
                raise RuntimeError("S2S score capture requires labels")
            logits = outputs.logits.float()
            token_losses = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)
            valid_tokens = labels.ne(-100).sum(dim=1)
            reconstructed_loss = token_losses.sum() / valid_tokens.sum()
            if not torch.allclose(
                reconstructed_loss.detach(), loss.detach().float(), rtol=5e-3, atol=5e-3
            ):
                raise RuntimeError(
                    "Per-example S2S losses do not reconstruct the trainer loss"
                )
            self.curriculum_state.record(
                curriculum_indices.detach().cpu().tolist(),
                token_losses.sum(dim=1).detach().cpu().tolist(),
                valid_tokens.detach().cpu().tolist(),
            )
        return (loss, outputs) if return_outputs else loss


def word_error_record(reference: str, prediction: str) -> Dict[str, Any]:
    if not reference.strip():
        insertions = len(prediction.split())
        return {
            "reference": reference,
            "prediction": prediction,
            "wer": float(insertions),
            "substitutions": 0,
            "deletions": 0,
            "insertions": insertions,
            "hits": 0,
            "reference_words": 0,
        }
    output = jiwer.process_words(reference, prediction)
    return {
        "reference": reference,
        "prediction": prediction,
        "wer": float(output.wer),
        "substitutions": int(output.substitutions),
        "deletions": int(output.deletions),
        "insertions": int(output.insertions),
        "hits": int(output.hits),
        "reference_words": int(output.hits + output.substitutions + output.deletions),
    }


class WERMarginCurriculumTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        curriculum_ids: list[str],
        curriculum_durations: list[float],
        curriculum_seed: int,
        curriculum_artifact_dir: Path,
        curriculum_tokenizer: Any,
        generation_max_length: int,
        **kwargs: Any,
    ) -> None:
        self.curriculum_state = WERMarginCurriculumState(
            curriculum_ids,
            curriculum_durations,
            curriculum_seed,
            curriculum_artifact_dir,
        )
        self.curriculum_tokenizer = curriculum_tokenizer
        self.generation_max_length = int(generation_max_length)
        self.curriculum_sampler: Optional[AuditedWERMarginSampler] = None
        super().__init__(*args, **kwargs)
        if self.train_dataset is None:
            raise ValueError("WER curriculum requires a training dataset")
        if len(self.train_dataset) != len(self.curriculum_state):
            raise ValueError("WER curriculum metadata is not aligned with training rows")
        if self.args.world_size != 1:
            raise ValueError("WER curriculum currently requires a single training process")

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None:
            return None
        if self.curriculum_sampler is None:
            self.curriculum_sampler = AuditedWERMarginSampler(self.curriculum_state)
        return self.curriculum_sampler

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        curriculum_indices = inputs.pop("_curriculum_index", None)
        labels = inputs.get("labels")
        records = None
        if model.training and curriculum_indices is not None:
            if labels is None:
                raise RuntimeError("WER score capture requires labels")
            previous_mode = model.training
            try:
                model.eval()
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_features=inputs["input_features"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=self.generation_max_length,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True,
                    )
                decoded_labels = labels.detach().clone()
                decoded_labels[decoded_labels == -100] = (
                    self.curriculum_tokenizer.pad_token_id
                )
                predictions = self.curriculum_tokenizer.batch_decode(
                    generated_ids.detach().cpu(), skip_special_tokens=True
                )
                references = self.curriculum_tokenizer.batch_decode(
                    decoded_labels.detach().cpu(), skip_special_tokens=True
                )
                records = [
                    word_error_record(reference, prediction)
                    for reference, prediction in zip(references, predictions)
                ]
                del generated_ids
            finally:
                model.train(previous_mode)

        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if curriculum_indices is not None and records is not None:
            self.curriculum_state.record(
                curriculum_indices.detach().cpu().tolist(), records
            )
        return (loss, outputs) if return_outputs else loss


def order_digest(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def add_curriculum_provenance(output_dir: Path, config: Dict[str, Any]) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["curriculum"] = config["curriculum"]
    manifest["sha256"]["curriculum_runner"] = sha256_file(Path(__file__).resolve())
    manifest["sha256"]["dynamic_curriculum_state"] = sha256_file(STATE_FILE)
    metadata_path = config["curriculum"].get("metadata_path")
    if metadata_path is not None:
        manifest["sha256"]["acoustic_metadata"] = sha256_file(
            resolve_path(metadata_path)
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    curriculum_type = config.get("curriculum", {}).get("type")
    if curriculum_type not in {"s2s_margin", "wer_margin", "acoustic_s2s_hybrid"}:
        raise ValueError(
            "train_s2s_curriculum.py requires an S2S or WER margin curriculum"
        )
    if args.resume_from_checkpoint is not None:
        raise ValueError("S2S dynamic curriculum resume is not supported")

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    wandb_config = config["wandb"]
    output_dir = resolve_path(args.output_dir or config["output_dir"])
    epochs = args.epochs if args.epochs is not None else training_config["num_train_epochs"]
    seed = args.seed if args.seed is not None else training_config["seed"]
    if int(epochs) != epochs or int(epochs) < 2:
        raise ValueError("S2S curriculum requires at least two complete epochs")

    load_environment(require_api_key=not args.dry_run, wandb_config=wandb_config)
    dataset = load_local_data(data_config)
    evaluation_dataset = load_evaluation_data(data_config)
    train_ids = [str(value) for value in dataset["train"]["id"]]
    train_durations = [float(value) for value in dataset["train"]["duration"]]
    acoustic_scores = None
    acoustic_weight = 0.5
    s2s_weight = 0.5
    if curriculum_type == "acoustic_s2s_hybrid":
        curriculum_config = config["curriculum"]
        metadata_path = resolve_path(curriculum_config["metadata_path"])
        snr_values = load_train_snr(metadata_path, train_ids)
        snr_percentiles = percentile_ranks(snr_values)
        duration_percentiles = percentile_ranks(train_durations)
        acoustic_scores = [
            0.5 * duration_percentile + 0.5 * (1.0 - snr_percentile)
            for duration_percentile, snr_percentile in zip(
                duration_percentiles, snr_percentiles
            )
        ]
        acoustic_weight = float(curriculum_config.get("acoustic_weight", 0.5))
        s2s_weight = float(curriculum_config.get("s2s_weight", 0.5))
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

    probe_dir = output_dir / ".dry_run_dynamic_scores"
    if curriculum_type == "s2s_margin":
        probe_state = S2SLossCurriculumState(
            train_ids, train_durations, seed, probe_dir
        )
    elif curriculum_type == "wer_margin":
        probe_state = WERMarginCurriculumState(
            train_ids, train_durations, seed, probe_dir
        )
    else:
        if acoustic_scores is None:
            raise RuntimeError("Hybrid curriculum is missing acoustic scores")
        probe_state = HybridS2SCurriculumState(
            train_ids,
            train_durations,
            seed,
            probe_dir,
            acoustic_scores,
            acoustic_weight,
            s2s_weight,
        )
    first_order = probe_state.order
    if sorted(first_order) != list(range(len(train_ids))):
        raise RuntimeError("C5 epoch-1 order is not a complete permutation")

    if args.dry_run:
        prepared = prepare_dataset(dataset["train"][first_order[0]])
        print(
            f"{curriculum_type} dry run passed: {len(train_ids)} aligned rows, "
            f"epoch-1 random order hash {order_digest(first_order)[:12]}, "
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
        "epoch_1_order_sha256": order_digest(first_order),
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
    dataset["train"] = dataset["train"].add_column(
        "_curriculum_index", list(range(len(dataset["train"])))
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
        gradient_checkpointing=training_config["gradient_checkpointing"],
        fp16=training_config["fp16"],
        eval_strategy=training_config["eval_strategy"],
        predict_with_generate=training_config["predict_with_generate"],
        generation_max_length=training_config["generation_max_length"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=training_config["save_total_limit"],
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
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": S2SCurriculumCollator(processor),
        "compute_metrics": compute_metrics,
        "processing_class": processor.feature_extractor,
        "curriculum_ids": train_ids,
        "curriculum_durations": train_durations,
        "curriculum_seed": seed,
        "curriculum_artifact_dir": output_dir / "curriculum_scores",
    }
    if curriculum_type in {"s2s_margin", "acoustic_s2s_hybrid"}:
        trainer = S2SLossCurriculumTrainer(
            **trainer_kwargs,
            curriculum_acoustic_scores=acoustic_scores,
            acoustic_weight=acoustic_weight,
            s2s_weight=s2s_weight,
        )
    else:
        trainer = WERMarginCurriculumTrainer(
            **trainer_kwargs,
            curriculum_tokenizer=processor.tokenizer,
            generation_max_length=training_config["generation_max_length"],
        )

    train_result = trainer.train()
    trainer.curriculum_state.finalize_current_epoch()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    expected_files = {
        output_dir / "curriculum_scores" / f"epoch_{epoch:02d}_{kind}.jsonl"
        for epoch in range(1, int(epochs) + 1)
        for kind in ("order", "scores")
    }
    missing_files = [str(path) for path in expected_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"Missing dynamic curriculum artifacts: {missing_files}")

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
    print(f"{curriculum_type} run complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()