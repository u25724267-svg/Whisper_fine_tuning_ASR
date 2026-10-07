"""Bounded CUDA feasibility probe; never produces an experiment checkpoint."""

import argparse
import copy
import importlib.metadata
import json
import os
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, Dataset
from transformers import Seq2SeqTrainer, TrainerCallback, enable_full_determinism

from asr_experiments.config import PROJECT_ROOT
from asr_experiments.provenance import sha256_file
from asr_experiments.training.whisper import (
    build_prepare_dataset, build_training_arguments, load_model, load_processor,
)
from asr_experiments.commands.training.train_full import (
    DataCollatorSpeechSeq2SeqWithPadding, configure_augmentation,
)


def select_longest(rows: list[dict[str, Any]], lengths: list[int], count: int) -> list[int]:
    """Select long decoder sequences deterministically, without touching audio."""
    if len(rows) != len(lengths) or len(rows) < count or count < 1:
        raise ValueError("Need aligned rows/lengths and at least one complete batch")
    return sorted(range(len(rows)), key=lambda index: (-lengths[index], str(rows[index]["id"])))[:count]


def diagnostic_training_config(config: dict[str, Any], steps: int) -> dict[str, Any]:
    """Bound duration and suppress checkpointing, leaving optimization unchanged."""
    if not 2 <= steps <= 32:
        raise ValueError("Memory probe requires 2 to 32 attempted optimizer steps")
    result = copy.deepcopy(config)
    result.update(max_steps=steps, eval_strategy="no", save_strategy="no", load_best_model_at_end=False)
    return result


def memory_snapshot() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device_free_mib": free / 2**20,
        "device_total_mib": total / 2**20,
    }


class MemoryCallback(TrainerCallback):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        torch.cuda.synchronize()
        state_entries = len(optimizer.state) if optimizer is not None else 0
        skipped = bool(getattr(optimizer, "step_was_skipped", False))
        scaler = getattr(optimizer, "scaler", None)
        if state_entries and not skipped:
            self.report["successful_optimizer_steps"] += 1
        self.report["steps"].append({
            "global_step": state.global_step,
            "optimizer_state_entries": state_entries,
            "optimizer_step_skipped": skipped,
            "grad_scaler_scale": scaler.get_scale() if scaler is not None else None,
            **memory_snapshot(),
        })
        if self.report["successful_optimizer_steps"] >= 2:
            control.should_training_stop = True
        return control


def prepare_batch(config, processor, split, count, report):
    path = Path(config["data"][f"{split}_manifest"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenized = processor.tokenizer([row[config["data"]["text_column"]] for row in rows])
    lengths = [len(tokens) for tokens in tokenized.input_ids]
    indices = select_longest(rows, lengths, count)
    selected = [rows[index] for index in indices]
    report["samples"][split] = {
        "manifest": str(path), "manifest_sha256": sha256_file(path),
        "selection": "longest tokenized transcripts; id tie-break",
        "ids": [row["id"] for row in selected],
        "label_lengths": [lengths[index] for index in indices],
        "durations": [row.get("duration") for row in selected],
    }
    dataset = Dataset.from_list(selected).cast_column(
        config["data"]["audio_column"], Audio(sampling_rate=config["data"]["sampling_rate"])
    )
    return dataset.map(
        build_prepare_dataset(processor, config["data"]),
        remove_columns=dataset.column_names, keep_in_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16, help="Maximum attempts; stop after two non-skipped updates (2-32).")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = diagnostic_training_config(config["training"], args.steps)
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Diagnostic report already exists: {report_path}")
    if config["model"]["id"] != "openai/whisper-medium":
        raise ValueError("This probe is restricted to Whisper Medium")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Memory probe requires exactly one visible CUDA device")
    os.environ["WANDB_DISABLED"] = "true"
    enable_full_determinism(training["seed"])
    report: dict[str, Any] = {
        "schema_version": 1, "diagnostic_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "model": config["model"], "training_contract": config["training"],
        "augmentation": config["augmentation"],
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
        "initial_memory": memory_snapshot(), "steps": [], "samples": {},
        "successful_optimizer_steps": 0,
        "diagnostic_overrides": {
            "max_steps": args.steps, "eval_strategy": "no", "save_strategy": "no",
            "load_best_model_at_end": False, "report_to": [],
            "stop_after_non_skipped_optimizer_steps": 2,
            "sample_selection": "repeat the six longest train transcripts, then longest validation batch",
            "generation_stress_min_length": config["training"]["generation_max_length"],
        },
        "limitations": "Small stress probe, not full-run feasibility or ASR quality evaluation; no checkpoint saved.",
        "stage": "preparation", "status": "running",
    }
    report["initial_compute_processes"] = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    failure = False
    try:
        processor = load_processor(config["model"])
        train_batch = prepare_batch(config, processor, "train", training["per_device_train_batch_size"], report)
        eval_batch = prepare_batch(config, processor, "validation", training["per_device_eval_batch_size"], report)
        train_data = train_batch.select(list(range(len(train_batch))) * args.steps * training["gradient_accumulation_steps"])
        with tempfile.TemporaryDirectory(prefix="whisper-medium-memory-") as temporary:
            trainer_args = build_training_arguments(
                Path(temporary), training, training["num_train_epochs"], training["seed"], "memory-probe-only",
            )
            trainer_args.report_to = []
            trainer_args.disable_tqdm = True
            report["stage"] = "model_loading"
            model = load_model(config["model"])
            configure_augmentation(model, config["augmentation"])
            trainer = Seq2SeqTrainer(
                model=model, args=trainer_args, train_dataset=train_data,
                data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
                processing_class=processor.feature_extractor, callbacks=[MemoryCallback(report)],
            )
            report["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
            report["stage"] = "training_forward_backward_optimizer"
            torch.cuda.reset_peak_memory_stats()
            trainer.train()
            report["training_memory"] = memory_snapshot()
            if report["successful_optimizer_steps"] < 2:
                raise RuntimeError("Fewer than two non-skipped optimizer steps; probe is inconclusive")
            report["stage"] = "generation_evaluation"
            torch.cuda.reset_peak_memory_stats()
            prediction = trainer.predict(eval_batch)
            torch.cuda.synchronize()
            report["evaluation_memory"] = memory_snapshot()
            report["evaluation_prediction_shape"] = list(prediction.predictions.shape)
            del prediction
            report["stage"] = "full_length_generation_stress"
            torch.cuda.reset_peak_memory_stats()
            prediction = trainer.predict(eval_batch, min_length=training["generation_max_length"])
            torch.cuda.synchronize()
            report["generation_stress_memory"] = memory_snapshot()
            report["generation_stress_prediction_shape"] = list(prediction.predictions.shape)
            report["status"] = "passed"
            report["stage"] = "complete"
    except Exception as error:
        failure = True
        report["status"] = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        report["failure_memory"] = memory_snapshot()
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()