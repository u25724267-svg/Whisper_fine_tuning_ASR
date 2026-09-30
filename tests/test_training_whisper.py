import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from asr_experiments.training.whisper import (
    build_compute_metrics,
    build_prepare_dataset,
    build_training_arguments,
    load_model,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[len(text)])

    def batch_decode(self, values: np.ndarray, skip_special_tokens: bool) -> list[str]:
        _ = skip_special_tokens
        return [" ".join(map(str, row)) for row in values.tolist()]


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def feature_extractor(
        self, array: list[float], sampling_rate: int, return_attention_mask: bool
    ) -> SimpleNamespace:
        _ = return_attention_mask
        return SimpleNamespace(input_features=[[array]], attention_mask=[[sampling_rate]])


class TrainingWhisperTests(unittest.TestCase):
    def training_config(self) -> dict[str, object]:
        return {
            "per_device_train_batch_size": 6,
            "per_device_eval_batch_size": 6,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "warmup_steps": 500,
            "gradient_checkpointing": True,
            "fp16": False,
            "eval_strategy": "epoch",
            "predict_with_generate": True,
            "generation_max_length": 225,
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "logging_steps": 250,
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "full_determinism": True,
        }

    def test_training_arguments_preserve_standard_defaults(self) -> None:
        arguments = build_training_arguments(
            Path("output"), self.training_config(), 3.0, 42, "run"
        )

        self.assertEqual(arguments.max_steps, -1)
        self.assertEqual(arguments.save_steps, 500)
        self.assertFalse(arguments.save_only_model)
        self.assertTrue(arguments.remove_unused_columns)
        self.assertEqual(arguments.seed, 42)
        self.assertEqual(arguments.data_seed, 42)

    def test_dynamic_training_arguments_preserve_curriculum_index(self) -> None:
        arguments = build_training_arguments(
            Path("output"),
            self.training_config(),
            3.0,
            42,
            "run",
            preserve_curriculum_index=True,
        )

        self.assertFalse(arguments.remove_unused_columns)

    def test_prepare_dataset_preserves_training_callback_contract(self) -> None:
        callback = build_prepare_dataset(
            FakeProcessor(), {"audio_column": "audio", "text_column": "text"}
        )

        prepared = callback(
            {"audio": {"array": [0.25], "sampling_rate": 16_000}, "text": "mhoro"}
        )

        self.assertEqual(prepared["input_features"], [[0.25]])
        self.assertEqual(prepared["attention_mask"], [16_000])
        self.assertEqual(prepared["labels"], [5])

    def test_compute_metrics_replaces_ignored_labels_and_scales_wer(self) -> None:
        processor = FakeProcessor()
        metric = SimpleNamespace(compute=lambda **_: 0.25)
        callback = build_compute_metrics(processor, metric)
        prediction = SimpleNamespace(
            predictions=np.asarray([[1, 2]]),
            label_ids=np.asarray([[1, -100]]),
        )

        result = callback(prediction)

        self.assertEqual(result, {"wer": 25.0})
        self.assertEqual(prediction.label_ids.tolist(), [[1, 0]])

    def test_load_model_applies_generation_contract(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(forced_decoder_ids=[1], suppress_tokens=[2]),
            generation_config=SimpleNamespace(
                forced_decoder_ids=[1], suppress_tokens=[2], language=None, task=None
            ),
        )
        config = {
            "id": "model",
            "revision": "revision",
            "language": "shona",
            "task": "transcribe",
            "clear_forced_decoder_ids": True,
            "clear_suppress_tokens": True,
        }

        with patch(
            "asr_experiments.training.whisper.WhisperForConditionalGeneration.from_pretrained",
            return_value=model,
        ) as loader:
            loaded = load_model(config)

        self.assertIs(loaded, model)
        loader.assert_called_once_with("model", revision="revision")
        self.assertIsNone(model.config.forced_decoder_ids)
        self.assertEqual(model.config.suppress_tokens, [])
        self.assertEqual(model.generation_config.language, "shona")
        self.assertEqual(model.generation_config.task, "transcribe")


if __name__ == "__main__":
    unittest.main()