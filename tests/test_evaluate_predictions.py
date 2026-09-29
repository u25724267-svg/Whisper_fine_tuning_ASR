import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate_predictions import resolve_model_source


MODEL_CONFIG = {
    "id": "openai/whisper-base",
    "revision": "pinned-revision",
}


class ModelSourceTests(unittest.TestCase):
    def test_hub_model_uses_config_revision_by_default(self) -> None:
        model_dir, model_source, revision = resolve_model_source(
            None,
            "openai/whisper-base",
            None,
            MODEL_CONFIG,
        )

        self.assertIsNone(model_dir)
        self.assertEqual(model_source, "openai/whisper-base")
        self.assertEqual(revision, "pinned-revision")

    def test_explicit_hub_revision_overrides_config(self) -> None:
        _, _, revision = resolve_model_source(
            None,
            "openai/whisper-base",
            "explicit-revision",
            MODEL_CONFIG,
        )

        self.assertEqual(revision, "explicit-revision")

    def test_trained_directory_remains_supported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "checkpoint"
            model_dir.mkdir()
            (model_dir / "model.safetensors").touch()

            resolved_dir, model_source, revision = resolve_model_source(
                model_dir,
                None,
                None,
                MODEL_CONFIG,
            )

            self.assertEqual(resolved_dir, model_dir.resolve())
            self.assertEqual(model_source, str(model_dir.resolve()))
            self.assertIsNone(revision)

    def test_missing_model_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model directory or model ID"):
            resolve_model_source(None, None, None, MODEL_CONFIG)