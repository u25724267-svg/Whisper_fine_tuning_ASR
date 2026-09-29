import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mirror_experiment_artifacts import mirror_experiment


class MirrorExperimentArtifactsTests(unittest.TestCase):
    def create_complete_source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        for relative_path in (
            "model.safetensors",
            "train_results.json",
            "validation_results.json",
            "test_results.json",
            "item_predictions/summary.json",
            "item_predictions/validation.jsonl",
            "logs/train.log",
            "curriculum_orders/epoch_01_order.jsonl",
        ):
            path = source / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"content:{relative_path}\n", encoding="utf-8")
        (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        checkpoint = source / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        return source

    def test_mirrors_results_and_excludes_model_assets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "artifacts" / "run"

            result = mirror_experiment(source, output)

            self.assertEqual(result["status"], "mirrored")
            self.assertTrue((output / "train_results.json").is_file())
            self.assertTrue((output / "item_predictions/validation.jsonl").is_file())
            self.assertTrue((output / "logs/train.log").is_file())
            self.assertFalse((output / "model.safetensors").exists())
            self.assertFalse((output / "tokenizer_config.json").exists())
            self.assertFalse((output / "checkpoint-1").exists())
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["files"], 7)

    def test_incomplete_source_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            (source / "item_predictions/summary.json").unlink()

            with self.assertRaisesRegex(ValueError, "incomplete"):
                mirror_experiment(source, root / "output")

    def test_matching_existing_mirror_is_idempotent(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "output"
            mirror_experiment(source, output)

            result = mirror_experiment(source, output)

            self.assertEqual(result["status"], "already-mirrored")