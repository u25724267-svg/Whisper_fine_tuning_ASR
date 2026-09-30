import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr_experiments.commands.artifacts.mirror_experiment_artifacts import (
    mirror_experiment,
)


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

    def test_refresh_adds_new_results_without_model_assets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "output"
            mirror_experiment(source, output)
            fleurs = source / "fleurs_corrected_v2" / "summary.json"
            fleurs.parent.mkdir()
            fleurs.write_text('{"wer": 42.0}\n', encoding="utf-8")

            result = mirror_experiment(source, output, refresh=True)

            self.assertEqual(result["status"], "refreshed")
            self.assertTrue((output / "fleurs_corrected_v2/summary.json").is_file())
            self.assertTrue((output / "item_predictions/summary.json").is_file())
            self.assertFalse((output / "model.safetensors").exists())
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            paths = {entry["path"] for entry in manifest["entries"]}
            self.assertIn("fleurs_corrected_v2/summary.json", paths)

    def test_refresh_requires_matching_manifest_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "output"
            mirror_experiment(source, output)
            manifest_path = output / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_dir"] = "/different/source"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source does not match"):
                mirror_experiment(source, output, refresh=True)

    def test_refresh_adopts_matching_legacy_mirror(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "output"
            output.mkdir()
            (output / "train_results.json").write_text(
                (source / "train_results.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            result = mirror_experiment(source, output, refresh=True)

            self.assertEqual(result["status"], "refreshed")
            self.assertTrue((output / "artifact_manifest.json").is_file())
            self.assertTrue((output / "item_predictions/summary.json").is_file())

    def test_refresh_rejects_divergent_legacy_mirror(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "output"
            output.mkdir()
            (output / "unexpected.txt").write_text("unknown\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unmanaged mirror"):
                mirror_experiment(source, output, refresh=True)

    def test_refresh_preserves_legacy_wandb_logs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.create_complete_source(root)
            output = root / "output"
            output.mkdir()
            legacy_log = output / "wandb/run/logs/debug.log"
            legacy_log.parent.mkdir(parents=True)
            legacy_log.write_text("historical\n", encoding="utf-8")

            result = mirror_experiment(source, output, refresh=True)

            self.assertEqual(result["status"], "refreshed")
            self.assertEqual(legacy_log.read_text(encoding="utf-8"), "historical\n")