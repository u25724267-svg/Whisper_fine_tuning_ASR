import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_experiments.config import PROJECT_ROOT
from asr_experiments.commands.data.prepare_medium_queue import (
    MODEL_ID, MODEL_REVISION, derive_config, expected_files, materialize, specifications,
)


class MediumQueueTests(unittest.TestCase):
    def test_all_27_configs_preserve_scientific_contracts(self) -> None:
        sources = specifications(PROJECT_ROOT)
        self.assertEqual(len(sources), 27)
        identities = {source["experiment_id"]: f"medium-{source['experiment_id']}" for _, source in sources}
        for path, source in sources:
            with self.subTest(source=path):
                original = copy.deepcopy(source)
                config = derive_config(source, path, PROJECT_ROOT, identities)
                self.assertEqual(source, original)
                for key in ("data", "curriculum", "augmentation", "waveform_augmentation"):
                    self.assertEqual(config.get(key), source.get(key))
                self.assertEqual(config["training"], source["training"])
                self.assertEqual(config["runner"], source.get("runner", "train_full.py"))
                self.assertEqual(config["model"], {**source["model"], "id": MODEL_ID, "revision": MODEL_REVISION})
                self.assertEqual(config["wandb"]["project"], "whisper-shona-multilingual")
                self.assertNotIn("whisper-base", config["wandb"]["tags"])
                if config["control_experiment"]:
                    self.assertIn(config["control_experiment"], identities.values())

    def test_batch_settings_are_inherited_not_hardcoded(self) -> None:
        path, source = specifications(PROJECT_ROOT)[0]
        source["training"].update(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            gradient_accumulation_steps=2,
        )
        config = derive_config(
            source, path, PROJECT_ROOT, {source["experiment_id"]: "medium-test"}
        )
        self.assertEqual(config["training"], source["training"])

    def test_only_model_and_operational_metadata_may_differ(self) -> None:
        sources = specifications(PROJECT_ROOT)
        identities = {source["experiment_id"]: f"medium-{source['experiment_id']}" for _, source in sources}
        operational_keys = {
            "experiment_id", "experiment_name", "control_experiment", "output_dir",
            "resources", "wandb", "medium_extension",
        }
        for path, source in sources:
            with self.subTest(source=path):
                config = derive_config(source, path, PROJECT_ROOT, identities)
                expected = copy.deepcopy(source)
                expected.setdefault("runner", "train_full.py")
                config["model"]["id"] = expected["model"]["id"]
                config["model"]["revision"] = expected["model"]["revision"]
                for key in operational_keys:
                    config.pop(key, None)
                    expected.pop(key, None)
                self.assertEqual(config, expected)

    def test_unknown_control_is_rejected(self) -> None:
        path, source = specifications(PROJECT_ROOT)[0]
        source["control_experiment"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unmapped Base control"):
            derive_config(source, path, PROJECT_ROOT, {source["experiment_id"]: "medium-test"})

    def test_generation_is_deterministic_and_does_not_write(self) -> None:
        self.assertEqual(expected_files(PROJECT_ROOT), expected_files(PROJECT_ROOT))
        self.assertEqual(len(expected_files(PROJECT_ROOT)), 82)

    def test_control_order_and_unique_destinations(self) -> None:
        import json

        content = expected_files(PROJECT_ROOT)[PROJECT_ROOT / "experiments/medium/queue.json"]
        queue = json.loads(content)
        seen = set()
        for entry in queue["runs"]:
            if entry["control_experiment"]:
                self.assertIn(entry["control_experiment"], seen)
            self.assertNotIn(entry["experiment_id"], seen)
            seen.add(entry["experiment_id"])
        self.assertEqual(len({entry["output_dir"] for entry in queue["runs"]}), 27)

    def test_materialization_refuses_drift_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "config.json"
            missing = root / "new.json"
            existing.write_text("original", encoding="utf-8")
            with patch(
                "asr_experiments.commands.data.prepare_medium_queue.expected_files",
                return_value={missing: "new", existing: "changed"},
            ):
                with self.assertRaises(FileExistsError):
                    materialize(root)
            self.assertFalse(missing.exists())
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")

    def test_check_only_never_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "config.json"
            with patch(
                "asr_experiments.commands.data.prepare_medium_queue.expected_files",
                return_value={missing: "{}"},
            ):
                with self.assertRaises(FileNotFoundError):
                    materialize(root, check_only=True)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()