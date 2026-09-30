import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from asr_experiments.config import PROJECT_ROOT, load_config, resolve_path


class ConfigTests(unittest.TestCase):
    def test_relative_paths_resolve_from_project_root(self) -> None:
        self.assertEqual(
            resolve_path("configs/wandb-projects.json"),
            PROJECT_ROOT / "configs/wandb-projects.json",
        )

    def test_environment_variables_expand_before_resolution(self) -> None:
        with patch.dict(os.environ, {"ASR_CONFIG_DIR": "configs"}):
            self.assertEqual(
                resolve_path("$ASR_CONFIG_DIR/wandb-projects.json"),
                PROJECT_ROOT / "configs/wandb-projects.json",
            )

    def test_load_config_records_resolved_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text('{"experiment_name": "test"}\n', encoding="utf-8")

            config = load_config(path)

            self.assertEqual(config["experiment_name"], "test")
            self.assertEqual(config["config_path"], str(path.resolve()))

    def test_load_config_reports_resolved_missing_path(self) -> None:
        missing = PROJECT_ROOT / "missing-config.json"
        with self.assertRaisesRegex(FileNotFoundError, str(missing)):
            load_config(Path("missing-config.json"))


if __name__ == "__main__":
    unittest.main()