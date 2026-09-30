import importlib
import unittest
from pathlib import Path

from asr_experiments.cli import COMMAND_MODULES


ROOT_DIR = Path(__file__).resolve().parents[1]


class CLIArchitectureTests(unittest.TestCase):
    def test_root_contains_only_unified_python_entry_point(self) -> None:
        root_python_files = sorted(path.name for path in ROOT_DIR.glob("*.py"))
        self.assertEqual(root_python_files, ["asr.py"])

    def test_every_registered_command_imports_and_has_main(self) -> None:
        self.assertEqual(len(COMMAND_MODULES), 27)
        for command, module_name in COMMAND_MODULES.items():
            with self.subTest(command=command):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, "main", None)))

    def test_config_runner_identifiers_are_registered_commands(self) -> None:
        for runner in (
            "train_full.py",
            "train_sortagrad.py",
            "train_snr_curriculum.py",
            "train_s2s_curriculum.py",
        ):
            self.assertIn(Path(runner).stem, COMMAND_MODULES)


if __name__ == "__main__":
    unittest.main()