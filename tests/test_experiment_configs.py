import json
import unittest
from pathlib import Path, PurePosixPath


ROOT_DIR = Path(__file__).resolve().parents[1]
ALLOWED_RUNNERS = {
    "train_full.py",
    "train_s2s_curriculum.py",
    "train_snr_curriculum.py",
    "train_sortagrad.py",
}


class ExperimentConfigContractTests(unittest.TestCase):
    def test_experiment_identity_output_and_runner_contracts(self) -> None:
        config_paths = sorted((ROOT_DIR / "experiments" / "rq1").glob("*/config.json"))
        config_paths.extend(
            sorted((ROOT_DIR / "experiments" / "rq2").glob("*/config.json"))
        )
        self.assertEqual(len(config_paths), 42)
        experiment_names: set[str] = set()
        output_paths: set[str] = set()

        for config_path in config_paths:
            with self.subTest(config=str(config_path.relative_to(ROOT_DIR))):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                experiment_name = str(config["experiment_name"])
                output_path = PurePosixPath(str(config["output_dir"]))
                runner = str(config.get("runner", "train_full.py"))

                self.assertTrue(experiment_name)
                self.assertNotIn(experiment_name, experiment_names)
                self.assertFalse(output_path.is_absolute())
                self.assertNotIn("..", output_path.parts)
                self.assertNotIn(str(output_path), {"", "."})
                self.assertNotIn(str(output_path), output_paths)
                self.assertIn(runner, ALLOWED_RUNNERS)

                experiment_names.add(experiment_name)
                output_paths.add(str(output_path))

    def test_every_experiment_retains_compatibility_launcher(self) -> None:
        config_paths = sorted((ROOT_DIR / "experiments" / "rq1").glob("*/config.json"))
        config_paths.extend(
            sorted((ROOT_DIR / "experiments" / "rq2").glob("*/config.json"))
        )
        non_executable = []
        for config_path in config_paths:
            launcher = config_path.with_name("run.sh")
            self.assertTrue(launcher.is_file(), launcher)
            if not launcher.stat().st_mode & 0o111:
                non_executable.append(launcher.relative_to(ROOT_DIR).as_posix())
        self.assertEqual(non_executable, ["experiments/rq1/c2_snr_seed42/run.sh"])


if __name__ == "__main__":
    unittest.main()