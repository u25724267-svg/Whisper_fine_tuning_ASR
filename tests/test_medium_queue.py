import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_experiments.config import PROJECT_ROOT
from asr_experiments.commands.data.prepare_medium_queue import (
    A100_GPU_NAME, MODEL_ID, MODEL_REVISION, a100_control_files,
    derive_config, expected_files, materialize, specifications,
)


class MediumQueueTests(unittest.TestCase):
    def test_a100_controls_change_only_operational_metadata(self) -> None:
        files = a100_control_files(PROJECT_ROOT)
        self.assertEqual(len(files), 16)
        configs = [json.loads(content) for path, content in files.items() if path.name == "config.json"]
        self.assertEqual(len(configs), 5)
        allowed = {"experiment_id", "experiment_name", "control_experiment", "output_dir", "resources", "wandb", "a100_replication"}
        identities = {config["experiment_id"] for config in configs}
        for config in configs:
            source_path = PROJECT_ROOT / config["a100_replication"]["source_config"]
            source = json.loads(source_path.read_text())
            with self.subTest(experiment=config["experiment_id"]):
                self.assertEqual({key: value for key, value in config.items() if key not in allowed},
                                 {key: value for key, value in source.items() if key not in allowed})
                self.assertNotEqual(config["output_dir"], source["output_dir"])
                self.assertEqual(config["resources"]["required_gpu_name"], A100_GPU_NAME)
                self.assertTrue(config["resources"]["require_idle_gpu"])
                self.assertIsNone(config["wandb"]["resume_run_id"])
                self.assertEqual(config["wandb"]["project"], "whisper-shona-multilingual")
                if config["control_experiment"]:
                    self.assertIn(config["control_experiment"], identities)
        queue = json.loads(files[PROJECT_ROOT / "experiments/medium_a100/queue.json"])
        self.assertEqual(queue["estimated_disk_reserve_gib"], 160)
        self.assertEqual(queue["reused_a100_controls"][0]["experiment_id"], "medium-rq1-c1-sortagrad-seed44")
        self.assertNotIn("a100-medium-rq1-c1-sortagrad-seed44", identities)
        self.assertEqual(files, a100_control_files(PROJECT_ROOT))

    def test_a100_materialization_refuses_existing_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "config.json"
            missing = root / "new.json"
            existing.write_text("original")
            with patch("asr_experiments.commands.data.prepare_medium_queue.a100_control_files",
                       return_value={missing: "new", existing: "changed"}):
                with self.assertRaises(FileExistsError):
                    materialize(root, a100_controls=True)
            self.assertFalse(missing.exists())
            self.assertEqual(existing.read_text(), "original")

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


class MediumGPULauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="medium gpu ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("run_medium_rq1_rq2.sh", "run_sequence.sh", "run_experiment.sh"):
            shutil.copy2(PROJECT_ROOT / "scripts" / name, scripts / name)
        interpreter = self.root / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        self.experiments = [f"experiments/run{index:02d}" for index in range(27)]
        config = {
            "experiment_name": "run00", "output_dir": "run00",
            "resources": {"min_free_gpu_mb": 20000, "min_free_disk_gb": 0},
        }
        for experiment in self.experiments:
            directory = self.root / experiment
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(config))
            (directory / "run.sh").touch()
            (directory / "run.sh").chmod(0o755)
        queue = self.root / "experiments/medium/queue.json"
        queue.parent.mkdir()
        queue.write_text(json.dumps({"estimated_disk_reserve_gib": 0}))
        (self.root / "asr.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            f"if sys.argv[1] == 'prepare_medium_queue': print('\\n'.join({self.experiments!r}[:5] if '--a100-controls' in sys.argv else {self.experiments!r}))\n"
            "elif sys.argv[1] == 'mirror_experiment_artifacts':\n"
            "    target = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    (target / 'artifact_manifest.json').write_text('{}')\n"
        )
        tools = self.root / "tools"
        tools.mkdir()
        (tools / "nvidia-smi").write_text(
            "#!/usr/bin/env python3\nimport os, sys\n"
            "if '--query-gpu=uuid' in sys.argv:\n"
            "    selected = [argument.split('=', 1)[1] for argument in sys.argv if argument.startswith('--id=')]\n"
            "    print(selected[0] if selected else '\\n'.join(f'GPU-test-{index}' for index in range(int(os.getenv('MOCK_GPU_COUNT', '2')))))\n"
            "elif '--query-gpu=name' in sys.argv: print(os.getenv('MOCK_GPU_NAME', 'NVIDIA A100-SXM4-40GB'))\n"
            "elif '--query-compute-apps=gpu_uuid' in sys.argv: print(os.getenv('MOCK_BUSY_GPU', ''))\n"
            "else:\n"
            "    key = 'MOCK_GPU1_FREE' if '--id=GPU-test-1' in sys.argv else 'MOCK_GPU0_FREE'\n"
            "    print(os.getenv(key, '40000'))\n"
        )
        (tools / "tmux").write_text(
            "#!/usr/bin/env python3\nimport json, os, shlex, sys\nfrom pathlib import Path\n"
            "state = Path(os.environ['MOCK_STATE_DIR'])\n"
            "with (state / 'calls.jsonl').open('a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] == 'has-session': sys.exit(0 if (state / sys.argv[-1]).exists() else 1)\n"
            "if sys.argv[1] == 'display-message': print('1' if sys.argv[-1] == '#{pane_dead}' else '0')\n"
            "if sys.argv[1] == 'new-session':\n"
            "    (state / sys.argv[sys.argv.index('-s') + 1]).touch()\n"
            "    tokens = shlex.split(sys.argv[-1])\n"
            "    if 'evaluate_predictions' in tokens:\n"
            "        target = Path(tokens[tokens.index('--output-dir') + 1])\n"
            "        target.mkdir(parents=True)\n"
            "        (target / 'summary.json').write_text('{}')\n"
        )
        for tool in tools.iterdir():
            tool.chmod(0o755)
        self.state = self.root / "state"
        self.state.mkdir()
        self.environment = {
            **os.environ,
            "PATH": f"{tools}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "ASR_OUTPUT_ROOT": str(self.root / "outputs"),
            "MOCK_STATE_DIR": str(self.state),
            "CUDA_VISIBLE_DEVICES": "GPU-test-1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def run_script(self, name: str, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.root / "scripts" / name), *arguments],
            env=self.environment, capture_output=True, text=True, timeout=20,
        )

    def launches(self) -> list[list[str]]:
        calls = self.state / "calls.jsonl"
        return [
            call for line in calls.read_text().splitlines()
            if (call := json.loads(line))[0] == "new-session"
        ] if calls.exists() else []

    def test_controller_partitions_all_runs_into_two_single_gpu_lanes(self) -> None:
        result = self.run_script("run_medium_rq1_rq2.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = self.launches()
        self.assertEqual(len(launches), 2)
        for gpu_index, launch in enumerate(launches):
            self.assertEqual(launch[launch.index("-s") + 1], f"whisper-medium-rq1-rq2-gpu{gpu_index}")
            tokens = shlex.split(launch[-1])
            self.assertIn(f"CUDA_VISIBLE_DEVICES=GPU-test-{gpu_index}", tokens)
            assigned = [token for token in tokens if token.startswith("experiments/")]
            self.assertEqual(assigned, self.experiments[gpu_index::2])

    def test_preflight_and_gpu_admission_never_launch(self) -> None:
        result = self.run_script("run_medium_rq1_rq2.sh", "--preflight-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.launches(), [])
        for overrides in ({"MOCK_GPU_COUNT": "1"}, {"MOCK_GPU1_FREE": "19999"}):
            with self.subTest(overrides=overrides), patch.dict(self.environment, overrides):
                result = self.run_script("run_medium_rq1_rq2.sh")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.launches(), [])

    def test_training_checks_and_pins_assigned_gpu_not_gpu_zero(self) -> None:
        self.environment["MOCK_GPU0_FREE"] = "0"
        result = self.run_script("run_experiment.sh", str(self.root / self.experiments[0]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.launches()), 1)
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-test-1", shlex.split(self.launches()[0][-1]))

    def test_a100_queue_partitions_five_and_rejects_wrong_or_busy_gpu(self) -> None:
        queue = self.root / "experiments/medium_a100/queue.json"
        queue.parent.mkdir()
        queue.write_text(json.dumps({"estimated_disk_reserve_gib": 0}))
        for overrides, message in (({"MOCK_GPU_NAME": "NVIDIA GeForce RTX 4090"}, "require"),
                                   ({"MOCK_BUSY_GPU": "GPU-test-1"}, "active compute")):
            with self.subTest(overrides=overrides), patch.dict(self.environment, overrides):
                result = self.run_script("run_medium_rq1_rq2.sh", "--a100-controls", "--preflight-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.launches(), [])
        result = self.run_script("run_medium_rq1_rq2.sh", "--a100-controls")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.launches()), 2)
        for index, launch in enumerate(self.launches()):
            self.assertEqual(launch[launch.index("-s") + 1], f"whisper-medium-a100-controls-gpu{index}")
            tokens = shlex.split(launch[-1])
            self.assertIn(f"CUDA_VISIBLE_DEVICES=GPU-test-{index}", tokens)
            self.assertEqual([token for token in tokens if token.startswith("experiments/")], self.experiments[:5][index::2])

    def test_individual_a100_run_requires_idle_selected_device(self) -> None:
        path = self.root / self.experiments[0] / "config.json"
        config = json.loads(path.read_text())
        config["resources"].update(required_gpu_name=A100_GPU_NAME, require_idle_gpu=True)
        path.write_text(json.dumps(config))
        for overrides, message in (({"MOCK_GPU_NAME": "NVIDIA GeForce RTX 4090"}, "Require GPU"),
                                   ({"MOCK_BUSY_GPU": "GPU-test-1"}, "active compute")):
            with self.subTest(overrides=overrides), patch.dict(self.environment, overrides):
                result = self.run_script("run_experiment.sh", str(path.parent))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.launches(), [])
                self.assertFalse((self.root / "outputs/run00").exists())
            environment_file = self.root / ".env"
            environment_file.write_text("CUDA_VISIBLE_DEVICES=0,1\n")
            with patch.dict(self.environment, {"PYTHONPATH": os.pathsep.join(sys.path)}):
                result = self.run_script("run_experiment.sh", str(path.parent))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Remove CUDA_VISIBLE_DEVICES from .env", result.stderr)
            self.assertEqual(self.launches(), [])
            environment_file.unlink()
        result = self.run_script("run_experiment.sh", str(path.parent))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--resume-from-checkpoint", shlex.split(self.launches()[0][-1]))

    def test_prediction_export_keeps_lane_gpu(self) -> None:
        output = self.root / "outputs/run00"
        (output / "logs").mkdir(parents=True)
        for name in ("train_results.json", "validation_results.json", "test_results.json", "model.safetensors"):
            (output / name).touch()
        result = self.run_script("run_sequence.sh", self.experiments[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.launches()), 1)
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-test-1", shlex.split(self.launches()[0][-1]))


if __name__ == "__main__":
    unittest.main()