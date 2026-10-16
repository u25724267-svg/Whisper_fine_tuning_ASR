import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "remaining_fleurs", Path(__file__).resolve().parents[1] / "scripts/run_remaining_fleurs.py"
)
remaining = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remaining)


class RemainingFleursTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="remaining fleurs ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = "medium/rq1/example"
        self.archive = "gs://casper/test"
        self.prefix = f"{self.archive}/{self.output}/"
        self.config = {
            "model": {"language": "shona", "task": "transcribe",
                      "clear_forced_decoder_ids": True, "clear_suppress_tokens": True},
            "training": {"per_device_eval_batch_size": 6, "fp16": True,
                         "generation_max_length": 225},
            "data": {},
        }
        self.protocol = {**self.config["model"], **self.config["training"]}
        self.state = {"best_model_checkpoint": "/original/output/checkpoint-4604"}
        self.config_path = self.root / "experiments" / self.output / "config.json"
        remaining.write_json(self.config_path, self.config)
        self.artifact = self.root / "artifacts" / self.output
        remaining.write_json(self.artifact / "experiment_config.json", self.config)
        remaining.write_json(self.artifact / "trainer_state.json", self.state)
        self.listing = []
        for name in (*remaining.REQUIRED, "checkpoint-4604/model.safetensors"):
            self.listing.append({"metadata": {
                "bucket": "casper", "name": f"test/{self.output}/{name}",
                "generation": "123", "size": "100", "crc32c": "same-checksum",
            }})

    def capture(self, arguments):
        return json.dumps(self.listing if arguments[2] == "ls" else self.state)

    def package(self):
        with patch.multiple(remaining, ROOT=self.root, OUTPUT_ROOT=self.root / "outputs",
                            ARTIFACT_ROOT=self.root / "artifacts"), patch.object(
                                remaining, "capture", side_effect=self.capture):
            return remaining.package(self.output, self.archive, self.protocol)

    def test_actual_queue_has_25_unique_primary_runs(self):
        queue = remaining.read_json(remaining.QUEUE)
        outputs = {output for output, _ in remaining.entries(queue)}
        excluded = {output for group in queue["historical_not_in_primary_queue"].values()
                    for output in group}
        self.assertEqual(len(outputs), 25)
        self.assertFalse(outputs.intersection(excluded))

    def test_generation_pinning_and_best_not_last_checkpoint(self):
        result = self.package()
        self.assertEqual(result["best_checkpoint"], "checkpoint-4604")
        self.assertTrue(all(metadata["url"].endswith("#123") for metadata in result["files"].values()))
        self.assertNotIn("checkpoint-4604/model.safetensors", result["files"])

    def test_wrong_best_checkpoint_weights_refused(self):
        self.listing[-1]["metadata"]["crc32c"] = "different"
        with self.assertRaisesRegex(ValueError, "recorded best checkpoint"):
            self.package()

    def test_missing_processor_refused(self):
        self.listing = [item for item in self.listing
                        if not item["metadata"]["name"].endswith("preprocessor_config.json")]
        with self.assertRaisesRegex(ValueError, "Missing archive file"):
            self.package()

    def test_partial_local_restore_refused(self):
        staging = self.root / "outputs" / "medium/rq1/example.fleurs-restore"
        staging.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "partial destination"):
            self.package()

    def test_partial_fleurs_mirror_refused(self):
        (self.artifact / "fleurs_corrected_v2").mkdir()
        with self.assertRaisesRegex(ValueError, "partial destination"):
            self.package()

    def test_changed_scientific_config_refused(self):
        self.config["training"]["generation_max_length"] = 448
        remaining.write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "Scientific config differs"):
            self.package()

    def test_checksum_verification_and_corruption(self):
        data = self.root / "payload"
        data.write_bytes(b"abc")
        metadata = {"size": 3, "crc32c": "correct"}
        with patch.object(remaining, "capture", return_value=json.dumps({"path": {"crc32c_hash": "correct"}})):
            remaining.verify_file(data, metadata)
        with patch.object(remaining, "capture", return_value=json.dumps({"path": {"crc32c_hash": "wrong"}})):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                remaining.verify_file(data, metadata)

    def test_busy_gpu_refused(self):
        with patch.object(remaining, "capture", return_value="GPU-busy\n"):
            with self.assertRaisesRegex(ValueError, "active CUDA process"):
                remaining.require_idle(["GPU-busy"])
            remaining.require_idle(["GPU-other"])

    def test_two_lane_partition_covers_queue_once(self):
        runs = remaining.entries(remaining.read_json(remaining.QUEUE))
        lanes = [runs[lane::2] for lane in (0, 1)]
        self.assertEqual([len(lane) for lane in lanes], [13, 12])
        self.assertEqual(set(lanes[0]).intersection(lanes[1]), set())
        self.assertEqual(set(lanes[0] + lanes[1]), set(runs))


if __name__ == "__main__":
    unittest.main()