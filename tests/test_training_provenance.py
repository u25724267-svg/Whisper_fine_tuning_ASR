import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr_experiments.training.provenance import add_curriculum_provenance


class TrainingProvenanceTests(unittest.TestCase):
    def test_adds_curriculum_runner_and_named_component_hashes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "run"
            output_dir.mkdir()
            manifest_path = output_dir / "run_manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "sha256": {}}) + "\n",
                encoding="utf-8",
            )
            runner = root / "runner.py"
            sampler = root / "sampler.py"
            runner.write_text("runner\n", encoding="utf-8")
            sampler.write_text("sampler\n", encoding="utf-8")

            add_curriculum_provenance(
                output_dir,
                {"type": "sortagrad"},
                runner,
                {"curriculum_sampler": sampler},
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["curriculum"], {"type": "sortagrad"})
            self.assertEqual(
                manifest["sha256"]["curriculum_runner"],
                hashlib.sha256(b"runner\n").hexdigest(),
            )
            self.assertEqual(
                manifest["sha256"]["curriculum_sampler"],
                hashlib.sha256(b"sampler\n").hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()