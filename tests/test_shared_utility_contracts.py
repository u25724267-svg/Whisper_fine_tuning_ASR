import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr_experiments.commands.analysis.aggregate_rq2_factorial import (
    sha256_file as aggregate_sha256_file,
)
from asr_experiments.commands.artifacts.mirror_experiment_artifacts import (
    sha256_file as mirror_sha256_file,
)
from asr_experiments.commands.data.inventory_waxal_unlabeled import (
    sha256_file as inventory_sha256_file,
)
from asr_experiments.commands.data.prepare_rq2_asset_partitions import (
    write_jsonl as replaceable_write_jsonl,
)
from asr_experiments.commands.rq4.prepare_rq4_admission import (
    write_jsonl as immutable_write_jsonl,
)
from asr_experiments.provenance import ordered_indices_sha256


class SharedUtilityContractTests(unittest.TestCase):
    def test_ordered_indices_hash_preserves_order_and_encoding(self) -> None:
        expected = hashlib.sha256(b"3,1,2").hexdigest()
        self.assertEqual(ordered_indices_sha256([3, 1, 2]), expected)
        self.assertNotEqual(
            ordered_indices_sha256([3, 1, 2]), ordered_indices_sha256([1, 2, 3])
        )

    def test_sha256_implementations_match_hashlib(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.bin"
            payload = b"reproducible-asr\x00payload"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            self.assertEqual(aggregate_sha256_file(path), expected)
            self.assertEqual(inventory_sha256_file(path), expected)
            self.assertEqual(mirror_sha256_file(path), expected)

    def test_jsonl_writers_have_distinct_overwrite_contracts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            immutable_path = root / "immutable.jsonl"
            replaceable_path = root / "replaceable.jsonl"

            immutable_write_jsonl(immutable_path, [{"version": 1}])
            with self.assertRaises(FileExistsError):
                immutable_write_jsonl(immutable_path, [{"version": 2}])

            replaceable_write_jsonl(replaceable_path, [{"version": 1}])
            replaceable_write_jsonl(replaceable_path, [{"version": 2}])
            rows = [
                json.loads(line)
                for line in replaceable_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows, [{"version": 2}])


if __name__ == "__main__":
    unittest.main()