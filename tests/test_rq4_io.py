import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr_experiments.rq4.io import atomic_write_json, atomic_write_jsonl


class RQ4IOTests(unittest.TestCase):
    def test_atomic_writers_replace_outputs_and_leave_no_temporary_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            json_path = root / "summary.json"
            jsonl_path = root / "predictions.jsonl"

            atomic_write_json(json_path, {"version": 1})
            atomic_write_json(json_path, {"version": 2})
            atomic_write_jsonl(jsonl_path, [{"version": 1}])
            atomic_write_jsonl(jsonl_path, [{"version": 2}])

            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")), {"version": 2}
            )
            self.assertEqual(
                [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                ],
                [{"version": 2}],
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()