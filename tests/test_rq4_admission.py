import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from asr_experiments.commands.rq4.prepare_rq4_admission import (
    canonical_pcm_hash,
    load_hashes,
    load_jsonl,
    write_json,
    write_jsonl,
)


class RQ4AdmissionUtilityTests(unittest.TestCase):
    def test_canonical_pcm_hash_is_stable_for_identical_audio(self) -> None:
        waveform = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
        audio = io.BytesIO()
        sf.write(audio, waveform, 16_000, format="WAV", subtype="PCM_16")
        audio_bytes = audio.getvalue()

        first = canonical_pcm_hash(audio_bytes)
        second = canonical_pcm_hash(audio_bytes)

        self.assertEqual(first, second)
        self.assertEqual(first["sample_rate"], 16_000)
        self.assertEqual(first["channels"], 1)
        self.assertEqual(first["frames"], 4)
        self.assertAlmostEqual(first["duration"], 4 / 16_000)

    def test_json_writers_are_immutable_and_leave_no_temporary_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jsonl_path = root / "rows.jsonl"
            json_path = root / "summary.json"

            write_jsonl(jsonl_path, [{"text": "Shona: mhoro"}])
            write_json(json_path, {"rows": 1})

            self.assertEqual(load_jsonl(jsonl_path), [{"text": "Shona: mhoro"}])
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")), {"rows": 1}
            )
            self.assertFalse((root / "rows.jsonl.tmp").exists())
            self.assertFalse((root / "summary.json.tmp").exists())
            with self.assertRaisesRegex(FileExistsError, "immutable output"):
                write_jsonl(jsonl_path, [])
            with self.assertRaisesRegex(FileExistsError, "immutable output"):
                write_json(json_path, {})

    def test_load_hashes_reads_pcm_sha256_column(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "hashes.csv"
            with path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=["id", "pcm_sha256"])
                writer.writeheader()
                writer.writerow({"id": "a", "pcm_sha256": "hash-a"})
                writer.writerow({"id": "b", "pcm_sha256": "hash-b"})

            self.assertEqual(load_hashes(path), {"hash-a", "hash-b"})


if __name__ == "__main__":
    unittest.main()