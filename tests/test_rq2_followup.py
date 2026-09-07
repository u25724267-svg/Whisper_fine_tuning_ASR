import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from compare_predictions_factorial_bootstrap import factorial_effects, holm_adjust
from prepare_rq2_asset_partitions import (
    assign_partition,
    canonical_audio_identity,
    sha256_file,
)
from prepare_rq2_followup_data import (
    add_noise,
    apply_speed,
    fit_noise,
    main as prepare_followup_data,
    validate_args,
)


class AssetPartitionTests(unittest.TestCase):
    def test_canonical_identity_matches_decoded_duplicate_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            waveform = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
            first = root / "first.wav"
            duplicate = root / "duplicate.wav"
            sf.write(first, waveform, 16_000, subtype="PCM_16")
            sf.write(duplicate, waveform, 16_000, subtype="PCM_16")

            first_identity = canonical_audio_identity(first)[0]
            duplicate_identity = canonical_audio_identity(duplicate)[0]

            self.assertEqual(first_identity, duplicate_identity)
            self.assertEqual(
                assign_partition(first_identity, "fixed-salt", 8, 1, 1),
                assign_partition(duplicate_identity, "fixed-salt", 8, 1, 1),
            )


class WaveformTests(unittest.TestCase):
    def test_noise_snr_and_resampling_direction(self) -> None:
        generator = np.random.default_rng(42)
        speech = generator.normal(0, 0.1, 16_000).astype(np.float32)
        noise = generator.normal(0, 0.1, 24_000).astype(np.float32)

        _, parameters = add_noise(speech, noise, 10.0, generator)
        slow = apply_speed(speech, 0.9)
        fast = apply_speed(speech, 1.1)

        self.assertAlmostEqual(parameters["achieved_snr_db"], 10.0, places=5)
        self.assertGreater(len(slow), len(speech))
        self.assertLess(len(fast), len(speech))
        self.assertAlmostEqual(len(slow) / len(speech), 1 / 0.9, places=3)
        self.assertAlmostEqual(len(fast) / len(speech), 1 / 1.1, places=3)

        with self.assertRaises(ValueError):
            fit_noise(noise[:100], len(speech), generator)

    def test_test_partition_requires_explicit_unlock(self) -> None:
        arguments = argparse.Namespace(
            condition="noise",
            clean_probability=0.0,
            sampling_rate=16_000,
            peak_limit=0.99,
            max_output_seconds=30.0,
            snr_db=[10.0],
            speed_factor=[],
            asset_root=Path("assets"),
            asset_manifest=Path("test.jsonl"),
            asset_partition="test",
            allow_test_partition=False,
        )
        with self.assertRaises(PermissionError):
            validate_args(arguments)

    def test_clean_materialization_preserves_source_path_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_audio = root / "speech.wav"
            waveform = np.linspace(-1.0, 1.0, 8_000, dtype=np.float32)
            sf.write(source_audio, waveform, 16_000, subtype="PCM_16")
            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                json.dumps(
                    {
                        "id": "utt-clean",
                        "audio_filepath": str(source_audio),
                        "duration": 0.5,
                        "text": "mhoro",
                        "speaker_id": "speaker-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_sha256 = sha256_file(source_audio)
            output_dir = root / "output"
            arguments = [
                "prepare_rq2_followup_data.py",
                "--source-manifest",
                str(source_manifest),
                "--output-dir",
                str(output_dir),
                "--condition",
                "clean",
                "--seed",
                "42",
            ]
            with patch.object(sys, "argv", arguments):
                prepare_followup_data()

            output_row = json.loads(
                (output_dir / "manifest.jsonl").read_text(encoding="utf-8")
            )
            parameters = json.loads(
                (output_dir / "parameters.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(output_row["audio_filepath"], str(source_audio))
            self.assertEqual(parameters["output_sha256"], source_sha256)
            self.assertEqual(parameters["anti_clipping_gain"], 1.0)

    def test_one_row_noise_materialization_is_paired_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asset_root = root / "assets"
            asset_root.mkdir()
            source_audio = root / "speech.wav"
            noise_audio = asset_root / "noise.wav"
            time = np.arange(8_000, dtype=np.float32) / 16_000
            speech = 0.1 * np.sin(2 * math.pi * 220 * time)
            noise = np.random.default_rng(7).normal(0, 0.1, 16_000).astype(
                np.float32
            )
            sf.write(source_audio, speech, 16_000, subtype="PCM_16")
            sf.write(noise_audio, noise, 16_000, subtype="PCM_16")

            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                json.dumps(
                    {
                        "id": "utt-1",
                        "audio_filepath": str(source_audio),
                        "duration": 0.5,
                        "text": "mhoro",
                        "speaker_id": "speaker-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            content_sha256, sampling_rate, frames, channels = canonical_audio_identity(
                noise_audio
            )
            asset_manifest = root / "validation.jsonl"
            asset_manifest.write_text(
                json.dumps(
                    {
                        "kind": "noise",
                        "relative_path": noise_audio.name,
                        "file_sha256": sha256_file(noise_audio),
                        "content_sha256": content_sha256,
                        "partition": "validation",
                        "sampling_rate": sampling_rate,
                        "frames": frames,
                        "channels": channels,
                        "duration_seconds": frames / sampling_rate,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            arguments = [
                "prepare_rq2_followup_data.py",
                "--source-manifest",
                str(source_manifest),
                "--output-dir",
                str(output_dir),
                "--condition",
                "noise",
                "--seed",
                "42",
                "--snr-db",
                "10",
                "--asset-root",
                str(asset_root),
                "--asset-manifest",
                str(asset_manifest),
                "--asset-partition",
                "validation",
            ]
            with patch.object(sys, "argv", arguments):
                prepare_followup_data()

            output_row = json.loads(
                (output_dir / "manifest.jsonl").read_text(encoding="utf-8")
            )
            parameters = json.loads(
                (output_dir / "parameters.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(output_row["id"], "utt-1")
            self.assertEqual(output_row["speaker_id"], "speaker-1")
            self.assertEqual(parameters["noise_content_sha256"], content_sha256)
            self.assertAlmostEqual(parameters["achieved_snr_db"], 10.0, places=5)
            self.assertTrue(Path(output_row["audio_filepath"]).is_file())


class FactorialAnalysisTests(unittest.TestCase):
    def test_factorial_effect_direction_and_holm_monotonicity(self) -> None:
        effects = factorial_effects(
            {
                "cell_00": 30.0,
                "cell_noise": 28.0,
                "cell_rir": 29.0,
                "cell_noise_rir": 25.0,
            }
        )
        self.assertEqual(effects["noise_main_improvement"], 3.0)
        self.assertEqual(effects["rir_main_improvement"], 2.0)
        self.assertEqual(effects["noise_rir_interaction_improvement"], 2.0)

        adjusted = holm_adjust({"noise": 0.01, "rir": 0.03, "interaction": 0.2})
        self.assertEqual(adjusted["noise"], 0.03)
        self.assertEqual(adjusted["rir"], 0.06)
        self.assertEqual(adjusted["interaction"], 0.2)


if __name__ == "__main__":
    unittest.main()