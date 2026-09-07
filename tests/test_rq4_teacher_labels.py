import math
import unittest

from generate_rq4_teacher_labels import (
    compression_ratio,
    repeated_ngram_fraction,
    select_duration_stratified_rows,
    token_statistics,
)


class TeacherLabelUtilityTests(unittest.TestCase):
    def test_duration_stratification_includes_extremes_and_preserves_order(self) -> None:
        rows = [
            {"id": f"utt-{index}", "duration": float(duration)}
            for index, duration in enumerate([10, 1, 30, 5, 20])
        ]
        selected = select_duration_stratified_rows(rows, 3)

        self.assertEqual([row["_admission_manifest_index"] for row in selected], [0, 1, 2])
        self.assertEqual({row["duration"] for row in selected}, {1.0, 10.0, 30.0})

    def test_token_statistics_trim_post_eos_padding(self) -> None:
        first = token_statistics(
            [10, 11, 99, 99], [-0.1, -0.2, -0.3, -9.0], {99}, 99
        )
        second = token_statistics(
            [10, 11, 99, 99, 99],
            [-0.1, -0.2, -0.3, -9.0, -9.0],
            {99},
            99,
        )

        self.assertEqual(first["generated_token_ids"], [10, 11, 99])
        self.assertEqual(first["generated_token_ids"], second["generated_token_ids"])
        self.assertEqual(
            first["generated_token_log_probabilities"],
            second["generated_token_log_probabilities"],
        )
        self.assertAlmostEqual(
            first["geometric_content_token_confidence"], math.exp(-0.15)
        )

    def test_repetition_and_compression_diagnostics(self) -> None:
        words = "a b a b a b".split()
        self.assertGreater(repeated_ngram_fraction(words, 2), 0)
        self.assertEqual(repeated_ngram_fraction(["a"], 2), 0)
        self.assertIsNone(compression_ratio(""))
        self.assertGreater(compression_ratio("zvakare " * 20), 1)


if __name__ == "__main__":
    unittest.main()