import unittest

from generate_rq4_calibration_predictions import error_record


class CalibrationErrorTests(unittest.TestCase):
    def test_exact_match_has_zero_error(self) -> None:
        record = error_record("mhoro shamwari", "mhoro shamwari")

        self.assertEqual(record["teacher_wer"], 0.0)
        self.assertEqual(record["teacher_cer"], 0.0)
        self.assertEqual(record["reference_words"], 2)

    def test_word_and_character_errors_are_auditable(self) -> None:
        record = error_record("mhoro shamwari", "mhoro")

        self.assertEqual(record["word_deletions"], 1)
        self.assertEqual(record["teacher_wer"], 0.5)
        self.assertGreater(record["teacher_cer"], 0.0)
        self.assertEqual(
            record["reference_words"],
            record["word_hits"]
            + record["word_substitutions"]
            + record["word_deletions"],
        )


if __name__ == "__main__":
    unittest.main()