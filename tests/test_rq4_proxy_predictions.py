import unittest

from generate_rq4_proxy_predictions import (
    normalized_character_disagreement,
    select_stratified_calibration_rows,
    trim_token_ids,
)


class ProxyPredictionTests(unittest.TestCase):
    def test_character_disagreement_is_symmetric_and_normalized(self) -> None:
        first = normalized_character_disagreement("mhoro", "moro")
        second = normalized_character_disagreement("moro", "mhoro")

        self.assertEqual(first, second)
        self.assertAlmostEqual(first, 1 / 5)
        self.assertEqual(normalized_character_disagreement("", ""), 0.0)

    def test_token_trimming_stops_at_eos_or_padding(self) -> None:
        self.assertEqual(trim_token_ids([1, 2, 3, 0, 0], 3, 0), [1, 2, 3])
        self.assertEqual(trim_token_ids([1, 2, 0, 0], 3, 0), [1, 2])
        self.assertEqual(
            trim_token_ids([3, 256078, 42, 3, 0], 3, 0),
            [3, 256078, 42, 3],
        )

    def test_calibration_pilot_covers_all_error_strata(self) -> None:
        rows = []
        for stratum, wer in enumerate((0.1, 0.45, 0.7, 0.9)):
            rows.extend(
                {
                    "id": f"{stratum}-{index}",
                    "speaker_id": f"speaker-{stratum}-{index}",
                    "duration": float(index + 1),
                    "teacher_wer": wer,
                }
                for index in range(10)
            )

        selected = select_stratified_calibration_rows(rows, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(len({row["proxy_pilot_stratum"] for row in selected}), 4)
        self.assertEqual(
            [row["proxy_input_index"] for row in selected],
            sorted(row["proxy_input_index"] for row in selected),
        )


if __name__ == "__main__":
    unittest.main()