import unittest

from build_rq4_proxy_shortlist import select_capped, selection_summary


class ProxyShortlistTests(unittest.TestCase):
    def test_selection_respects_duration_and_row_caps(self) -> None:
        rows = [
            {
                "id": f"{speaker}-{index}",
                "speaker_id": speaker,
                "duration": 1.0,
            }
            for speaker in ("a", "b", "c", "d")
            for index in range(3)
        ]
        scores = [index / len(rows) for index in range(len(rows))]

        selected = select_capped(
            rows,
            scores,
            target_seconds=6.0,
            speaker_seconds_cap=2.0,
            speaker_rows_cap=2,
        )
        summary = selection_summary(rows, selected)

        self.assertEqual(summary["rows"], 6)
        self.assertEqual(summary["maximum_speaker_rows"], 2)
        self.assertLessEqual(summary["maximum_speaker_hours"], 2 / 3600)

    def test_empty_selection_summary_is_defined(self) -> None:
        summary = selection_summary(
            [{"id": "one", "speaker_id": "speaker", "duration": 1.0}],
            set(),
        )

        self.assertEqual(summary["rows"], 0)
        self.assertIsNone(summary["maximum_speaker_rows"])


if __name__ == "__main__":
    unittest.main()