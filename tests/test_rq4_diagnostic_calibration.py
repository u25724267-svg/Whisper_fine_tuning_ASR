import unittest

import numpy as np

from calibrate_rq4_teacher_diagnostics import (
    FEATURES,
    choose_c,
    confidence_poor_score,
    discrimination_metrics,
    feature_matrix,
    grouped_oof_probabilities,
    target_vector,
)


def synthetic_rows() -> list[dict]:
    rows = []
    for speaker_index in range(8):
        for poor in (False, True):
            confidence = 0.9 if not poor else 0.2
            row = {feature: 0.0 for feature in FEATURES}
            row.update(
                {
                    "id": f"speaker-{speaker_index}-{int(poor)}",
                    "speaker_id": f"speaker-{speaker_index}",
                    "teacher_wer": 0.1 if not poor else 0.9,
                    "geometric_content_token_confidence": confidence,
                    "minimum_content_token_probability": confidence / 2,
                    "p10_content_token_probability": confidence / 1.5,
                    "duration": 10.0,
                    "content_token_count": 20,
                    "hit_max_length": poor,
                }
            )
            rows.append(row)
    return rows


class DiagnosticCalibrationTests(unittest.TestCase):
    def test_grouped_predictions_and_metrics_follow_error_direction(self) -> None:
        rows = synthetic_rows()
        matrix = feature_matrix(rows)
        labels = target_vector(rows, 0.5)
        groups = np.asarray([row["speaker_id"] for row in rows])

        probabilities = grouped_oof_probabilities(
            matrix, labels, groups, folds=4, c_value=1.0, seed=42
        )
        metrics = discrimination_metrics(labels, probabilities)

        self.assertEqual(matrix.shape, (16, len(FEATURES)))
        self.assertGreater(metrics["roc_auc"], 0.9)
        self.assertGreater(metrics["pr_auc"], 0.9)

    def test_confidence_baseline_and_regularization_selection(self) -> None:
        rows = synthetic_rows()
        matrix = feature_matrix(rows)
        labels = target_vector(rows, 0.5)
        groups = np.asarray([row["speaker_id"] for row in rows])

        confidence_scores = confidence_poor_score(rows)
        selected_c, scores = choose_c(
            matrix, labels, groups, folds=4, seed=42
        )

        self.assertGreater(confidence_scores[1], confidence_scores[0])
        self.assertIn(selected_c, (0.1, 1.0, 10.0))
        self.assertEqual(set(scores), {"0.1", "1.0", "10.0"})


if __name__ == "__main__":
    unittest.main()