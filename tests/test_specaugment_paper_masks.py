import unittest
from types import SimpleNamespace

import torch

from train_full import (
    apply_paper_specaugment_masks,
    configure_augmentation,
    validate_augmentation_config,
)


class FakeWhisperModule:
    def __init__(self) -> None:
        self.training = True

    def _mask_input_features(self, input_features, attention_mask=None):
        return input_features


class FakeWhisperModel:
    def __init__(self, mel_bins: int = 8) -> None:
        self.config = SimpleNamespace(num_mel_bins=mel_bins)
        self.model = FakeWhisperModule()


class PaperSpecAugmentTests(unittest.TestCase):
    def test_time_masks_do_not_enter_padding(self) -> None:
        torch.manual_seed(7)
        features = torch.ones((1, 8, 20))
        attention_mask = torch.tensor([[1] * 10 + [0] * 10])

        masked = apply_paper_specaugment_masks(
            features,
            attention_mask,
            application_probability=1.0,
            frequency_mask_count=0,
            frequency_mask_max_width=0,
            time_mask_count=8,
            time_mask_max_width=6,
            time_mask_max_proportion=0.5,
        )

        self.assertTrue(torch.equal(masked[:, :, 10:], features[:, :, 10:]))
        self.assertTrue(torch.any(masked[:, :, :10] == 0))

    def test_frequency_masks_cover_complete_time_axis(self) -> None:
        torch.manual_seed(11)
        features = torch.ones((1, 8, 20))

        masked = apply_paper_specaugment_masks(
            features,
            None,
            application_probability=1.0,
            frequency_mask_count=8,
            frequency_mask_max_width=3,
            time_mask_count=0,
            time_mask_max_width=0,
            time_mask_max_proportion=1.0,
        )

        zero_rows = torch.all(masked[0] == 0, dim=1)
        self.assertTrue(torch.any(zero_rows))

    def test_wrapper_is_training_only(self) -> None:
        config = {
            "type": "specaugment_paper_masks",
            "enabled": True,
            "application_probability": 1.0,
            "frequency_mask_count": 4,
            "frequency_mask_max_width": 3,
            "time_mask_count": 4,
            "time_mask_max_width": 5,
            "time_mask_max_proportion": 0.5,
        }
        model = FakeWhisperModel()
        configure_augmentation(model, config)
        features = torch.ones((1, 8, 20))
        attention_mask = torch.ones((1, 20), dtype=torch.long)

        torch.manual_seed(3)
        training_output = model.model._mask_input_features(features, attention_mask)
        model.model.training = False
        evaluation_output = model.model._mask_input_features(features, attention_mask)

        self.assertTrue(torch.any(training_output == 0))
        self.assertTrue(torch.equal(evaluation_output, features))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_augmentation_config(
                {
                    "type": "specaugment_paper_masks",
                    "enabled": True,
                    "frequency_mask_count": 2,
                    "frequency_mask_max_width": 81,
                    "time_mask_count": 2,
                    "time_mask_max_width": 20,
                    "time_mask_max_proportion": 0.2,
                },
                num_mel_bins=80,
            )
        with self.assertRaises(ValueError):
            validate_augmentation_config(
                {"type": "none", "enabled": True}, num_mel_bins=80
            )


if __name__ == "__main__":
    unittest.main()