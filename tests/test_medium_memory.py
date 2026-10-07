import unittest
from types import SimpleNamespace
from unittest.mock import patch

from asr_experiments.commands.training.test_medium_memory import (
    MemoryCallback, diagnostic_training_config, select_longest,
)


class MediumMemoryTests(unittest.TestCase):
    def test_skipped_updates_do_not_satisfy_probe(self):
        report = {"steps": [], "successful_optimizer_steps": 0}
        callback = MemoryCallback(report)
        optimizer = SimpleNamespace(state={"parameter": {}}, step_was_skipped=True)
        control = SimpleNamespace(should_training_stop=False)
        with patch("torch.cuda.synchronize"), patch(
            "asr_experiments.commands.training.test_medium_memory.memory_snapshot", return_value={}
        ):
            callback.on_step_end(None, SimpleNamespace(global_step=1), control, optimizer)
            self.assertEqual(report["successful_optimizer_steps"], 0)
            optimizer.step_was_skipped = False
            callback.on_step_end(None, SimpleNamespace(global_step=2), control, optimizer)
            self.assertFalse(control.should_training_stop)
            callback.on_step_end(None, SimpleNamespace(global_step=3), control, optimizer)
            self.assertTrue(control.should_training_stop)
        self.assertEqual(report["successful_optimizer_steps"], 2)

    def test_longest_selection_is_deterministic(self):
        rows = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
        self.assertEqual(select_longest(rows, [20, 20, 10], 2), [1, 0])
        with self.assertRaises(ValueError):
            select_longest(rows, [20], 6)

    def test_diagnostic_overrides_do_not_change_batch_or_optimizer(self):
        source = {"per_device_train_batch_size": 6, "gradient_accumulation_steps": 1, "learning_rate": 1e-5, "warmup_steps": 500}
        config = diagnostic_training_config(source, 2)
        for key, value in source.items():
            self.assertEqual(config[key], value)
        self.assertNotIn("max_steps", source)
        self.assertEqual(config["max_steps"], 2)
        self.assertEqual(config["save_strategy"], "no")
        self.assertFalse(config["load_best_model_at_end"])
        with self.assertRaises(ValueError):
            diagnostic_training_config(source, 100)