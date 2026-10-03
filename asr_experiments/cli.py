"""Dispatch command names to focused implementation modules."""

import importlib
import sys
from collections.abc import Sequence


COMMAND_MODULES = {
    "aggregate_rq2_factorial": "asr_experiments.commands.analysis.aggregate_rq2_factorial",
    "audit_waxal_unlabeled_shards": "asr_experiments.commands.data.audit_waxal_unlabeled_shards",
    "build_rq4_proxy_shortlist": "asr_experiments.commands.rq4.build_rq4_proxy_shortlist",
    "calibrate_rq4_teacher_diagnostics": "asr_experiments.commands.rq4.calibrate_rq4_teacher_diagnostics",
    "compare_predictions_bootstrap": "asr_experiments.commands.analysis.compare_predictions_bootstrap",
    "compare_predictions_factorial_bootstrap": "asr_experiments.commands.analysis.compare_predictions_factorial_bootstrap",
    "evaluate_predictions": "asr_experiments.commands.evaluation.evaluate_predictions",
    "evaluate_zero_shot": "asr_experiments.commands.evaluation.evaluate_zero_shot",
    "generate_rq4_calibration_predictions": "asr_experiments.commands.rq4.generate_rq4_calibration_predictions",
    "generate_rq4_proxy_predictions": "asr_experiments.commands.rq4.generate_rq4_proxy_predictions",
    "generate_rq4_teacher_labels": "asr_experiments.commands.rq4.generate_rq4_teacher_labels",
    "inventory_waxal_unlabeled": "asr_experiments.commands.data.inventory_waxal_unlabeled",
    "mirror_experiment_artifacts": "asr_experiments.commands.artifacts.mirror_experiment_artifacts",
    "prepare_acoustic_metadata": "asr_experiments.commands.data.prepare_acoustic_metadata",
    "prepare_fleurs_corrected_v2": "asr_experiments.commands.data.prepare_fleurs_corrected_v2",
    "prepare_medium_queue": "asr_experiments.commands.data.prepare_medium_queue",
    "prepare_rq2_asset_partitions": "asr_experiments.commands.data.prepare_rq2_asset_partitions",
    "prepare_rq2_followup_data": "asr_experiments.commands.data.prepare_rq2_followup_data",
    "prepare_rq2_replication_configs": "asr_experiments.commands.data.prepare_rq2_replication_configs",
    "prepare_rq4_admission": "asr_experiments.commands.rq4.prepare_rq4_admission",
    "prepare_waveform_augmentation": "asr_experiments.commands.data.prepare_waveform_augmentation",
    "prepare_waxal_fleurs": "asr_experiments.commands.data.prepare_waxal_fleurs",
    "prepare_waxal_speaker_disjoint": "asr_experiments.commands.data.prepare_waxal_speaker_disjoint",
    "prepare_waxal_speaker_disjoint_v2": "asr_experiments.commands.data.prepare_waxal_speaker_disjoint_v2",
    "train_full": "asr_experiments.commands.training.train_full",
    "train_s2s_curriculum": "asr_experiments.commands.training.train_s2s_curriculum",
    "train_snr_curriculum": "asr_experiments.commands.training.train_snr_curriculum",
    "train_sortagrad": "asr_experiments.commands.training.train_sortagrad",
}


def main(arguments: Sequence[str] | None = None) -> None:
    """Load a registered command module and delegate its remaining arguments."""
    values = list(sys.argv[1:] if arguments is None else arguments)
    if not values or values[0] in {"-h", "--help"}:
        commands = "\n  ".join(sorted(COMMAND_MODULES))
        print(f"Usage: asr.py COMMAND [ARGS...]\n\nCommands:\n  {commands}")
        return
    command = values.pop(0)
    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        raise SystemExit(f"Unknown command: {command}. Use --help to list commands.")
    module = importlib.import_module(module_name)
    command_main = getattr(module, "main", None)
    if command_main is None:
        raise RuntimeError(f"Command module has no main(): {module_name}")
    original_argv = sys.argv
    try:
        sys.argv = [f"asr.py {command}", *values]
        command_main()
    finally:
        sys.argv = original_argv