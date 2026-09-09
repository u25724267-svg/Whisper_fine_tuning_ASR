import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from calibrate_rq4_teacher_diagnostics import FEATURES, feature_matrix
from generate_rq4_teacher_labels import atomic_write_json, atomic_write_jsonl
from inventory_waxal_unlabeled import sha256_file


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = Path(
    "/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1/predictions.jsonl"
)
DEFAULT_TEACHER_SUMMARY = Path(
    "/ext_data/casper/asr_data/rq4_teacher_labels_c0_v1/summary.json"
)
DEFAULT_CALIBRATION = Path("/ext_data/casper/asr_data/rq4_diagnostic_calibration_v1")
DEFAULT_OUTPUT = Path("/ext_data/casper/asr_data/rq4_proxy_shortlist_v1")
AMENDMENT_PATH = (
    ROOT_DIR / "documents" / "data" / "rq2_rq4_parameter_sweep_amendment_v2.md"
)
COMPONENT_TARGET_SECONDS = 80 * 3600
COMPONENT_SPEAKER_SECONDS = 2.0 * 3600
COMPONENT_SPEAKER_ROWS = 378


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen dual-ranking RQ4 proxy shortlist."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--teacher-summary", type=Path, default=DEFAULT_TEACHER_SUMMARY)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def select_capped(
    rows: Sequence[dict[str, Any]],
    poor_scores: Sequence[float],
    target_seconds: float,
    speaker_seconds_cap: float,
    speaker_rows_cap: int,
) -> set[str]:
    if len(rows) != len(poor_scores):
        raise ValueError("Rows and scores must have equal length")
    if target_seconds <= 0 or speaker_seconds_cap <= 0 or speaker_rows_cap <= 0:
        raise ValueError("Selection targets and caps must be positive")
    ranked = sorted(
        zip(rows, poor_scores),
        key=lambda item: (float(item[1]), str(item[0]["id"])),
    )
    selected = set()
    total_seconds = 0.0
    speaker_seconds: dict[str, float] = defaultdict(float)
    speaker_rows: Counter[str] = Counter()
    for row, _ in ranked:
        duration = float(row["duration"])
        speaker = str(row["speaker_id"])
        if total_seconds + duration > target_seconds:
            continue
        if speaker_seconds[speaker] + duration > speaker_seconds_cap:
            continue
        if speaker_rows[speaker] >= speaker_rows_cap:
            continue
        selected.add(str(row["id"]))
        total_seconds += duration
        speaker_seconds[speaker] += duration
        speaker_rows[speaker] += 1
    if target_seconds - total_seconds > 30.0:
        raise RuntimeError(
            f"Capped selection is {target_seconds - total_seconds:.3f} seconds short"
        )
    return selected


def selection_summary(rows: Sequence[dict[str, Any]], selected: set[str]) -> dict[str, Any]:
    selected_rows = [row for row in rows if str(row["id"]) in selected]
    if not selected_rows:
        return {
            "rows": 0,
            "hours": 0.0,
            "speakers": 0,
            "maximum_speaker_hours": None,
            "maximum_speaker_rows": None,
            "maximum_speaker_duration_share": None,
            "maximum_speaker_row_share": None,
        }
    speaker_seconds: dict[str, float] = defaultdict(float)
    speaker_rows: Counter[str] = Counter()
    for row in selected_rows:
        speaker = str(row["speaker_id"])
        speaker_seconds[speaker] += float(row["duration"])
        speaker_rows[speaker] += 1
    total_seconds = sum(float(row["duration"]) for row in selected_rows)
    return {
        "rows": len(selected_rows),
        "hours": total_seconds / 3600,
        "speakers": len(speaker_rows),
        "maximum_speaker_hours": max(speaker_seconds.values()) / 3600,
        "maximum_speaker_rows": max(speaker_rows.values()),
        "maximum_speaker_duration_share": max(speaker_seconds.values()) / total_seconds,
        "maximum_speaker_row_share": max(speaker_rows.values()) / len(selected_rows),
    }


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions.expanduser().resolve()
    teacher_summary_path = args.teacher_summary.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    calibration_summary_path = calibration_dir / "summary.json"
    model_path = calibration_dir / "diagnostic_model.joblib"
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    required = (
        predictions_path,
        teacher_summary_path,
        calibration_summary_path,
        model_path,
        AMENDMENT_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shortlist prerequisites: {missing}")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Output or staging path already exists: {output_dir}")

    teacher_summary = json.loads(teacher_summary_path.read_text(encoding="utf-8"))
    if teacher_summary["predictions_sha256"] != sha256_file(predictions_path):
        raise ValueError("Teacher predictions do not match their immutable summary")
    calibration_summary = json.loads(
        calibration_summary_path.read_text(encoding="utf-8")
    )
    if calibration_summary["final_model"]["sha256"] != sha256_file(model_path):
        raise ValueError("Diagnostic model does not match calibration summary")
    model_artifact = joblib.load(model_path)
    if tuple(model_artifact["features"]) != FEATURES:
        raise ValueError("Diagnostic model features do not match frozen features")

    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Teacher prediction IDs must be unique and non-empty")
    eligible_rows = [
        row for row in rows if not row["empty_text"] and not row["hit_max_length"]
    ]
    matrix = feature_matrix(eligible_rows)
    diagnostic_scores = model_artifact["model"].predict_proba(matrix)[:, 1]
    confidence_scores = np.asarray(
        [
            1.0 - float(row["geometric_content_token_confidence"])
            for row in eligible_rows
        ]
    )

    confidence_ids = select_capped(
        eligible_rows,
        confidence_scores,
        COMPONENT_TARGET_SECONDS,
        COMPONENT_SPEAKER_SECONDS,
        COMPONENT_SPEAKER_ROWS,
    )
    diagnostic_ids = select_capped(
        eligible_rows,
        diagnostic_scores,
        COMPONENT_TARGET_SECONDS,
        COMPONENT_SPEAKER_SECONDS,
        COMPONENT_SPEAKER_ROWS,
    )
    union_ids = confidence_ids | diagnostic_ids
    diagnostic_by_id = {
        str(row["id"]): float(score)
        for row, score in zip(eligible_rows, diagnostic_scores)
    }
    confidence_by_id = {
        str(row["id"]): float(score)
        for row, score in zip(eligible_rows, confidence_scores)
    }
    shortlist_rows = [
        {
            **row,
            "confidence_poor_score": confidence_by_id[str(row["id"])],
            "diagnostic_poor_probability": diagnostic_by_id[str(row["id"])],
            "shortlist_sources": sorted(
                source
                for source, selected in (
                    ("confidence_top_80h", confidence_ids),
                    ("diagnostic_top_80h", diagnostic_ids),
                )
                if str(row["id"]) in selected
            ),
        }
        for row in eligible_rows
        if str(row["id"]) in union_ids
    ]
    union_summary = selection_summary(eligible_rows, union_ids)
    if union_summary["maximum_speaker_hours"] > 4.0 + 1e-9:
        raise RuntimeError("Union duration cap exceeded")
    if union_summary["maximum_speaker_rows"] > 756:
        raise RuntimeError("Union row cap exceeded")
    if union_summary["speakers"] < 40:
        raise RuntimeError("Shortlist must contain at least 40 speakers")

    try:
        staging_dir.mkdir(parents=True)
        shortlist_path = staging_dir / "shortlist.jsonl"
        atomic_write_jsonl(shortlist_path, shortlist_rows)
        summary = {
            "schema_version": 1,
            "protocol": "rq4-proxy-shortlist-v1",
            "ranking_rule": "union(confidence_top_80h, diagnostic_top_80h)",
            "hard_rejections": {
                "empty_text": sum(bool(row["empty_text"]) for row in rows),
                "hit_max_length": sum(bool(row["hit_max_length"]) for row in rows),
            },
            "component_caps": {
                "target_hours": COMPONENT_TARGET_SECONDS / 3600,
                "speaker_hours": COMPONENT_SPEAKER_SECONDS / 3600,
                "speaker_rows": COMPONENT_SPEAKER_ROWS,
            },
            "confidence_component": selection_summary(eligible_rows, confidence_ids),
            "diagnostic_component": selection_summary(eligible_rows, diagnostic_ids),
            "overlap": selection_summary(
                eligible_rows, confidence_ids & diagnostic_ids
            ),
            "union": union_summary,
            "teacher_predictions": str(predictions_path),
            "teacher_predictions_sha256": sha256_file(predictions_path),
            "calibration_summary": str(calibration_summary_path),
            "calibration_summary_sha256": sha256_file(calibration_summary_path),
            "diagnostic_model": str(model_path),
            "diagnostic_model_sha256": sha256_file(model_path),
            "amendment": str(AMENDMENT_PATH),
            "amendment_sha256": sha256_file(AMENDMENT_PATH),
            "shortlist": str(output_dir / shortlist_path.name),
            "shortlist_sha256": sha256_file(shortlist_path),
        }
        atomic_write_json(staging_dir / "summary.json", summary)
        staging_dir.replace(output_dir)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


if __name__ == "__main__":
    main()