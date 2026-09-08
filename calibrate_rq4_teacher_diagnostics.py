import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_rq4_teacher_labels import atomic_write_json, atomic_write_jsonl
from inventory_waxal_unlabeled import sha256_file


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(
    "/ext_data/casper/asr_data/rq4_calibration_c0_train_v1/predictions.jsonl"
)
DEFAULT_OUTPUT = Path("/ext_data/casper/asr_data/rq4_diagnostic_calibration_v1")
AMENDMENT_PATH = (
    ROOT_DIR / "documents" / "data" / "rq2_rq4_parameter_sweep_amendment_v2.md"
)
FEATURES = (
    "geometric_content_token_confidence",
    "minimum_content_token_probability",
    "p10_content_token_probability",
    "raw_no_speech_token_probability",
    "utf8_compression_ratio",
    "words_per_second",
    "characters_per_second",
    "unique_word_ratio",
    "maximum_consecutive_word_run",
    "repeated_bigram_fraction",
    "repeated_trigram_fraction",
    "non_letter_fraction_raw",
    "duration",
    "content_token_count",
    "hit_max_length",
)
C_VALUES = (0.1, 1.0, 10.0)
TARGET_THRESHOLDS = (0.4, 0.5, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested speaker-grouped calibration of RQ4 teacher diagnostics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                np.nan
                if row.get(feature) is None
                else float(bool(row[feature]))
                if isinstance(row[feature], bool)
                else float(row[feature])
                for feature in FEATURES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def target_vector(rows: Sequence[dict[str, Any]], threshold: float) -> np.ndarray:
    return np.asarray(
        [float(row["teacher_wer"]) > threshold for row in rows], dtype=np.int64
    )


def confidence_poor_score(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            1.0 - float(row["geometric_content_token_confidence"])
            if row.get("geometric_content_token_confidence") is not None
            else 1.0
            for row in rows
        ],
        dtype=np.float64,
    )


def build_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def validate_group_folds(labels: np.ndarray, groups: np.ndarray, folds: int) -> None:
    if folds < 2:
        raise ValueError("Grouped cross-validation requires at least two folds")
    if len(set(groups.tolist())) < folds:
        raise ValueError("Not enough speaker groups for requested folds")
    splitter = GroupKFold(n_splits=folds)
    placeholder = np.zeros((len(labels), 1))
    for _, test_indices in splitter.split(placeholder, labels, groups):
        if len(np.unique(labels[test_indices])) != 2:
            raise ValueError("Every grouped test fold must contain both target classes")


def grouped_oof_probabilities(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    c_value: float,
    seed: int,
) -> np.ndarray:
    validate_group_folds(labels, groups, folds)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train_indices, test_indices in splitter.split(matrix, labels, groups):
        model = build_model(c_value, seed)
        model.fit(matrix[train_indices], labels[train_indices])
        probabilities[test_indices] = model.predict_proba(matrix[test_indices])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Grouped predictions are incomplete or non-finite")
    return probabilities


def choose_c(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    scores = {}
    for c_value in C_VALUES:
        probabilities = grouped_oof_probabilities(
            matrix, labels, groups, folds, c_value, seed
        )
        scores[str(c_value)] = float(average_precision_score(labels, probabilities))
    selected = max(C_VALUES, key=lambda value: (scores[str(value)], -value))
    return selected, scores


def discrimination_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return {
            "rows": len(labels),
            "positives": positives,
            "negatives": negatives,
            "roc_auc": None,
            "pr_auc": None,
        }
    return {
        "rows": len(labels),
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / len(labels),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def gate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    true_positives = int(np.logical_and(labels == 1, predictions == 1).sum())
    false_positives = int(np.logical_and(labels == 0, predictions == 1).sum())
    false_negatives = int(np.logical_and(labels == 1, predictions == 0).sum())
    true_negatives = int(np.logical_and(labels == 0, predictions == 0).sum())
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else None
        ),
        "recall": (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing calibration predictions: {input_path}")
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen amendment: {AMENDMENT_PATH}")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Output or staging path already exists: {output_dir}")

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["id"]) for row in rows]
    groups = np.asarray([str(row["speaker_id"]) for row in rows])
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Calibration predictions require unique non-empty rows")

    matrix = feature_matrix(rows)
    primary_labels = target_vector(rows, 0.5)
    validate_group_folds(primary_labels, groups, args.outer_folds)
    outer_splitter = GroupKFold(n_splits=args.outer_folds)
    diagnostic_probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    outer_fold = np.full(len(rows), -1, dtype=np.int64)
    outer_selection = []
    for fold, (train_indices, test_indices) in enumerate(
        outer_splitter.split(matrix, primary_labels, groups)
    ):
        selected_c, inner_scores = choose_c(
            matrix[train_indices],
            primary_labels[train_indices],
            groups[train_indices],
            args.inner_folds,
            args.seed,
        )
        model = build_model(selected_c, args.seed)
        model.fit(matrix[train_indices], primary_labels[train_indices])
        diagnostic_probabilities[test_indices] = model.predict_proba(
            matrix[test_indices]
        )[:, 1]
        outer_fold[test_indices] = fold
        outer_selection.append(
            {
                "fold": fold,
                "selected_c": selected_c,
                "inner_pr_auc": inner_scores,
                "train_speakers": len(set(groups[train_indices].tolist())),
                "test_speakers": len(set(groups[test_indices].tolist())),
                "test_rows": len(test_indices),
            }
        )
    if not np.isfinite(diagnostic_probabilities).all() or (outer_fold < 0).any():
        raise RuntimeError("Outer grouped predictions are incomplete")

    confidence_scores = confidence_poor_score(rows)
    metrics = {}
    for threshold in TARGET_THRESHOLDS:
        labels = target_vector(rows, threshold)
        metrics[str(threshold)] = {
            "target": f"teacher_wer>{threshold}",
            "confidence_only": discrimination_metrics(labels, confidence_scores),
            "diagnostic_logistic": discrimination_metrics(
                labels, diagnostic_probabilities
            ),
        }

    mean_log_probability = np.asarray(
        [
            float(row["content_token_log_probability_mean"])
            if row.get("content_token_log_probability_mean") is not None
            else -math.inf
            for row in rows
        ]
    )
    compression = np.asarray(
        [
            float(row["utf8_compression_ratio"])
            if row.get("utf8_compression_ratio") is not None
            else 0.0
            for row in rows
        ]
    )
    no_speech = np.asarray(
        [float(row["raw_no_speech_token_probability"]) for row in rows]
    )
    primary_gate_labels = target_vector(rows, 0.5)
    canonical_diagnostics = {
        "compression_gt_2_4": gate_metrics(primary_gate_labels, compression > 2.4),
        "mean_logprob_lt_minus_1": gate_metrics(
            primary_gate_labels, mean_log_probability < -1.0
        ),
        "no_speech_gt_0_6_and_mean_logprob_lt_minus_1": gate_metrics(
            primary_gate_labels,
            np.logical_and(no_speech > 0.6, mean_log_probability < -1.0),
        ),
        "hit_max_length": gate_metrics(
            primary_gate_labels,
            np.asarray([bool(row["hit_max_length"]) for row in rows]),
        ),
    }

    selected_c, full_inner_scores = choose_c(
        matrix, primary_labels, groups, args.inner_folds, args.seed
    )
    final_model = build_model(selected_c, args.seed)
    final_model.fit(matrix, primary_labels)

    try:
        staging_dir.mkdir(parents=True)
        oof_path = staging_dir / "oof_predictions.jsonl"
        atomic_write_jsonl(
            oof_path,
            (
                {
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "teacher_wer": row["teacher_wer"],
                    "teacher_cer": row["teacher_cer"],
                    "outer_fold": int(outer_fold[index]),
                    "confidence_poor_score": float(confidence_scores[index]),
                    "diagnostic_poor_probability": float(
                        diagnostic_probabilities[index]
                    ),
                }
                for index, row in enumerate(rows)
            ),
        )
        model_path = staging_dir / "diagnostic_model.joblib"
        joblib.dump(
            {
                "model": final_model,
                "features": FEATURES,
                "target": "teacher_wer>0.5",
                "selected_c": selected_c,
            },
            model_path,
        )
        provenance = {
            "schema_version": 1,
            "protocol": "rq4-diagnostic-calibration-v1",
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "amendment": str(AMENDMENT_PATH),
            "amendment_sha256": sha256_file(AMENDMENT_PATH),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "features": list(FEATURES),
            "c_values": list(C_VALUES),
            "target_thresholds": list(TARGET_THRESHOLDS),
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": args.seed,
        }
        summary = {
            **provenance,
            "rows": len(rows),
            "speakers": len(set(groups.tolist())),
            "metrics": metrics,
            "canonical_whisper_diagnostics": canonical_diagnostics,
            "outer_selection": outer_selection,
            "final_model": {
                "selected_c": selected_c,
                "inner_pr_auc": full_inner_scores,
                "path": str(output_dir / model_path.name),
                "sha256": sha256_file(model_path),
            },
            "oof_predictions": str(output_dir / oof_path.name),
            "oof_predictions_sha256": sha256_file(oof_path),
            "proxy_qualification_auc": 0.70,
            "proxy_qualification_status": "pending_proxy_predictions",
            "provenance_sha256": canonical_json_sha256(provenance),
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