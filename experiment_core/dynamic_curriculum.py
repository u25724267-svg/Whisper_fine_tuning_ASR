import json
import math
import threading
import uuid
from pathlib import Path
from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler

from experiment_core.samplers import percentile_ranks


class S2SLossCurriculumState:
    def __init__(
        self,
        utterance_ids: Sequence[str],
        durations: Sequence[float],
        seed: int,
        artifact_dir: Path,
    ) -> None:
        self.utterance_ids = [str(value) for value in utterance_ids]
        self.durations = [float(value) for value in durations]
        self.seed = int(seed)
        self.artifact_dir = artifact_dir
        self.current_epoch = 0
        self._lock = threading.RLock()
        self._finalized_epochs: set[int] = set()
        self._sequence_nll: list[Optional[float]] = []
        self._valid_tokens: list[Optional[int]] = []
        self.order = self._random_order()
        self.order_scores: Optional[list[float]] = None
        self._reset_scores()

        if not self.utterance_ids:
            raise ValueError("S2S curriculum requires at least one utterance")
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("S2S curriculum utterance IDs must be unique")
        if len(self.utterance_ids) != len(self.durations):
            raise ValueError("S2S curriculum IDs and durations must have equal length")
        if any(not math.isfinite(value) or value <= 0 for value in self.durations):
            raise ValueError("S2S curriculum durations must be positive and finite")

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def _random_order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed)
        return torch.randperm(len(self.utterance_ids), generator=generator).tolist()

    def _reset_scores(self) -> None:
        self._sequence_nll = [None] * len(self)
        self._valid_tokens = [None] * len(self)

    def record(
        self,
        dataset_indices: Sequence[int],
        sequence_nll: Sequence[float],
        valid_tokens: Sequence[int],
    ) -> None:
        if not (len(dataset_indices) == len(sequence_nll) == len(valid_tokens)):
            raise ValueError("S2S score batches must have equal lengths")
        with self._lock:
            for index, nll, token_count in zip(
                dataset_indices, sequence_nll, valid_tokens
            ):
                dataset_index = int(index)
                numeric_nll = float(nll)
                numeric_tokens = int(token_count)
                if not 0 <= dataset_index < len(self):
                    raise IndexError(f"S2S dataset index out of range: {dataset_index}")
                if self._sequence_nll[dataset_index] is not None:
                    raise RuntimeError(
                        f"Duplicate S2S score for dataset index {dataset_index} "
                        f"in epoch {self.current_epoch + 1}"
                    )
                if not math.isfinite(numeric_nll) or numeric_nll < 0:
                    raise ValueError("S2S sequence NLL must be finite and non-negative")
                if numeric_tokens <= 0:
                    raise ValueError("S2S token counts must be positive")
                self._sequence_nll[dataset_index] = numeric_nll
                self._valid_tokens[dataset_index] = numeric_tokens

    def _atomic_jsonl(self, path: Path, rows: Sequence[dict]) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(path)

    def write_order_audit(self) -> None:
        with self._lock:
            epoch_number = self.current_epoch + 1
            path = self.artifact_dir / f"epoch_{epoch_number:02d}_order.jsonl"
            if path.exists():
                return
            positions = {index: position for position, index in enumerate(self.order)}
            self._atomic_jsonl(
                path,
                [
                    {
                        "epoch": epoch_number,
                        "position": positions[index],
                        "dataset_index": index,
                        "id": self.utterance_ids[index],
                        "duration": self.durations[index],
                        "ranking_score": (
                            None if self.order_scores is None else self.order_scores[index]
                        ),
                        "score_source_epoch": None if epoch_number == 1 else epoch_number - 1,
                        "strategy": "s2s_margin",
                    }
                    for index in self.order
                ],
            )

    def finalize_current_epoch(self) -> None:
        with self._lock:
            if self.current_epoch in self._finalized_epochs:
                return
            missing = [
                index for index, value in enumerate(self._sequence_nll) if value is None
            ]
            if missing:
                raise RuntimeError(
                    f"Epoch {self.current_epoch + 1} is missing {len(missing)} S2S scores"
                )
            raw_scores = [
                float(self._sequence_nll[index]) / self.durations[index]
                for index in range(len(self))
            ]
            normalized_scores = percentile_ranks(raw_scores)
            next_order = sorted(
                range(len(self)),
                key=lambda index: (
                    raw_scores[index],
                    self.durations[index],
                    self.utterance_ids[index],
                    index,
                ),
            )
            next_positions = {
                index: position for position, index in enumerate(next_order)
            }
            epoch_number = self.current_epoch + 1
            self._atomic_jsonl(
                self.artifact_dir / f"epoch_{epoch_number:02d}_scores.jsonl",
                [
                    {
                        "epoch": epoch_number,
                        "dataset_index": index,
                        "id": self.utterance_ids[index],
                        "duration": self.durations[index],
                        "valid_tokens": self._valid_tokens[index],
                        "sequence_nll": self._sequence_nll[index],
                        "s2s_nll_per_second": raw_scores[index],
                        "percentile_rank": normalized_scores[index],
                        "next_epoch_position": next_positions[index],
                    }
                    for index in range(len(self))
                ],
            )
            self.order = next_order
            self.order_scores = raw_scores
            self._finalized_epochs.add(self.current_epoch)

    def prepare_epoch(self, epoch: int) -> None:
        selected_epoch = int(epoch)
        if selected_epoch < 0:
            raise ValueError("S2S curriculum epoch must be non-negative")
        with self._lock:
            if selected_epoch == self.current_epoch:
                return
            if selected_epoch != self.current_epoch + 1:
                raise RuntimeError(
                    f"Invalid S2S epoch transition {self.current_epoch} -> {selected_epoch}"
                )
            self.finalize_current_epoch()
            self.current_epoch = selected_epoch
            self._reset_scores()


class AuditedS2SLossSampler(Sampler[int]):
    def __init__(self, state: S2SLossCurriculumState) -> None:
        self.state = state
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.state)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.state.prepare_epoch(self.epoch)

    def __iter__(self) -> Iterator[int]:
        self.state.prepare_epoch(self.epoch)
        indices = self.state.order.copy()
        if sorted(indices) != list(range(len(self))):
            raise RuntimeError("S2S curriculum order is not a complete permutation")
        self.state.write_order_audit()
        return iter(indices)


class HybridS2SCurriculumState(S2SLossCurriculumState):
    def __init__(
        self,
        utterance_ids: Sequence[str],
        durations: Sequence[float],
        seed: int,
        artifact_dir: Path,
        acoustic_scores: Sequence[float],
        acoustic_weight: float = 0.5,
        s2s_weight: float = 0.5,
    ) -> None:
        super().__init__(utterance_ids, durations, seed, artifact_dir)
        self.acoustic_scores = [float(value) for value in acoustic_scores]
        self.acoustic_weight = float(acoustic_weight)
        self.s2s_weight = float(s2s_weight)
        if len(self.acoustic_scores) != len(self):
            raise ValueError("Hybrid acoustic scores must align with training rows")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.acoustic_scores
        ):
            raise ValueError("Hybrid acoustic scores must be finite values in [0, 1]")
        if not math.isclose(self.acoustic_weight + self.s2s_weight, 1.0):
            raise ValueError("Hybrid curriculum weights must sum to one")
        if self.acoustic_weight < 0 or self.s2s_weight < 0:
            raise ValueError("Hybrid curriculum weights must be non-negative")

    def write_order_audit(self) -> None:
        with self._lock:
            epoch_number = self.current_epoch + 1
            path = self.artifact_dir / f"epoch_{epoch_number:02d}_order.jsonl"
            if path.exists():
                return
            positions = {index: position for position, index in enumerate(self.order)}
            self._atomic_jsonl(
                path,
                [
                    {
                        "epoch": epoch_number,
                        "position": positions[index],
                        "dataset_index": index,
                        "id": self.utterance_ids[index],
                        "duration": self.durations[index],
                        "ranking_score": (
                            None if self.order_scores is None else self.order_scores[index]
                        ),
                        "score_source_epoch": None if epoch_number == 1 else epoch_number - 1,
                        "strategy": "acoustic_s2s_hybrid",
                    }
                    for index in self.order
                ],
            )

    def finalize_current_epoch(self) -> None:
        with self._lock:
            if self.current_epoch in self._finalized_epochs:
                return
            missing = [
                index for index, value in enumerate(self._sequence_nll) if value is None
            ]
            if missing:
                raise RuntimeError(
                    f"Epoch {self.current_epoch + 1} is missing {len(missing)} S2S scores"
                )
            raw_s2s_scores = [
                float(self._sequence_nll[index]) / self.durations[index]
                for index in range(len(self))
            ]
            s2s_percentiles = percentile_ranks(raw_s2s_scores)
            hybrid_scores = [
                self.acoustic_weight * self.acoustic_scores[index]
                + self.s2s_weight * s2s_percentiles[index]
                for index in range(len(self))
            ]
            next_order = sorted(
                range(len(self)),
                key=lambda index: (
                    hybrid_scores[index],
                    self.durations[index],
                    self.utterance_ids[index],
                    index,
                ),
            )
            next_positions = {
                index: position for position, index in enumerate(next_order)
            }
            epoch_number = self.current_epoch + 1
            self._atomic_jsonl(
                self.artifact_dir / f"epoch_{epoch_number:02d}_scores.jsonl",
                [
                    {
                        "epoch": epoch_number,
                        "dataset_index": index,
                        "id": self.utterance_ids[index],
                        "duration": self.durations[index],
                        "valid_tokens": self._valid_tokens[index],
                        "sequence_nll": self._sequence_nll[index],
                        "s2s_nll_per_second": raw_s2s_scores[index],
                        "s2s_percentile_rank": s2s_percentiles[index],
                        "acoustic_score": self.acoustic_scores[index],
                        "acoustic_weight": self.acoustic_weight,
                        "s2s_weight": self.s2s_weight,
                        "hybrid_score": hybrid_scores[index],
                        "next_epoch_position": next_positions[index],
                    }
                    for index in range(len(self))
                ],
            )
            self.order = next_order
            self.order_scores = hybrid_scores
            self._finalized_epochs.add(self.current_epoch)


class AuditedHybridS2SSampler(AuditedS2SLossSampler):
    state: HybridS2SCurriculumState


class WERMarginCurriculumState(S2SLossCurriculumState):
    def _reset_scores(self) -> None:
        self._wer_records: list[Optional[dict]] = [None] * len(self)

    def record(self, dataset_indices: Sequence[int], records: Sequence[dict]) -> None:
        if len(dataset_indices) != len(records):
            raise ValueError("WER score batches must have equal lengths")
        with self._lock:
            for index, record in zip(dataset_indices, records):
                dataset_index = int(index)
                if not 0 <= dataset_index < len(self):
                    raise IndexError(f"WER dataset index out of range: {dataset_index}")
                if self._wer_records[dataset_index] is not None:
                    raise RuntimeError(
                        f"Duplicate WER score for dataset index {dataset_index} "
                        f"in epoch {self.current_epoch + 1}"
                    )
                raw_wer = float(record["wer"])
                if not math.isfinite(raw_wer) or raw_wer < 0:
                    raise ValueError("WER curriculum scores must be finite and non-negative")
                self._wer_records[dataset_index] = {
                    "reference": str(record["reference"]),
                    "prediction": str(record["prediction"]),
                    "wer": raw_wer,
                    "substitutions": int(record["substitutions"]),
                    "deletions": int(record["deletions"]),
                    "insertions": int(record["insertions"]),
                    "hits": int(record["hits"]),
                    "reference_words": int(record["reference_words"]),
                }

    def write_order_audit(self) -> None:
        with self._lock:
            epoch_number = self.current_epoch + 1
            path = self.artifact_dir / f"epoch_{epoch_number:02d}_order.jsonl"
            if path.exists():
                return
            positions = {index: position for position, index in enumerate(self.order)}
            self._atomic_jsonl(
                path,
                [
                    {
                        "epoch": epoch_number,
                        "position": positions[index],
                        "dataset_index": index,
                        "id": self.utterance_ids[index],
                        "duration": self.durations[index],
                        "ranking_score": (
                            None if self.order_scores is None else self.order_scores[index]
                        ),
                        "score_source_epoch": None if epoch_number == 1 else epoch_number - 1,
                        "strategy": "wer_margin",
                    }
                    for index in self.order
                ],
            )

    def finalize_current_epoch(self) -> None:
        with self._lock:
            if self.current_epoch in self._finalized_epochs:
                return
            missing = [
                index for index, value in enumerate(self._wer_records) if value is None
            ]
            if missing:
                raise RuntimeError(
                    f"Epoch {self.current_epoch + 1} is missing {len(missing)} WER scores"
                )
            records = [record for record in self._wer_records if record is not None]
            raw_scores = [float(record["wer"]) for record in records]
            minimum = min(raw_scores)
            maximum = max(raw_scores)
            score_range = maximum - minimum
            normalized_scores = [
                0.0 if score_range == 0 else (score - minimum) / score_range
                for score in raw_scores
            ]
            next_order = sorted(
                range(len(self)),
                key=lambda index: (
                    raw_scores[index],
                    self.durations[index],
                    self.utterance_ids[index],
                    index,
                ),
            )
            next_positions = {
                index: position for position, index in enumerate(next_order)
            }
            epoch_number = self.current_epoch + 1
            self._atomic_jsonl(
                self.artifact_dir / f"epoch_{epoch_number:02d}_scores.jsonl",
                [
                    {
                        "epoch": epoch_number,
                        "dataset_index": index,
                        "id": self.utterance_ids[index],
                        "duration": self.durations[index],
                        **records[index],
                        "normalized_score": normalized_scores[index],
                        "next_epoch_position": next_positions[index],
                    }
                    for index in range(len(self))
                ],
            )
            self.order = next_order
            self.order_scores = raw_scores
            self._finalized_epochs.add(self.current_epoch)


class AuditedWERMarginSampler(AuditedS2SLossSampler):
    state: WERMarginCurriculumState