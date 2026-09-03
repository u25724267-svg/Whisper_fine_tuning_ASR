import json
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler


def percentile_ranks(values: Sequence[float]) -> list[float]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("Percentile ranks require at least one value")
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("Percentile rank values must be finite")
    if len(numeric_values) == 1:
        return [0.5]

    sorted_indices = sorted(range(len(numeric_values)), key=lambda index: numeric_values[index])
    ranks = [0.0] * len(numeric_values)
    start = 0
    while start < len(sorted_indices):
        end = start + 1
        value = numeric_values[sorted_indices[start]]
        while end < len(sorted_indices) and numeric_values[sorted_indices[end]] == value:
            end += 1
        average_rank = (start + end - 1) / 2
        percentile = average_rank / (len(numeric_values) - 1)
        for position in range(start, end):
            ranks[sorted_indices[position]] = percentile
        start = end
    return ranks


class AuditedStaticScoreSampler(Sampler[int]):
    def __init__(
        self,
        utterance_ids: Sequence[str],
        difficulty_scores: Sequence[float],
        strategy: str,
        score_metadata: Optional[Dict[str, Sequence[float]]] = None,
        artifact_dir: Optional[Path] = None,
    ) -> None:
        self.utterance_ids = [str(utterance_id) for utterance_id in utterance_ids]
        self.difficulty_scores = [float(score) for score in difficulty_scores]
        self.strategy = str(strategy)
        self.score_metadata = {
            name: [float(value) for value in values]
            for name, values in (score_metadata or {}).items()
        }
        self.artifact_dir = artifact_dir
        self.epoch = 0

        if len(self.utterance_ids) != len(self.difficulty_scores):
            raise ValueError("Static curriculum IDs and scores must have equal length")
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("Static curriculum utterance IDs must be unique")
        if not self.utterance_ids:
            raise ValueError("Static curriculum requires at least one utterance")
        if any(not math.isfinite(score) for score in self.difficulty_scores):
            raise ValueError("Static curriculum scores must be finite")
        for name, values in self.score_metadata.items():
            if len(values) != len(self.utterance_ids):
                raise ValueError(f"Static curriculum metadata '{name}' is misaligned")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Static curriculum metadata '{name}' must be finite")

        self.sorted_indices = sorted(
            range(len(self.difficulty_scores)),
            key=lambda index: (
                self.difficulty_scores[index],
                self.utterance_ids[index],
                index,
            ),
        )

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Static curriculum epoch must be non-negative")
        self.epoch = int(epoch)

    def indices_for_epoch(self) -> list[int]:
        return self.sorted_indices.copy()

    def _write_audit(self, indices: Sequence[int]) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self.artifact_dir / f"epoch_{self.epoch + 1:02d}_order.jsonl"
        if audit_path.exists():
            return
        temporary_path = audit_path.with_suffix(f"{audit_path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as destination:
            for position, index in enumerate(indices):
                destination.write(
                    json.dumps(
                        {
                            "epoch": self.epoch + 1,
                            "position": position,
                            "dataset_index": index,
                            "id": self.utterance_ids[index],
                            "difficulty_score": self.difficulty_scores[index],
                            "strategy": self.strategy,
                            **{
                                name: values[index]
                                for name, values in self.score_metadata.items()
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        temporary_path.replace(audit_path)

    def __iter__(self) -> Iterator[int]:
        indices = self.indices_for_epoch()
        if sorted(indices) != list(range(len(self))):
            raise RuntimeError("Static curriculum order is not a complete dataset permutation")
        self._write_audit(indices)
        return iter(indices)


class AuditedCumulativePacingSampler(Sampler[int]):
    def __init__(
        self,
        utterance_ids: Sequence[str],
        priority_scores: Sequence[float],
        strategy: str,
        seed: int,
        eligible_fractions: Sequence[float] = (1 / 3, 2 / 3, 1.0),
        score_metadata: Optional[Dict[str, Sequence[float]]] = None,
        artifact_dir: Optional[Path] = None,
    ) -> None:
        self.utterance_ids = [str(utterance_id) for utterance_id in utterance_ids]
        self.priority_scores = [float(score) for score in priority_scores]
        self.strategy = str(strategy)
        self.seed = int(seed)
        self.eligible_fractions = [float(fraction) for fraction in eligible_fractions]
        self.score_metadata = {
            name: [float(value) for value in values]
            for name, values in (score_metadata or {}).items()
        }
        self.artifact_dir = artifact_dir
        self.epoch = 0

        if len(self.utterance_ids) != len(self.priority_scores):
            raise ValueError("Pacing IDs and priority scores must have equal length")
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("Pacing utterance IDs must be unique")
        if not self.utterance_ids:
            raise ValueError("Pacing requires at least one utterance")
        if any(not math.isfinite(score) for score in self.priority_scores):
            raise ValueError("Pacing priority scores must be finite")
        if not self.eligible_fractions or self.eligible_fractions[-1] != 1.0:
            raise ValueError("Pacing fractions must end at 1.0")
        if any(
            fraction <= 0 or fraction > 1
            for fraction in self.eligible_fractions
        ):
            raise ValueError("Pacing fractions must be in (0, 1]")
        if any(
            left >= right
            for left, right in zip(
                self.eligible_fractions, self.eligible_fractions[1:]
            )
        ):
            raise ValueError("Pacing fractions must be strictly increasing")
        for name, values in self.score_metadata.items():
            if len(values) != len(self.utterance_ids):
                raise ValueError(f"Pacing metadata '{name}' is misaligned")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Pacing metadata '{name}' must be finite")

        self.priority_indices = sorted(
            range(len(self.priority_scores)),
            key=lambda index: (
                self.priority_scores[index],
                self.utterance_ids[index],
                index,
            ),
        )

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Pacing epoch must be non-negative")
        self.epoch = int(epoch)

    def eligible_count(self, epoch: int) -> int:
        fraction = self.eligible_fractions[min(epoch, len(self.eligible_fractions) - 1)]
        return min(len(self), math.ceil(len(self) * fraction))

    def eligible_indices(self, epoch: int) -> list[int]:
        return self.priority_indices[: self.eligible_count(epoch)]

    def indices_for_epoch(self, epoch: Optional[int] = None) -> list[int]:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        eligible = self.eligible_indices(selected_epoch)
        generator = torch.Generator()
        generator.manual_seed(self.seed + selected_epoch)
        shuffled = [eligible[index] for index in torch.randperm(len(eligible), generator=generator)]
        repeats, remainder = divmod(len(self), len(shuffled))
        presentations = shuffled * repeats + shuffled[:remainder]
        final_order = torch.randperm(len(presentations), generator=generator).tolist()
        return [presentations[index] for index in final_order]

    def _write_priority_audit(self) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        priority_path = self.artifact_dir / "priority_ranking.jsonl"
        if priority_path.exists():
            return
        temporary_path = priority_path.with_name(
            f".{priority_path.name}.{uuid.uuid4().hex}.tmp"
        )
        stage_counts = [self.eligible_count(epoch) for epoch in range(len(self.eligible_fractions))]
        with temporary_path.open("w", encoding="utf-8") as destination:
            for priority_rank, index in enumerate(self.priority_indices):
                first_stage = next(
                    stage + 1
                    for stage, count in enumerate(stage_counts)
                    if priority_rank < count
                )
                destination.write(
                    json.dumps(
                        {
                            "priority_rank": priority_rank,
                            "dataset_index": index,
                            "id": self.utterance_ids[index],
                            "priority_score": self.priority_scores[index],
                            "first_eligible_epoch": first_stage,
                            "strategy": self.strategy,
                            **{
                                name: values[index]
                                for name, values in self.score_metadata.items()
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        temporary_path.replace(priority_path)

    def _write_epoch_audit(self, indices: Sequence[int]) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self.artifact_dir / f"epoch_{self.epoch + 1:02d}_order.jsonl"
        if audit_path.exists():
            return
        temporary_path = audit_path.with_name(
            f".{audit_path.name}.{uuid.uuid4().hex}.tmp"
        )
        occurrence_counts: Dict[int, int] = {}
        with temporary_path.open("w", encoding="utf-8") as destination:
            for position, index in enumerate(indices):
                occurrence_counts[index] = occurrence_counts.get(index, 0) + 1
                destination.write(
                    json.dumps(
                        {
                            "epoch": self.epoch + 1,
                            "position": position,
                            "dataset_index": index,
                            "id": self.utterance_ids[index],
                            "occurrence": occurrence_counts[index],
                            "eligible_unique_rows": self.eligible_count(self.epoch),
                            "priority_score": self.priority_scores[index],
                            "strategy": self.strategy,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        temporary_path.replace(audit_path)

    def __iter__(self) -> Iterator[int]:
        indices = self.indices_for_epoch()
        eligible = set(self.eligible_indices(self.epoch))
        if len(indices) != len(self) or set(indices) != eligible:
            raise RuntimeError("Pacing order does not match its exposure contract")
        occurrence_counts = Counter(indices)
        if max(occurrence_counts.values()) - min(occurrence_counts.values()) > 1:
            raise RuntimeError("Pacing repetition counts differ by more than one")
        self._write_priority_audit()
        self._write_epoch_audit(indices)
        return iter(indices)


class AuditedSortaGradSampler(Sampler[int]):
    def __init__(
        self,
        utterance_ids: Sequence[str],
        durations: Sequence[float],
        seed: int,
        artifact_dir: Optional[Path] = None,
    ) -> None:
        self.utterance_ids = [str(utterance_id) for utterance_id in utterance_ids]
        self.durations = [float(duration) for duration in durations]
        self.seed = int(seed)
        self.artifact_dir = artifact_dir
        self.epoch = 0

        if len(self.utterance_ids) != len(self.durations):
            raise ValueError("SortaGrad IDs and durations must have equal length")
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("SortaGrad utterance IDs must be unique")
        if not self.utterance_ids:
            raise ValueError("SortaGrad requires at least one utterance")
        if any(not math.isfinite(duration) or duration <= 0 for duration in self.durations):
            raise ValueError("SortaGrad durations must be finite and positive")

        self.sorted_indices = sorted(
            range(len(self.durations)),
            key=lambda index: (
                self.durations[index],
                self.utterance_ids[index],
                index,
            ),
        )

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("SortaGrad epoch must be non-negative")
        self.epoch = int(epoch)

    def indices_for_epoch(self, epoch: Optional[int] = None) -> list[int]:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        if selected_epoch == 0:
            return self.sorted_indices.copy()
        generator = torch.Generator()
        generator.manual_seed(self.seed + selected_epoch)
        return torch.randperm(len(self), generator=generator).tolist()

    def _write_audit(self, indices: Sequence[int]) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self.artifact_dir / f"epoch_{self.epoch + 1:02d}_order.jsonl"
        if audit_path.exists():
            return
        temporary_path = audit_path.with_suffix(f"{audit_path.suffix}.tmp")
        strategy = "duration_ascending" if self.epoch == 0 else "seeded_random"
        with temporary_path.open("w", encoding="utf-8") as destination:
            for position, index in enumerate(indices):
                destination.write(
                    json.dumps(
                        {
                            "epoch": self.epoch + 1,
                            "position": position,
                            "dataset_index": index,
                            "id": self.utterance_ids[index],
                            "duration": self.durations[index],
                            "strategy": strategy,
                            "seed": None if self.epoch == 0 else self.seed + self.epoch,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        temporary_path.replace(audit_path)

    def __iter__(self) -> Iterator[int]:
        indices = self.indices_for_epoch()
        if sorted(indices) != list(range(len(self))):
            raise RuntimeError("SortaGrad order is not a complete dataset permutation")
        self._write_audit(indices)
        return iter(indices)