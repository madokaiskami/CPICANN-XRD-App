"""Reproducible scientific validation helpers for decomposition workflows."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from cpicann_xrd.reports.exporters import write_csv_atomic, write_observed_png, write_text_atomic
from cpicann_xrd.schemas import SpectrumData

ValidationStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class ComponentMatch:
    """One target-to-predicted component match."""

    target_index: int
    predicted_index: int
    correlation: float
    cosine_similarity: float
    rmse: float


@dataclass(frozen=True)
class ValidationCaseMetrics:
    """Raw metrics for one validation case."""

    case_id: str
    case_family: str
    status: ValidationStatus
    active_source_count_expected: int
    active_source_count_predicted: int
    active_source_count_correct: bool
    reconstruction_rmse: float
    mean_component_correlation: float
    mean_cosine_similarity: float
    estimated_weight_mae: float
    false_active_slots: int
    missed_minor_phases: int
    component_top1_accuracy: float | None = None
    component_top5_recall: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class ValidationArtifactPaths:
    """Generated scientific validation artifact paths."""

    output_dir: Path
    raw_metrics_csv: Path
    failed_cases_csv: Path
    summary_md: Path
    reconstruction_rmse_png: Path
    component_correlation_png: Path


def match_components_permutation_invariant(
    *,
    target_patterns: list[list[float]] | npt.NDArray[np.float32],
    predicted_patterns: list[list[float]] | npt.NDArray[np.float32],
) -> list[ComponentMatch]:
    """Match predicted components to targets by maximizing total correlation."""
    targets = _as_2d_float32(target_patterns, name="target_patterns")
    predicted = _as_2d_float32(predicted_patterns, name="predicted_patterns")
    if targets.shape[1] != predicted.shape[1]:
        raise ValueError("target and predicted patterns must have the same point count")
    if targets.shape[0] == 0 or predicted.shape[0] == 0:
        return []

    match_count = min(targets.shape[0], predicted.shape[0])
    best_score = -math.inf
    best_assignment: tuple[int, ...] | None = None
    for assignment in itertools.permutations(range(predicted.shape[0]), r=match_count):
        score = sum(
            pearson_correlation(targets[target_index], predicted[predicted_index])
            for target_index, predicted_index in enumerate(assignment)
        )
        if score > best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        return []
    return [
        ComponentMatch(
            target_index=target_index,
            predicted_index=predicted_index,
            correlation=pearson_correlation(targets[target_index], predicted[predicted_index]),
            cosine_similarity=cosine_similarity(
                targets[target_index],
                predicted[predicted_index],
            ),
            rmse=rmse(targets[target_index], predicted[predicted_index]),
        )
        for target_index, predicted_index in enumerate(best_assignment)
    ]


def evaluate_decomposition_case(
    *,
    case_id: str,
    case_family: str,
    target_patterns: list[list[float]] | npt.NDArray[np.float32],
    predicted_patterns: list[list[float]] | npt.NDArray[np.float32],
    target_weights: list[float],
    predicted_weights: list[float],
    reconstruction_rmse: float,
    active_source_count_predicted: int,
    minor_phase_threshold: float = 0.15,
    component_top1_accuracy: float | None = None,
    component_top5_recall: float | None = None,
    notes: str = "",
) -> ValidationCaseMetrics:
    """Evaluate one validation case without applying pass/fail threshold tuning."""
    targets = _as_2d_float32(target_patterns, name="target_patterns")
    predicted = _as_2d_float32(predicted_patterns, name="predicted_patterns")
    matches = match_components_permutation_invariant(
        target_patterns=targets,
        predicted_patterns=predicted,
    )
    expected_active = targets.shape[0]
    false_active_slots = max(0, active_source_count_predicted - expected_active)
    missed_minor_phases = _missed_minor_phase_count(
        target_weights=target_weights,
        matches=matches,
        correlation_threshold=0.5,
        minor_phase_threshold=minor_phase_threshold,
    )
    status: ValidationStatus = "passed" if matches and len(matches) == expected_active else "failed"
    if active_source_count_predicted != expected_active or missed_minor_phases:
        status = "failed"

    return ValidationCaseMetrics(
        case_id=case_id,
        case_family=case_family,
        status=status,
        active_source_count_expected=expected_active,
        active_source_count_predicted=active_source_count_predicted,
        active_source_count_correct=active_source_count_predicted == expected_active,
        reconstruction_rmse=reconstruction_rmse,
        mean_component_correlation=_mean([match.correlation for match in matches]),
        mean_cosine_similarity=_mean([match.cosine_similarity for match in matches]),
        estimated_weight_mae=_weight_mae(
            target_weights=target_weights,
            predicted_weights=predicted_weights,
            matches=matches,
        ),
        false_active_slots=false_active_slots,
        missed_minor_phases=missed_minor_phases,
        component_top1_accuracy=component_top1_accuracy,
        component_top5_recall=component_top5_recall,
        notes=notes,
    )


def validation_dataset_blueprint() -> list[dict[str, object]]:
    """Return the minimum XD-9 validation dataset design as machine-readable rows."""
    return [
        {"case_family": "synthetic_two_phase", "human_required": False},
        {"case_family": "synthetic_three_phase", "human_required": False},
        {"case_family": "ratio_90_10", "human_required": False},
        {"case_family": "ratio_70_30", "human_required": False},
        {"case_family": "ratio_50_50", "human_required": False},
        {"case_family": "ratio_30_70", "human_required": False},
        {"case_family": "minor_impurity_phase", "human_required": False},
        {"case_family": "noise_sweep", "human_required": False},
        {"case_family": "background_shift", "human_required": False},
        {"case_family": "peak_position_shift", "human_required": False},
        {"case_family": "peak_width_variation", "human_required": False},
        {"case_family": "experimental_multiphase", "human_required": True},
        {"case_family": "out_of_catalog_component", "human_required": True},
        {"case_family": "correct_element_constraints", "human_required": False},
        {"case_family": "incorrect_element_constraints", "human_required": False},
        {"case_family": "single_phase_in_multiphase_mode", "human_required": False},
        {"case_family": "sources_exceed_checkpoint_num_sources", "human_required": True},
        {"case_family": "amorphous_background", "human_required": True},
    ]


def write_validation_artifacts(
    *,
    output_dir: Path,
    metrics: list[ValidationCaseMetrics],
) -> ValidationArtifactPaths:
    """Write raw metrics, failed cases and simple evaluation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics_csv = output_dir / "raw_metrics.csv"
    failed_cases_csv = output_dir / "failed_cases.csv"
    summary_md = output_dir / "scientific_validation_summary.md"
    reconstruction_rmse_png = output_dir / "reconstruction_rmse.png"
    component_correlation_png = output_dir / "component_correlation.png"

    rows = [_metrics_row(item) for item in metrics]
    write_csv_atomic(raw_metrics_csv, list(rows[0]) if rows else _metric_fields(), rows)
    failed_rows = [row for row in rows if row["status"] == "failed"]
    write_csv_atomic(failed_cases_csv, list(rows[0]) if rows else _metric_fields(), failed_rows)
    write_text_atomic(summary_md, render_validation_summary(metrics))
    _write_metric_plot(
        reconstruction_rmse_png,
        sample_id="reconstruction_rmse",
        values=[item.reconstruction_rmse for item in metrics],
    )
    _write_metric_plot(
        component_correlation_png,
        sample_id="mean_component_correlation",
        values=[item.mean_component_correlation for item in metrics],
    )
    return ValidationArtifactPaths(
        output_dir=output_dir,
        raw_metrics_csv=raw_metrics_csv,
        failed_cases_csv=failed_cases_csv,
        summary_md=summary_md,
        reconstruction_rmse_png=reconstruction_rmse_png,
        component_correlation_png=component_correlation_png,
    )


def render_validation_summary(metrics: list[ValidationCaseMetrics]) -> str:
    """Render a Markdown validation summary that avoids scientific conclusions."""
    total = len(metrics)
    failed = sum(1 for item in metrics if item.status == "failed")
    lines = [
        "# XDecomposer Scientific Validation Summary",
        "",
        "Status: experimental",
        "",
        "This file reports reproducible metrics only. It is not a scientific sign-off.",
        "",
        f"- Cases evaluated: {total}",
        f"- Failed cases retained: {failed}",
        f"- Mean reconstruction RMSE: {_mean([item.reconstruction_rmse for item in metrics]):.6g}",
        "- Scientific conclusion: pending human review",
        "",
        "## Required Human Review",
        "",
        "- Ground-truth reliability",
        "- Physical validity of synthetic mixtures",
        "- Acceptance thresholds for minor phases and unknown phases",
        "- Whether estimated weights may be shown to users",
    ]
    return "\n".join(lines) + "\n"


def pearson_correlation(
    left: npt.NDArray[np.float32],
    right: npt.NDArray[np.float32],
) -> float:
    """Return Pearson correlation with deterministic zero-vector handling."""
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    _validate_same_shape(left64, right64)
    centered_left = left64 - float(np.mean(left64))
    centered_right = right64 - float(np.mean(right64))
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denominator <= 0:
        return 0.0
    return float(np.dot(centered_left, centered_right) / denominator)


def cosine_similarity(
    left: npt.NDArray[np.float32],
    right: npt.NDArray[np.float32],
) -> float:
    """Return spectral-angle-style cosine similarity."""
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    _validate_same_shape(left64, right64)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left64, right64) / denominator)


def rmse(left: npt.NDArray[np.float32], right: npt.NDArray[np.float32]) -> float:
    """Return root mean square error."""
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    _validate_same_shape(left64, right64)
    return float(math.sqrt(float(np.mean((left64 - right64) ** 2))))


def _metrics_row(item: ValidationCaseMetrics) -> dict[str, object]:
    return {
        "case_id": item.case_id,
        "case_family": item.case_family,
        "status": item.status,
        "active_source_count_expected": item.active_source_count_expected,
        "active_source_count_predicted": item.active_source_count_predicted,
        "active_source_count_correct": item.active_source_count_correct,
        "reconstruction_rmse": item.reconstruction_rmse,
        "mean_component_correlation": item.mean_component_correlation,
        "mean_cosine_similarity": item.mean_cosine_similarity,
        "estimated_weight_mae": item.estimated_weight_mae,
        "false_active_slots": item.false_active_slots,
        "missed_minor_phases": item.missed_minor_phases,
        "component_top1_accuracy": item.component_top1_accuracy
        if item.component_top1_accuracy is not None
        else "",
        "component_top5_recall": item.component_top5_recall
        if item.component_top5_recall is not None
        else "",
        "notes": item.notes,
    }


def _metric_fields() -> list[str]:
    return list(_metrics_row(_empty_metric()).keys())


def _empty_metric() -> ValidationCaseMetrics:
    return ValidationCaseMetrics(
        case_id="",
        case_family="",
        status="failed",
        active_source_count_expected=0,
        active_source_count_predicted=0,
        active_source_count_correct=False,
        reconstruction_rmse=0.0,
        mean_component_correlation=0.0,
        mean_cosine_similarity=0.0,
        estimated_weight_mae=0.0,
        false_active_slots=0,
        missed_minor_phases=0,
    )


def _write_metric_plot(path: Path, *, sample_id: str, values: list[float]) -> None:
    if not values:
        values = [0.0]
    spectrum = SpectrumData(
        sample_id=sample_id,
        source_filename=f"{sample_id}.png",
        two_theta=[float(index) for index in range(len(values))],
        intensity=[float(value) for value in values],
    )
    write_observed_png(path, spectrum)


def _as_2d_float32(
    value: list[list[float]] | npt.NDArray[np.float32],
    *,
    name: str,
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_same_shape(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> None:
    if left.shape != right.shape:
        raise ValueError("arrays must have the same shape")


def _weight_mae(
    *,
    target_weights: list[float],
    predicted_weights: list[float],
    matches: list[ComponentMatch],
) -> float:
    errors = []
    for match in matches:
        if match.target_index >= len(target_weights) or match.predicted_index >= len(
            predicted_weights
        ):
            continue
        errors.append(
            abs(target_weights[match.target_index] - predicted_weights[match.predicted_index])
        )
    return _mean(errors)


def _missed_minor_phase_count(
    *,
    target_weights: list[float],
    matches: list[ComponentMatch],
    correlation_threshold: float,
    minor_phase_threshold: float,
) -> int:
    missed = 0
    by_target = {match.target_index: match for match in matches}
    for index, weight in enumerate(target_weights):
        if weight > minor_phase_threshold:
            continue
        match = by_target.get(index)
        if match is None or match.correlation < correlation_threshold:
            missed += 1
    return missed


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
