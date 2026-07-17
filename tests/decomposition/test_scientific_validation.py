"""Tests for reproducible XDecomposer scientific validation helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cpicann_xrd.decomposition.scientific_validation import (
    evaluate_decomposition_case,
    match_components_permutation_invariant,
    validation_dataset_blueprint,
    write_validation_artifacts,
)


def test_permutation_invariant_matching_handles_swapped_components() -> None:
    target_a = [0.0, 1.0, 0.0, 0.0]
    target_b = [0.0, 0.0, 1.0, 0.0]

    matches = match_components_permutation_invariant(
        target_patterns=[target_a, target_b],
        predicted_patterns=[target_b, target_a],
    )

    assert [(match.target_index, match.predicted_index) for match in matches] == [(0, 1), (1, 0)]
    assert all(match.correlation == pytest.approx(1.0) for match in matches)
    assert all(match.cosine_similarity == pytest.approx(1.0) for match in matches)


def test_evaluate_decomposition_case_records_count_and_minor_phase_failures() -> None:
    metrics = evaluate_decomposition_case(
        case_id="ratio-90-10",
        case_family="ratio_90_10",
        target_patterns=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        predicted_patterns=[[1.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
        target_weights=[0.9, 0.1],
        predicted_weights=[0.95, 0.05],
        reconstruction_rmse=0.12,
        active_source_count_predicted=3,
        component_top1_accuracy=0.5,
        component_top5_recall=1.0,
        notes="intentional failed minor phase",
    )

    assert metrics.status == "failed"
    assert metrics.false_active_slots == 1
    assert metrics.missed_minor_phases == 1
    assert metrics.estimated_weight_mae == pytest.approx(0.05)
    assert metrics.component_top5_recall == 1.0


def test_validation_artifacts_write_raw_metrics_failures_and_plots(tmp_path: Path) -> None:
    passed = evaluate_decomposition_case(
        case_id="synthetic-two-phase",
        case_family="synthetic_two_phase",
        target_patterns=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        predicted_patterns=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        target_weights=[0.7, 0.3],
        predicted_weights=[0.68, 0.32],
        reconstruction_rmse=0.02,
        active_source_count_predicted=2,
    )
    failed = evaluate_decomposition_case(
        case_id="single-phase-extra-slot",
        case_family="single_phase_in_multiphase_mode",
        target_patterns=[[1.0, 0.0, 0.0]],
        predicted_patterns=[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        target_weights=[1.0],
        predicted_weights=[0.8, 0.2],
        reconstruction_rmse=0.2,
        active_source_count_predicted=2,
    )

    paths = write_validation_artifacts(output_dir=tmp_path / "validation", metrics=[passed, failed])

    raw_rows = _read_csv(paths.raw_metrics_csv)
    failed_rows = _read_csv(paths.failed_cases_csv)
    assert [row["case_id"] for row in raw_rows] == [
        "synthetic-two-phase",
        "single-phase-extra-slot",
    ]
    assert [row["case_id"] for row in failed_rows] == ["single-phase-extra-slot"]
    assert paths.reconstruction_rmse_png.read_bytes().startswith(b"\x89PNG")
    assert paths.component_correlation_png.read_bytes().startswith(b"\x89PNG")
    assert "Status: experimental" in paths.summary_md.read_text(encoding="utf-8")
    assert "not a scientific sign-off" in paths.summary_md.read_text(encoding="utf-8")


def test_validation_dataset_blueprint_covers_xd9_required_families() -> None:
    families = {row["case_family"] for row in validation_dataset_blueprint()}

    assert "synthetic_two_phase" in families
    assert "synthetic_three_phase" in families
    assert "ratio_90_10" in families
    assert "ratio_70_30" in families
    assert "ratio_50_50" in families
    assert "ratio_30_70" in families
    assert "experimental_multiphase" in families
    assert "out_of_catalog_component" in families
    assert "single_phase_in_multiphase_mode" in families
    assert "amorphous_background" in families


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
