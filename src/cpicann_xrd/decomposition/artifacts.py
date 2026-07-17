"""Artifact generation for XDecomposer results."""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.preprocessing import preprocess_for_xdecomposer
from cpicann_xrd.decomposition.report import render_decomposition_report
from cpicann_xrd.decomposition.schemas import XDecomposerRequest, XDecomposerResult
from cpicann_xrd.reports.exporters import (
    write_csv_atomic,
    write_json_atomic,
    write_observed_png,
    write_text_atomic,
    write_zip_bundle,
)
from cpicann_xrd.schemas import SpectrumData

SUMMARY_FIELDS = [
    "sample_id",
    "component_display_index",
    "original_slot_index",
    "active_probability",
    "is_active",
    "estimated_weight",
    "pattern_sha256",
    "reconstruction_rmse",
    "warning",
]


@dataclass(frozen=True)
class DecompositionArtifactPaths:
    """Generated decomposition artifact paths."""

    run_dir: Path
    summary_csv: Path
    metadata_json: Path
    report_md: Path
    observed_png: Path
    reconstruction_png: Path
    residual_png: Path
    component_overview_png: Path
    diagnostics_csv: Path
    bundle_zip: Path
    component_files: list[Path]


def write_decomposition_artifacts(
    *,
    run_dir: Path,
    request: XDecomposerRequest,
    result: XDecomposerResult,
    source_sha256: str,
    source_filename: str,
    git_commit: str,
    upstream_commit: str | None = None,
    checkpoint_hashes: dict[str, str] | None = None,
    reference_bank_hash: str | None = None,
    device: str = "cpu",
    container_image_digest: str | None = None,
    elapsed_seconds: float | None = None,
    include_inactive: bool = True,
) -> DecompositionArtifactPaths:
    """Write the standard XD-5 decomposition artifact tree."""
    start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    components_dir = run_dir / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = preprocess_for_xdecomposer(request.two_theta, request.intensity)
    component_arrays = _component_arrays(result, include_inactive=include_inactive)
    reconstruction = np.sum(component_arrays, axis=0).astype(np.float32)
    residual = (preprocessed.intensity - reconstruction).astype(np.float32)

    summary_csv = run_dir / "decomposition_summary.csv"
    metadata_json = run_dir / "decomposition_metadata.json"
    report_md = run_dir / "decomposition_report.md"
    observed_png = run_dir / "observed_xrd.png"
    reconstruction_png = run_dir / "reconstruction.png"
    residual_png = run_dir / "residual.png"
    component_overview_png = run_dir / "component_overview.png"
    diagnostics_csv = run_dir / "diagnostics.csv"
    bundle_zip = run_dir / "result_bundle.zip"

    write_csv_atomic(summary_csv, SUMMARY_FIELDS, _summary_rows(result))
    metadata = _metadata_payload(
        request=request,
        result=result,
        source_sha256=source_sha256,
        source_filename=source_filename,
        git_commit=git_commit,
        upstream_commit=upstream_commit,
        checkpoint_hashes=checkpoint_hashes or {},
        reference_bank_hash=reference_bank_hash,
        device=device,
        container_image_digest=container_image_digest,
        elapsed_seconds=elapsed_seconds
        if elapsed_seconds is not None
        else time.perf_counter() - start,
    )
    write_json_atomic(metadata_json, metadata)
    write_csv_atomic(
        diagnostics_csv,
        ["sample_id", "level", "message"],
        [
            {"sample_id": request.sample_id, "level": "warning", "message": warning}
            for warning in result.warnings
        ],
    )

    axis = preprocessed.two_theta.tolist()
    write_observed_png(
        observed_png,
        SpectrumData(
            sample_id=request.sample_id,
            source_filename=source_filename,
            two_theta=request.two_theta,
            intensity=request.intensity,
            sha256=source_sha256,
        ),
    )
    write_observed_png(
        reconstruction_png,
        _spectrum(request.sample_id, "reconstruction", axis, reconstruction.tolist()),
    )
    write_observed_png(
        residual_png,
        _spectrum(request.sample_id, "residual", axis, np.abs(residual).tolist()),
    )
    write_observed_png(
        component_overview_png,
        _spectrum(request.sample_id, "component_overview", axis, reconstruction.tolist()),
    )

    component_files: list[Path] = []
    for display_index, component in enumerate(result.components, start=1):
        if component.pattern is None:
            continue
        csv_path = components_dir / f"component_{display_index:02d}.csv"
        png_path = components_dir / f"component_{display_index:02d}.png"
        write_csv_atomic(
            csv_path,
            ["two_theta", "intensity"],
            [
                {"two_theta": angle, "intensity": value}
                for angle, value in zip(axis, component.pattern, strict=True)
            ],
        )
        write_observed_png(
            png_path,
            _spectrum(request.sample_id, f"component_{display_index:02d}", axis, component.pattern),
        )
        component_files.extend([csv_path, png_path])

    write_text_atomic(
        report_md, render_decomposition_report(request=request, result=result, metadata=metadata)
    )
    write_zip_bundle(run_dir, bundle_zip)
    return DecompositionArtifactPaths(
        run_dir=run_dir,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
        report_md=report_md,
        observed_png=observed_png,
        reconstruction_png=reconstruction_png,
        residual_png=residual_png,
        component_overview_png=component_overview_png,
        diagnostics_csv=diagnostics_csv,
        bundle_zip=bundle_zip,
        component_files=component_files,
    )


def _summary_rows(result: XDecomposerResult) -> list[dict[str, object]]:
    rows = []
    for display_index, component in enumerate(result.components, start=1):
        rows.append(
            {
                "sample_id": result.sample_id,
                "component_display_index": display_index,
                "original_slot_index": component.original_slot_index,
                "active_probability": component.active_probability,
                "is_active": component.is_active,
                "estimated_weight": component.estimated_weight,
                "pattern_sha256": component.pattern_sha256,
                "reconstruction_rmse": result.reconstruction_error,
                "warning": "; ".join(component.warnings),
            }
        )
    return rows


def _metadata_payload(
    *,
    request: XDecomposerRequest,
    result: XDecomposerResult,
    source_sha256: str,
    source_filename: str,
    git_commit: str,
    upstream_commit: str | None,
    checkpoint_hashes: dict[str, str],
    reference_bank_hash: str | None,
    device: str,
    container_image_digest: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    angle_min = min(request.two_theta)
    angle_max = max(request.two_theta)
    return {
        "schema_version": "1.0",
        "sample_id": request.sample_id,
        "input_sha256": source_sha256,
        "source_filename": source_filename,
        "preprocessing_protocol": result.preprocessing_version,
        "input_angle_range": [angle_min, angle_max],
        "xdecomposer_axis": [
            result.preprocessing.two_theta_min,
            result.preprocessing.two_theta_max,
            result.preprocessing.output_points,
        ],
        "model_id": result.model_id,
        "upstream_commit": upstream_commit or "UNKNOWN",
        "checkpoint_hashes": checkpoint_hashes,
        "reference_bank_hash": reference_bank_hash or "UNKNOWN",
        "activity_threshold": request.activity_threshold,
        "max_sources": request.max_sources,
        "device": device,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": "not_loaded",
            "cuda": "not_loaded",
        },
        "git_commit": git_commit,
        "container_image_digest": container_image_digest or "UNKNOWN",
        "elapsed_seconds": elapsed_seconds,
        "warnings": result.warnings,
    }


def _component_arrays(
    result: XDecomposerResult,
    *,
    include_inactive: bool,
) -> np.ndarray:
    arrays = []
    for component in result.components:
        if not include_inactive and not component.is_active:
            continue
        if component.pattern is None:
            raise DecompositionError(
                DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
                "生成分解工件需要 component pattern",
                details={"component_index": component.component_index},
            )
        arrays.append(np.asarray(component.pattern, dtype=np.float32))
    if not arrays:
        raise DecompositionError(
            DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
            "没有可写出的 component pattern",
        )
    return np.stack(arrays, axis=0)


def _spectrum(
    sample_id: str,
    source_filename: str,
    two_theta: list[float],
    intensity: list[float],
) -> SpectrumData:
    return SpectrumData(
        sample_id=sample_id,
        source_filename=source_filename,
        two_theta=two_theta,
        intensity=intensity,
        sha256=hashlib.sha256(
            np.ascontiguousarray(np.asarray(intensity, dtype=np.float32)).tobytes()
        ).hexdigest(),
    )
