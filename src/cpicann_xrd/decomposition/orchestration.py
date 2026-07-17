"""Orchestrate XDecomposer components through CPICANN ranking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

import numpy as np
import torch
from pydantic import Field

from cpicann_xrd.catalog.catalog import CatalogManifest, PhaseCatalog
from cpicann_xrd.core.preprocessing import preprocess_spectrum
from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.schemas import (
    DecomposedComponent,
    XDecomposerPreprocessingMetadata,
    XDecomposerRequest,
    XDecomposerResult,
)
from cpicann_xrd.exceptions import CpicannXrdError
from cpicann_xrd.model.protocol import InferenceBackend
from cpicann_xrd.schemas import FilterSpec, SamplePrediction, SpectrumData, StrictBaseModel
from cpicann_xrd.services.predictor import PredictionService


class DecompositionBackend(Protocol):
    """Backend protocol for XDecomposer-style services."""

    def decompose(self, request: XDecomposerRequest) -> XDecomposerResult:
        """Return a decomposition result for one request."""
        ...


class ComponentError(StrictBaseModel):
    """Serializable per-component failure payload."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class ComponentAxisMetadata(StrictBaseModel):
    """Physical axis used when restoring an XDecomposer component."""

    preprocessing_version: Literal["xdecomposer-v1"] = "xdecomposer-v1"
    two_theta_min: float
    two_theta_max: float
    points: int = Field(gt=0)


class CatalogSnapshot(StrictBaseModel):
    """Catalog metadata captured with component predictions."""

    model_id: str | None = None
    catalog_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=0)


class ComponentIdentificationResult(StrictBaseModel):
    """CPICANN ranking outcome for one decomposed component."""

    component_index: int = Field(ge=0)
    original_slot_index: int = Field(ge=0)
    status: Literal["success", "failed", "skipped"]
    active_probability: float = Field(ge=0.0, le=1.0)
    is_active: bool
    estimated_weight: float = Field(ge=0.0)
    pattern_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xdecomposer_axis: ComponentAxisMetadata
    cpicann_preprocessing_version: str | None = None
    cpicann_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    filter_spec: FilterSpec = Field(default_factory=FilterSpec)
    catalog: CatalogSnapshot | None = None
    cpicann: SamplePrediction | None = None
    warnings: list[str] = Field(default_factory=list)
    error: ComponentError | None = None


class MultiphaseIdentificationResult(StrictBaseModel):
    """Joint decomposition plus per-component CPICANN ranking result."""

    sample_id: str = Field(min_length=1)
    mode: Literal["multiphase-identification"] = "multiphase-identification"
    status: Literal["success", "partial", "failed"]
    requested_top_k: int = Field(gt=0)
    decomposition: XDecomposerResult
    components: list[ComponentIdentificationResult]
    filter_spec: FilterSpec = Field(default_factory=FilterSpec)
    warnings: list[str] = Field(default_factory=list)
    report_markdown: str

    @property
    def successful_component_count(self) -> int:
        """Return the number of successfully classified components."""
        return sum(1 for component in self.components if component.status == "success")

    @property
    def failed_component_count(self) -> int:
        """Return the number of failed component classifications."""
        return sum(1 for component in self.components if component.status == "failed")


def identify_decomposed_components(
    *,
    request: XDecomposerRequest,
    decomposition_backend: DecompositionBackend,
    phase_backend: InferenceBackend,
    catalog: PhaseCatalog | None = None,
    catalog_manifest: CatalogManifest | None = None,
    top_k: int = 5,
    filter_spec: FilterSpec | None = None,
    component_filter_specs: Mapping[int, FilterSpec] | None = None,
    include_inactive: bool = False,
    low_activity_threshold: float | None = None,
    low_weight_threshold: float = 0.05,
    residual_warning_threshold: float = 0.05,
) -> MultiphaseIdentificationResult:
    """Decompose a spectrum, then rank every usable component with CPICANN."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    active_filter_spec = filter_spec or FilterSpec()
    decomposition_request = request.model_copy(update={"return_component_patterns": True})
    decomposition = decomposition_backend.decompose(decomposition_request)
    if decomposition.preprocessing_version != "xdecomposer-v1":
        raise DecompositionError(
            DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
            "XDecomposer result must use xdecomposer-v1 preprocessing",
            details={"preprocessing_version": decomposition.preprocessing_version},
        )

    warnings = list(decomposition.warnings)
    if decomposition.reconstruction_error > residual_warning_threshold:
        warnings.append(
            "High XDecomposer reconstruction residual; component rankings are conditional."
        )

    prediction_service = PredictionService(phase_backend, catalog=catalog)
    catalog_snapshot = _catalog_snapshot(catalog, catalog_manifest)
    component_results = [
        _identify_one_component(
            sample_id=request.sample_id,
            source_filename=request.source_filename or request.sample_id,
            component=component,
            preprocessing=decomposition.preprocessing,
            prediction_service=prediction_service,
            top_k=top_k,
            filter_spec=_select_filter_spec(
                component,
                default_filter_spec=active_filter_spec,
                component_filter_specs=component_filter_specs,
            ),
            catalog_snapshot=catalog_snapshot,
            include_inactive=include_inactive,
            activity_threshold=low_activity_threshold
            if low_activity_threshold is not None
            else decomposition_request.activity_threshold,
            low_weight_threshold=low_weight_threshold,
            high_residual=decomposition.reconstruction_error > residual_warning_threshold,
        )
        for component in decomposition.components
    ]

    status = _overall_status(component_results)
    result_without_report = MultiphaseIdentificationResult(
        sample_id=request.sample_id,
        status=status,
        requested_top_k=top_k,
        decomposition=decomposition,
        components=component_results,
        filter_spec=active_filter_spec,
        warnings=warnings,
        report_markdown="pending",
    )
    return result_without_report.model_copy(
        update={"report_markdown": render_joint_identification_report(result_without_report)}
    )


def restore_component_spectrum(
    *,
    sample_id: str,
    source_filename: str,
    component: DecomposedComponent,
    preprocessing: XDecomposerPreprocessingMetadata,
) -> SpectrumData:
    """Restore one component to a physical 2-theta spectrum."""
    if component.pattern is None:
        raise DecompositionError(
            DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
            "XDecomposer component pattern is required for CPICANN ranking",
            details={"component_index": component.component_index},
        )
    pattern = np.asarray(component.pattern, dtype=np.float32)
    if pattern.ndim != 1 or pattern.shape[0] != preprocessing.output_points:
        raise DecompositionError(
            DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
            "XDecomposer component pattern length does not match preprocessing metadata",
            details={
                "component_index": component.component_index,
                "expected_points": preprocessing.output_points,
                "actual_shape": list(pattern.shape),
            },
        )
    if not np.isfinite(pattern).all():
        raise DecompositionError(
            DecompositionErrorCode.INVALID_INPUT,
            "XDecomposer component pattern contains non-finite values",
            details={"component_index": component.component_index},
        )

    two_theta = np.linspace(
        preprocessing.two_theta_min,
        preprocessing.two_theta_max,
        preprocessing.output_points,
        dtype=np.float32,
    )
    component_label = f"component-{component.component_index:02d}"
    return SpectrumData(
        sample_id=f"{sample_id}-{component_label}",
        source_filename=f"{source_filename}#{component_label}",
        two_theta=[float(value) for value in two_theta],
        intensity=[float(value) for value in pattern],
        sha256=component.pattern_sha256,
    )


def render_joint_identification_report(result: MultiphaseIdentificationResult) -> str:
    """Render a compact Markdown summary for joint decomposition/classification."""
    lines = [
        f"# Multiphase identification: {result.sample_id}",
        "",
        f"- Status: {result.status}",
        f"- XDecomposer model: {result.decomposition.model_id}",
        f"- Reconstruction RMSE: {result.decomposition.reconstruction_error:.6g}",
        f"- Requested CPICANN Top-K: {result.requested_top_k}",
        "",
        "| Component | Status | Activity | Weight | Candidates | Top prediction | Warnings |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for component in result.components:
        candidate_count = (
            component.cpicann.candidate_count_after_filter if component.cpicann is not None else 0
        )
        top_prediction = "-"
        if component.cpicann is not None and component.cpicann.predictions:
            first = component.cpicann.predictions[0]
            top_prediction = (
                f"{first.phase.formula} "
                f"({first.filtered_confidence:.3f} conditional confidence)"
            )
        warnings = "; ".join(component.warnings) if component.warnings else "-"
        lines.append(
            "| "
            f"{component.component_index} | "
            f"{component.status} | "
            f"{component.active_probability:.3f} | "
            f"{component.estimated_weight:.3f} | "
            f"{candidate_count} | "
            f"{top_prediction} | "
            f"{warnings} |"
        )
    if result.warnings:
        lines.extend(["", "## Run warnings"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _identify_one_component(
    *,
    sample_id: str,
    source_filename: str,
    component: DecomposedComponent,
    preprocessing: XDecomposerPreprocessingMetadata,
    prediction_service: PredictionService,
    top_k: int,
    filter_spec: FilterSpec,
    catalog_snapshot: CatalogSnapshot | None,
    include_inactive: bool,
    activity_threshold: float,
    low_weight_threshold: float,
    high_residual: bool,
) -> ComponentIdentificationResult:
    axis_metadata = ComponentAxisMetadata(
        two_theta_min=preprocessing.two_theta_min,
        two_theta_max=preprocessing.two_theta_max,
        points=preprocessing.output_points,
    )
    warnings = _component_warnings(
        component,
        activity_threshold=activity_threshold,
        low_weight_threshold=low_weight_threshold,
        high_residual=high_residual,
    )
    if not component.is_active and not include_inactive:
        warnings.append("Component is inactive and was skipped before CPICANN ranking.")
        return _component_result(
            component,
            status="skipped",
            axis_metadata=axis_metadata,
            filter_spec=filter_spec,
            catalog_snapshot=catalog_snapshot,
            warnings=warnings,
        )

    try:
        spectrum = restore_component_spectrum(
            sample_id=sample_id,
            source_filename=source_filename,
            component=component,
            preprocessing=preprocessing,
        )
        preprocessed = preprocess_spectrum(spectrum)
        tensor = torch.from_numpy(preprocessed.model_input.astype(np.float32, copy=False))
        prediction = prediction_service.predict_tensor(
            sample_id=spectrum.sample_id,
            source_filename=spectrum.source_filename,
            tensor=tensor,
            top_k=top_k,
            filter_spec=filter_spec,
        )
        warnings.extend(preprocessed.warnings)
        warnings.extend(prediction.warnings)
        return _component_result(
            component,
            status="success",
            axis_metadata=axis_metadata,
            filter_spec=filter_spec,
            catalog_snapshot=catalog_snapshot,
            warnings=warnings,
            cpicann_preprocessing_version=preprocessed.config.name,
            cpicann_input_sha256=preprocessed.array_sha256,
            cpicann=prediction,
        )
    except Exception as exc:  # noqa: BLE001 - component failures must not abort the run.
        return _component_result(
            component,
            status="failed",
            axis_metadata=axis_metadata,
            filter_spec=filter_spec,
            catalog_snapshot=catalog_snapshot,
            warnings=warnings,
            error=_component_error(exc),
        )


def _component_result(
    component: DecomposedComponent,
    *,
    status: Literal["success", "failed", "skipped"],
    axis_metadata: ComponentAxisMetadata,
    filter_spec: FilterSpec,
    catalog_snapshot: CatalogSnapshot | None,
    warnings: list[str],
    cpicann_preprocessing_version: str | None = None,
    cpicann_input_sha256: str | None = None,
    cpicann: SamplePrediction | None = None,
    error: ComponentError | None = None,
) -> ComponentIdentificationResult:
    return ComponentIdentificationResult(
        component_index=component.component_index,
        original_slot_index=component.original_slot_index,
        status=status,
        active_probability=component.active_probability,
        is_active=component.is_active,
        estimated_weight=component.estimated_weight,
        pattern_sha256=component.pattern_sha256,
        xdecomposer_axis=axis_metadata,
        cpicann_preprocessing_version=cpicann_preprocessing_version,
        cpicann_input_sha256=cpicann_input_sha256,
        filter_spec=filter_spec,
        catalog=catalog_snapshot,
        cpicann=cpicann,
        warnings=warnings,
        error=error,
    )


def _select_filter_spec(
    component: DecomposedComponent,
    *,
    default_filter_spec: FilterSpec,
    component_filter_specs: Mapping[int, FilterSpec] | None,
) -> FilterSpec:
    if component_filter_specs is None:
        return default_filter_spec
    return component_filter_specs.get(component.component_index, default_filter_spec)


def _catalog_snapshot(
    catalog: PhaseCatalog | None,
    manifest: CatalogManifest | None,
) -> CatalogSnapshot | None:
    if catalog is None and manifest is None:
        return None
    record_count = len(catalog) if catalog is not None else 0
    if catalog is None and manifest is not None:
        record_count = manifest.record_count
    return CatalogSnapshot(
        model_id=manifest.model_id if manifest is not None else None,
        catalog_sha256=manifest.catalog_sha256 if manifest is not None else None,
        record_count=record_count,
    )


def _component_warnings(
    component: DecomposedComponent,
    *,
    activity_threshold: float,
    low_weight_threshold: float,
    high_residual: bool,
) -> list[str]:
    warnings = list(component.warnings)
    if component.active_probability < activity_threshold:
        warnings.append("Low activity probability; component ranking may be unreliable.")
    if component.estimated_weight < low_weight_threshold:
        warnings.append("Low estimated component weight; ranking may be unstable.")
    if high_residual:
        warnings.append("High reconstruction residual; CPICANN confidence is conditional.")
    return warnings


def _component_error(exc: Exception) -> ComponentError:
    if isinstance(exc, DecompositionError):
        return ComponentError(code=exc.code.value, message=exc.message, details=exc.details)
    if isinstance(exc, CpicannXrdError):
        return ComponentError(
            code=exc.error_code.value,
            message=exc.message,
            details=exc.details,
        )
    return ComponentError(code=exc.__class__.__name__, message=str(exc) or repr(exc))


def _overall_status(components: list[ComponentIdentificationResult]) -> Literal[
    "success", "partial", "failed"
]:
    if components and all(component.status == "success" for component in components):
        return "success"
    if any(component.status == "success" for component in components):
        return "partial"
    return "failed"
