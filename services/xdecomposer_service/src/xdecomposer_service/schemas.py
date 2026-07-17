"""Response schemas for the isolated XDecomposer service."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base schema with stable external fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentInfo(StrictBaseModel):
    """Python, PyTorch and device information."""

    python: str
    expected_python: str
    python_matches_expected: bool
    torch: str | None
    torch_available: bool
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str] = Field(default_factory=list)


class AssetStatus(StrictBaseModel):
    """Manifest and local asset availability."""

    manifest_path: str
    assets_present: bool
    manifest_valid: bool
    model_id: str | None = None
    xrd_length: int | None = None
    num_sources: int | None = None
    message: str


class UpstreamStatus(StrictBaseModel):
    """Whether the upstream XDecomposer code is importable."""

    source_dir: str | None = None
    importable: bool
    module: str | None = None
    message: str


class HealthResponse(StrictBaseModel):
    """Liveness response."""

    status: Literal["ok"] = "ok"
    service: Literal["xdecomposer-service"] = "xdecomposer-service"


class ReadyResponse(StrictBaseModel):
    """Readiness response."""

    status: Literal["ready", "not_ready"]
    environment: EnvironmentInfo
    assets: AssetStatus
    upstream: UpstreamStatus
    warnings: list[str] = Field(default_factory=list)


class InfoResponse(StrictBaseModel):
    """Runtime info response."""

    service: Literal["xdecomposer-service"] = "xdecomposer-service"
    environment: EnvironmentInfo
    assets: AssetStatus
    upstream: UpstreamStatus


class XDecomposerRequest(StrictBaseModel):
    """One product-level decomposition request."""

    sample_id: str = Field(min_length=1)
    two_theta: list[float] = Field(min_length=1)
    intensity: list[float] = Field(min_length=1)
    max_sources: int = Field(default=4, ge=1, le=16)
    activity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    reference_top_k: int = Field(default=5, ge=0, le=100)
    return_component_patterns: bool = False
    source_filename: str | None = None

    @field_validator("two_theta", "intensity")
    @classmethod
    def values_must_be_finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("spectrum values must be finite")
        return values

    @model_validator(mode="after")
    def lengths_must_match(self) -> XDecomposerRequest:
        if len(self.two_theta) != len(self.intensity):
            raise ValueError("two_theta and intensity must have the same length")
        return self


class XDecomposerPreprocessingMetadata(StrictBaseModel):
    """Deterministic preprocessing metadata."""

    preprocessing_version: Literal["xdecomposer-v1"] = "xdecomposer-v1"
    two_theta_min: float = 10.0
    two_theta_max: float = 80.0
    output_points: int = 3500
    normalization: Literal["max_to_1"] = "max_to_1"
    input_points: int = Field(ge=0)
    finite_points: int = Field(ge=0)
    unique_points: int = Field(ge=0)
    clipped_points: int = Field(ge=0)
    intensity_max_before_normalization: float = Field(ge=0.0)
    corrections: list[str] = Field(default_factory=list)
    tensor_shape: tuple[int, int, int] = (1, 1, 3500)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReferenceMatch(StrictBaseModel):
    """One optional reference-bank match for a component."""

    rank: int = Field(ge=1)
    reference_id: str = Field(min_length=1)
    similarity: float
    source_database: str | None = None
    cod_id: str | None = None
    formula: str | None = None
    space_group: str | None = None


class DecomposedComponent(StrictBaseModel):
    """One decomposed component slot."""

    component_index: int = Field(ge=0)
    original_slot_index: int = Field(ge=0)
    active_probability: float = Field(ge=0.0, le=1.0)
    is_active: bool
    estimated_weight: float = Field(ge=0.0)
    pattern_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pattern: list[float] | None = None
    reference_matches: list[ReferenceMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class XDecomposerResult(StrictBaseModel):
    """Product-level decomposition result."""

    sample_id: str = Field(min_length=1)
    backend: Literal["xdecomposer"] = "xdecomposer"
    model_id: str = Field(min_length=1)
    preprocessing_version: Literal["xdecomposer-v1"] = "xdecomposer-v1"
    status: Literal["success", "failed"] = "success"
    components: list[DecomposedComponent]
    reconstruction_error: float = Field(ge=0.0)
    residual_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reconstruction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing: XDecomposerPreprocessingMetadata
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @field_validator("artifacts", mode="before")
    @classmethod
    def stringify_artifacts(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        return {str(key): str(item) for key, item in dict(value).items()}
