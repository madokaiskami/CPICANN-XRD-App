"""Pydantic schemas shared by core services and interfaces."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from cpicann_xrd.exceptions import ErrorCode

ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def _normalize_elements(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        raw_values = [item for item in re.split(r"[\s,;_]+", value) if item]
    else:
        raw_values = list(value)
    elements = frozenset(str(item).strip() for item in raw_values if str(item).strip())
    invalid = sorted(element for element in elements if not ELEMENT_RE.fullmatch(element))
    if invalid:
        msg = f"Invalid element symbols: {', '.join(invalid)}"
        raise ValueError(msg)
    return elements


def _sorted_elements(value: frozenset[str] | set[str]) -> list[str]:
    return sorted(value)


class StrictBaseModel(BaseModel):
    """Base model for externally visible schemas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class SpectrumData(StrictBaseModel):
    """Raw or parsed XRD spectrum data before model tensor conversion."""

    sample_id: str = Field(min_length=1)
    source_filename: str
    two_theta: list[float] = Field(min_length=1)
    intensity: list[float] = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("two_theta", "intensity")
    @classmethod
    def values_must_be_finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("spectrum values must be finite")
        return values

    @model_validator(mode="after")
    def angle_and_intensity_lengths_match(self) -> SpectrumData:
        if len(self.two_theta) != len(self.intensity):
            raise ValueError("two_theta and intensity must have the same length")
        return self


class PreprocessingConfig(StrictBaseModel):
    """Preprocessing protocol configuration."""

    name: str = "legacy-cpicann-v1"
    two_theta_min: float = 10.0
    two_theta_max: float = 80.0
    points: int = Field(default=4500, gt=0)
    interpolation: str = "slinear"
    normalize_max_to: float = Field(default=100.0, gt=0)

    @model_validator(mode="after")
    def angle_range_must_be_valid(self) -> PreprocessingConfig:
        if self.two_theta_min >= self.two_theta_max:
            raise ValueError("two_theta_min must be less than two_theta_max")
        return self


class FilterSpec(StrictBaseModel):
    """Element constraints supplied by a user."""

    include_must: frozenset[str] = Field(default_factory=frozenset)
    allowed_elements: frozenset[str] | None = None

    @field_validator("include_must", mode="before")
    @classmethod
    def normalize_include_must(cls, value: Any) -> frozenset[str]:
        return _normalize_elements(value)

    @field_validator("allowed_elements", mode="before")
    @classmethod
    def normalize_allowed_elements(cls, value: Any) -> frozenset[str] | None:
        if value is None:
            return None
        return _normalize_elements(value)

    @field_serializer("include_must")
    def serialize_include_must(self, value: frozenset[str]) -> list[str]:
        return _sorted_elements(value)

    @field_serializer("allowed_elements")
    def serialize_allowed_elements(self, value: frozenset[str] | None) -> list[str] | None:
        if value is None:
            return None
        return _sorted_elements(value)


class ModelInfo(StrictBaseModel):
    """Runtime model metadata exposed by an inference backend."""

    model_id: str = Field(min_length=1)
    backend: Literal["fake", "cpicann"]
    num_classes: int = Field(gt=0)
    preprocessing_version: str = "legacy-cpicann-v1"
    device: str = "cpu"
    source_revision: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, str] = Field(default_factory=dict)


class PhaseRecord(StrictBaseModel):
    """One class-index-to-phase catalog record."""

    class_index: int = Field(ge=0)
    cod_id: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    reduced_formula: str = Field(min_length=1)
    elements: frozenset[str]
    space_group: str | None = None
    space_group_number: int | None = Field(default=None, ge=1)

    @field_validator("elements", mode="before")
    @classmethod
    def normalize_phase_elements(cls, value: Any) -> frozenset[str]:
        elements = _normalize_elements(value)
        if not elements:
            raise ValueError("phase elements must not be empty")
        return elements

    @field_serializer("elements")
    def serialize_elements(self, value: frozenset[str]) -> list[str]:
        return _sorted_elements(value)


class PredictionItem(StrictBaseModel):
    """One ranked prediction item."""

    filtered_rank: int = Field(ge=1)
    global_rank: int = Field(ge=1)
    class_index: int = Field(ge=0)
    phase: PhaseRecord
    raw_logit: float
    unfiltered_probability: float = Field(ge=0.0, le=1.0)
    filtered_confidence: float = Field(ge=0.0, le=1.0)


class SamplePrediction(StrictBaseModel):
    """Prediction result for one input sample."""

    sample_id: str = Field(min_length=1)
    source_filename: str
    status: Literal["success", "failed", "ignored"]
    backend: Literal["fake", "cpicann"]
    model_id: str
    preprocessing_version: str
    filter_spec: FilterSpec = Field(default_factory=FilterSpec)
    candidate_count_before_filter: int = Field(ge=0)
    candidate_count_after_filter: int = Field(ge=0)
    requested_top_k: int = Field(gt=0)
    returned_top_k: int = Field(ge=0)
    predictions: list[PredictionItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def returned_top_k_matches_predictions(self) -> SamplePrediction:
        if self.returned_top_k != len(self.predictions):
            raise ValueError("returned_top_k must match predictions length")
        return self


class DiagnosticRecord(StrictBaseModel):
    """Stable diagnostic record for ignored or failed files."""

    source_filename: str
    status: Literal["success", "failed", "ignored"]
    stage: str
    error_code: ErrorCode | None = None
    message: str
    rows_read: int | None = Field(default=None, ge=0)
    rows_invalid: int | None = Field(default=None, ge=0)
    angle_min: float | None = None
    angle_max: float | None = None
    ignored_reason: str | None = None


class RunMetadata(StrictBaseModel):
    """Reproducibility metadata for one run."""

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    application: dict[str, str]
    model: ModelInfo
    runtime: dict[str, str]
    preprocessing: PreprocessingConfig
    filter: FilterSpec = Field(default_factory=FilterSpec)
    top_k: int = Field(gt=0)
    inputs: list[dict[str, str]] = Field(default_factory=list)
