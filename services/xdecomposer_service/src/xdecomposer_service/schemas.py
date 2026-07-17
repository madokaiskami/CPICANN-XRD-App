"""Response schemas for the isolated XDecomposer service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
