"""Model manifest schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from cpicann_xrd.schemas import StrictBaseModel


class ModelManifest(StrictBaseModel):
    """Version-locked model and catalog manifest."""

    model_id: str = Field(min_length=1)
    backend: Literal["fake", "cpicann"]
    num_classes: int = Field(gt=0)
    architecture: str = Field(min_length=1)
    preprocessing_version: str = "legacy-cpicann-v1"
    source_url: str | None = None
    source_revision: str | None = None
    weight_path: Path | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    catalog_path: Path | None = None
    catalog_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("weight_path", "catalog_path", mode="before")
    @classmethod
    def normalize_optional_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value)
