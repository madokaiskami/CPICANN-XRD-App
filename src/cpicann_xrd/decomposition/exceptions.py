"""Structured XDecomposer product-layer errors."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class DecompositionErrorCode(StrEnum):
    """Stable XDecomposer error codes."""

    ASSET_UNAVAILABLE = "ASSET_UNAVAILABLE"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"
    PREPROCESSING_FAILED = "PREPROCESSING_FAILED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    OUTPUT_SHAPE_MISMATCH = "OUTPUT_SHAPE_MISMATCH"


class DecompositionError(Exception):
    """Base decomposition exception with structured details."""

    def __init__(
        self,
        code: DecompositionErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable error payload."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }
