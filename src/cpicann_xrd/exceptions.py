"""Stable exceptions and error codes."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable machine-readable error codes."""

    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    EMPTY_FILE = "EMPTY_FILE"
    NO_VALID_NUMERIC_ROWS = "NO_VALID_NUMERIC_ROWS"
    INVALID_COLUMN_COUNT = "INVALID_COLUMN_COUNT"
    NON_FINITE_VALUES = "NON_FINITE_VALUES"
    INVALID_ANGLE_RANGE = "INVALID_ANGLE_RANGE"
    INSUFFICIENT_ANGLE_COVERAGE = "INSUFFICIENT_ANGLE_COVERAGE"
    PREPROCESSING_FAILED = "PREPROCESSING_FAILED"
    MODEL_NOT_INSTALLED = "MODEL_NOT_INSTALLED"
    MODEL_HASH_MISMATCH = "MODEL_HASH_MISMATCH"
    CATALOG_HASH_MISMATCH = "CATALOG_HASH_MISMATCH"
    MODEL_OUTPUT_SIZE_MISMATCH = "MODEL_OUTPUT_SIZE_MISMATCH"
    NO_CANDIDATES_AFTER_FILTER = "NO_CANDIDATES_AFTER_FILTER"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    REPORT_GENERATION_FAILED = "REPORT_GENERATION_FAILED"


class CpicannXrdError(Exception):
    """Base application exception with a stable error code."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a stable serializable error payload."""
        return {
            "code": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }
