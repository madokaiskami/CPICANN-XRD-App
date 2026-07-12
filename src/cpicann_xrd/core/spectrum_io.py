"""Spectrum file reading and validation."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from cpicann_xrd.exceptions import ErrorCode
from cpicann_xrd.schemas import DiagnosticRecord, SpectrumData, StrictBaseModel

SUPPORTED_SPECTRUM_EXTENSIONS = frozenset({".txt", ".csv", ".xy"})
TOKEN_SPLIT_RE = re.compile(r"[,\s]+")


class SpectrumReadResult(StrictBaseModel):
    """Result of attempting to read one spectrum file."""

    status: Literal["success", "failed", "ignored"]
    spectrum: SpectrumData | None = None
    diagnostics: list[DiagnosticRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rows_read: int = Field(default=0, ge=0)
    rows_invalid: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def success_requires_spectrum(self) -> SpectrumReadResult:
        if self.status == "success" and self.spectrum is None:
            raise ValueError("successful spectrum read requires SpectrumData")
        return self


def is_supported_spectrum_path(path: Path) -> bool:
    """Return whether the file extension is supported as spectrum input."""
    return path.suffix.lower() in SUPPORTED_SPECTRUM_EXTENSIONS


def file_sha256(path: Path) -> str:
    """Compute SHA-256 for an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_spectrum_file(path: Path, *, sample_id: str | None = None) -> SpectrumReadResult:
    """Read `.txt`, `.csv` or `.xy` spectrum files into validated data."""
    source_filename = path.name
    if not is_supported_spectrum_path(path):
        diagnostic = _diagnostic(
            source_filename=source_filename,
            status="ignored",
            error_code=ErrorCode.UNSUPPORTED_EXTENSION,
            message="Unsupported spectrum extension",
            ignored_reason=f"unsupported extension: {path.suffix}",
        )
        return SpectrumReadResult(status="ignored", diagnostics=[diagnostic])

    sha256 = file_sha256(path)
    raw_text = path.read_text(encoding="utf-8-sig")
    if not raw_text.strip():
        diagnostic = _diagnostic(
            source_filename=source_filename,
            status="failed",
            error_code=ErrorCode.EMPTY_FILE,
            message="Spectrum file is empty",
        )
        return SpectrumReadResult(status="failed", diagnostics=[diagnostic])

    parsed_rows: list[tuple[float, float]] = []
    diagnostics: list[DiagnosticRecord] = []
    rows_read = 0
    rows_invalid = 0
    non_finite_seen = False

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        rows_read += 1
        row = _parse_numeric_row(stripped)
        if row is None:
            if _looks_like_header(stripped) and not parsed_rows:
                diagnostics.append(
                    _diagnostic(
                        source_filename=source_filename,
                        status="success",
                        error_code=None,
                        message=f"Skipped header row {line_number}",
                        rows_read=rows_read,
                        rows_invalid=rows_invalid,
                    )
                )
                continue
            rows_invalid += 1
            continue
        if not math.isfinite(row[0]) or not math.isfinite(row[1]):
            non_finite_seen = True
            rows_invalid += 1
            continue
        parsed_rows.append(row)

    if non_finite_seen:
        diagnostic = _diagnostic(
            source_filename=source_filename,
            status="failed",
            error_code=ErrorCode.NON_FINITE_VALUES,
            message="Spectrum contains NaN or Inf values",
            rows_read=rows_read,
            rows_invalid=rows_invalid,
        )
        return SpectrumReadResult(
            status="failed",
            diagnostics=[*diagnostics, diagnostic],
            rows_read=rows_read,
            rows_invalid=rows_invalid,
        )

    if not parsed_rows:
        diagnostic = _diagnostic(
            source_filename=source_filename,
            status="failed",
            error_code=ErrorCode.NO_VALID_NUMERIC_ROWS,
            message="No valid numeric rows found",
            rows_read=rows_read,
            rows_invalid=rows_invalid,
        )
        return SpectrumReadResult(
            status="failed",
            diagnostics=[*diagnostics, diagnostic],
            rows_read=rows_read,
            rows_invalid=rows_invalid,
        )

    normalized_rows, normalization_diagnostics = _normalize_rows(
        parsed_rows,
        source_filename=source_filename,
    )
    diagnostics.extend(normalization_diagnostics)
    if len(normalized_rows) < 2:
        diagnostic = _diagnostic(
            source_filename=source_filename,
            status="failed",
            error_code=ErrorCode.INSUFFICIENT_ANGLE_COVERAGE,
            message="At least two unique angles are required",
            rows_read=rows_read,
            rows_invalid=rows_invalid,
        )
        return SpectrumReadResult(
            status="failed",
            diagnostics=[*diagnostics, diagnostic],
            rows_read=rows_read,
            rows_invalid=rows_invalid,
        )

    angle_min = normalized_rows[0][0]
    angle_max = normalized_rows[-1][0]
    spectrum = SpectrumData(
        sample_id=sample_id or path.stem,
        source_filename=source_filename,
        two_theta=[row[0] for row in normalized_rows],
        intensity=[row[1] for row in normalized_rows],
        sha256=sha256,
    )
    diagnostics.append(
        _diagnostic(
            source_filename=source_filename,
            status="success",
            error_code=None,
            message="Spectrum file parsed successfully",
            rows_read=rows_read,
            rows_invalid=rows_invalid,
            angle_min=angle_min,
            angle_max=angle_max,
        )
    )
    return SpectrumReadResult(
        status="success",
        spectrum=spectrum,
        diagnostics=diagnostics,
        warnings=[
            diagnostic.message for diagnostic in diagnostics if diagnostic.error_code is None
        ],
        rows_read=rows_read,
        rows_invalid=rows_invalid,
    )


def _parse_numeric_row(line: str) -> tuple[float, float] | None:
    tokens = [token for token in TOKEN_SPLIT_RE.split(line.strip()) if token]
    if len(tokens) == 2:
        try:
            return float(tokens[0]), float(tokens[1])
        except ValueError:
            return None
    if len(tokens) == 3:
        try:
            return float(tokens[0]), float(tokens[1]) - float(tokens[2])
        except ValueError:
            return None
    return None


def _looks_like_header(line: str) -> bool:
    tokens = [token for token in TOKEN_SPLIT_RE.split(line.strip()) if token]
    if len(tokens) < 2:
        return False
    return any(any(character.isalpha() for character in token) for token in tokens)


def _normalize_rows(
    rows: list[tuple[float, float]],
    *,
    source_filename: str,
) -> tuple[list[tuple[float, float]], list[DiagnosticRecord]]:
    diagnostics: list[DiagnosticRecord] = []
    sorted_rows = sorted(rows, key=lambda row: row[0])
    if sorted_rows != rows:
        diagnostics.append(
            _diagnostic(
                source_filename=source_filename,
                status="success",
                error_code=None,
                message="Sorted rows by increasing two_theta",
            )
        )

    grouped: dict[float, list[float]] = defaultdict(list)
    for angle, intensity in sorted_rows:
        grouped[angle].append(intensity)

    duplicate_count = sum(len(values) - 1 for values in grouped.values())
    if duplicate_count:
        diagnostics.append(
            _diagnostic(
                source_filename=source_filename,
                status="success",
                error_code=None,
                message=f"Aggregated {duplicate_count} duplicate angle rows by mean intensity",
            )
        )

    normalized = [
        (angle, sum(values) / len(values))
        for angle, values in sorted(grouped.items(), key=lambda item: item[0])
    ]
    return normalized, diagnostics


def _diagnostic(
    *,
    source_filename: str,
    status: Literal["success", "failed", "ignored"],
    error_code: ErrorCode | None,
    message: str,
    rows_read: int | None = None,
    rows_invalid: int | None = None,
    angle_min: float | None = None,
    angle_max: float | None = None,
    ignored_reason: str | None = None,
) -> DiagnosticRecord:
    return DiagnosticRecord(
        source_filename=source_filename,
        status=status,
        stage="spectrum_io",
        error_code=error_code,
        message=message,
        rows_read=rows_read,
        rows_invalid=rows_invalid,
        angle_min=angle_min,
        angle_max=angle_max,
        ignored_reason=ignored_reason,
    )
