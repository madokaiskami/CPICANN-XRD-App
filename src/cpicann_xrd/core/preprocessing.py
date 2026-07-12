"""CPICANN-compatible spectrum preprocessing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.schemas import DiagnosticRecord, PreprocessingConfig, SpectrumData


@dataclass(frozen=True)
class PreprocessedSpectrum:
    """Fixed-length model input and exportable preprocessed data."""

    sample_id: str
    source_filename: str
    config: PreprocessingConfig
    two_theta: npt.NDArray[np.float32]
    intensity: npt.NDArray[np.float32]
    model_input: npt.NDArray[np.float32]
    diagnostics: tuple[DiagnosticRecord, ...]
    warnings: tuple[str, ...]

    @property
    def array_sha256(self) -> str:
        """SHA-256 of the normalized float32 model input bytes."""
        return array_sha256(self.model_input)

    def export_table(self) -> list[dict[str, float]]:
        """Return a CSV-friendly two-column representation."""
        return [
            {"two_theta": float(angle), "intensity": float(intensity)}
            for angle, intensity in zip(self.two_theta, self.intensity, strict=True)
        ]


def preprocess_spectrum(
    spectrum: SpectrumData,
    config: PreprocessingConfig | None = None,
) -> PreprocessedSpectrum:
    """Preprocess one parsed spectrum using `legacy-cpicann-v1` semantics."""
    active_config = config or PreprocessingConfig()
    if active_config.name != "legacy-cpicann-v1":
        raise ValueError(f"unsupported preprocessing protocol: {active_config.name}")

    angles = np.asarray(spectrum.two_theta, dtype=np.float32)
    intensities = np.asarray(spectrum.intensity, dtype=np.float32)
    if angles.size < 2:
        raise CpicannXrdError(
            ErrorCode.INSUFFICIENT_ANGLE_COVERAGE,
            "至少需要两个有效角度点才能插值",
            details={"source_filename": spectrum.source_filename},
        )
    if not np.isfinite(angles).all() or not np.isfinite(intensities).all():
        raise CpicannXrdError(
            ErrorCode.NON_FINITE_VALUES,
            "谱图包含 NaN 或 Inf",
            details={"source_filename": spectrum.source_filename},
        )

    diagnostics: list[DiagnosticRecord] = []
    warnings: list[str] = []
    angles, intensities = _ensure_legacy_boundaries(
        angles,
        intensities,
        config=active_config,
        source_filename=spectrum.source_filename,
        diagnostics=diagnostics,
        warnings=warnings,
    )

    if np.any(np.diff(angles) <= 0):
        raise CpicannXrdError(
            ErrorCode.INVALID_ANGLE_RANGE,
            "预处理需要严格递增的角度",
            details={"source_filename": spectrum.source_filename},
        )

    target_angles = np.linspace(
        active_config.two_theta_min,
        active_config.two_theta_max,
        active_config.points,
        dtype=np.float32,
    )
    interpolated = np.interp(target_angles, angles, intensities).astype(np.float32)
    normalized = _normalize_intensity(
        interpolated,
        config=active_config,
        source_filename=spectrum.source_filename,
    )
    model_input = normalized.reshape(1, 1, active_config.points).astype(np.float32)
    diagnostics.append(
        DiagnosticRecord(
            source_filename=spectrum.source_filename,
            status="success",
            stage="preprocessing",
            error_code=None,
            message="Preprocessed spectrum to fixed-length model input",
            rows_read=len(spectrum.two_theta),
            rows_invalid=0,
            angle_min=float(target_angles[0]),
            angle_max=float(target_angles[-1]),
        )
    )
    return PreprocessedSpectrum(
        sample_id=spectrum.sample_id,
        source_filename=spectrum.source_filename,
        config=active_config,
        two_theta=target_angles,
        intensity=normalized,
        model_input=model_input,
        diagnostics=tuple(diagnostics),
        warnings=tuple(warnings),
    )


def array_sha256(array: npt.NDArray[np.float32]) -> str:
    """Compute a stable hash for a float32 NumPy array."""
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _ensure_legacy_boundaries(
    angles: npt.NDArray[np.float32],
    intensities: npt.NDArray[np.float32],
    *,
    config: PreprocessingConfig,
    source_filename: str,
    diagnostics: list[DiagnosticRecord],
    warnings: list[str],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    output_angles = angles
    output_intensities = intensities

    if float(output_angles[0]) > config.two_theta_min:
        output_angles = np.insert(output_angles, 0, np.float32(config.two_theta_min))
        output_intensities = np.insert(output_intensities, 0, output_intensities[0])
        message = f"Prepended {config.two_theta_min} degree boundary using first intensity"
        warnings.append(message)
        diagnostics.append(_preprocessing_diagnostic(source_filename, message))

    if float(output_angles[-1]) < config.two_theta_max:
        output_angles = np.append(output_angles, np.float32(config.two_theta_max))
        output_intensities = np.append(output_intensities, output_intensities[-1])
        message = f"Appended {config.two_theta_max} degree boundary using last intensity"
        warnings.append(message)
        diagnostics.append(_preprocessing_diagnostic(source_filename, message))

    return output_angles.astype(np.float32), output_intensities.astype(np.float32)


def _normalize_intensity(
    intensity: npt.NDArray[np.float32],
    *,
    config: PreprocessingConfig,
    source_filename: str,
) -> npt.NDArray[np.float32]:
    max_intensity = float(np.max(intensity))
    if not np.isfinite(intensity).all() or not np.isfinite(max_intensity):
        raise CpicannXrdError(
            ErrorCode.NON_FINITE_VALUES,
            "预处理结果包含 NaN 或 Inf",
            details={"source_filename": source_filename},
        )
    if max_intensity <= 0:
        raise CpicannXrdError(
            ErrorCode.PREPROCESSING_FAILED,
            "谱图最大强度必须为正数才能归一化",
            details={"source_filename": source_filename, "max_intensity": max_intensity},
        )
    return (intensity / max_intensity * config.normalize_max_to).astype(np.float32)


def _preprocessing_diagnostic(
    source_filename: str,
    message: str,
) -> DiagnosticRecord:
    return DiagnosticRecord(
        source_filename=source_filename,
        status="success",
        stage="preprocessing",
        error_code=None,
        message=message,
    )
