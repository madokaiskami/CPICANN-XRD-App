"""Deterministic xdecomposer-v1 preprocessing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.schemas import XDecomposerPreprocessingMetadata

XDECOMPOSER_TWO_THETA_MIN = 10.0
XDECOMPOSER_TWO_THETA_MAX = 80.0
XDECOMPOSER_OUTPUT_POINTS = 3500
XDECOMPOSER_PREPROCESSING_VERSION = "xdecomposer-v1"


@dataclass(frozen=True)
class PreprocessedDecompositionInput:
    """Preprocessed tensor-like array and metadata."""

    tensor: npt.NDArray[np.float32]
    two_theta: npt.NDArray[np.float32]
    intensity: npt.NDArray[np.float32]
    metadata: XDecomposerPreprocessingMetadata


def preprocess_for_xdecomposer(
    two_theta: list[float],
    intensity: list[float],
) -> PreprocessedDecompositionInput:
    """Normalize physical XRD data to `[1, 1, 3500]` float32."""
    if len(two_theta) == 0 or len(intensity) == 0:
        raise DecompositionError(
            DecompositionErrorCode.INVALID_INPUT,
            "XDecomposer 输入不能为空",
        )
    if len(two_theta) != len(intensity):
        raise DecompositionError(
            DecompositionErrorCode.INVALID_INPUT,
            "two_theta 和 intensity 长度必须一致",
            details={"two_theta": len(two_theta), "intensity": len(intensity)},
        )

    angles = np.asarray(two_theta, dtype=np.float64)
    values = np.asarray(intensity, dtype=np.float64)
    finite_mask = np.isfinite(angles) & np.isfinite(values)
    corrections: list[str] = []
    if not finite_mask.all():
        corrections.append("non_finite_rows_removed")
    angles = angles[finite_mask]
    values = values[finite_mask]
    if angles.size == 0:
        raise DecompositionError(
            DecompositionErrorCode.INVALID_INPUT,
            "XDecomposer 输入没有有限数值点",
        )

    order = np.argsort(angles, kind="mergesort")
    if not np.array_equal(order, np.arange(order.size)):
        corrections.append("two_theta_sorted")
    angles = angles[order]
    values = values[order]

    unique_angles, inverse = np.unique(angles, return_inverse=True)
    if unique_angles.size != angles.size:
        corrections.append("duplicate_two_theta_averaged")
        sums = np.zeros(unique_angles.shape, dtype=np.float64)
        counts = np.zeros(unique_angles.shape, dtype=np.float64)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1.0)
        values = sums / counts
        angles = unique_angles

    range_mask = (angles >= XDECOMPOSER_TWO_THETA_MIN) & (angles <= XDECOMPOSER_TWO_THETA_MAX)
    clipped_points = int(range_mask.sum())
    if clipped_points < angles.size:
        corrections.append("two_theta_clipped_to_10_80")
    if clipped_points < 2:
        raise DecompositionError(
            DecompositionErrorCode.PREPROCESSING_FAILED,
            "XDecomposer 需要 10-80 度范围内至少两个点",
            details={"points_in_range": clipped_points},
        )
    angles = angles[range_mask]
    values = values[range_mask]

    values = np.clip(values, a_min=0.0, a_max=None)
    if np.max(values) <= 0:
        raise DecompositionError(
            DecompositionErrorCode.PREPROCESSING_FAILED,
            "XDecomposer 输入强度全零",
        )

    target_axis = np.linspace(
        XDECOMPOSER_TWO_THETA_MIN,
        XDECOMPOSER_TWO_THETA_MAX,
        XDECOMPOSER_OUTPUT_POINTS,
        dtype=np.float64,
    )
    interpolated = np.interp(target_axis, angles, values, left=0.0, right=0.0)
    interpolated = np.clip(interpolated, a_min=0.0, a_max=None)
    max_before_normalization = float(np.max(interpolated))
    if max_before_normalization <= 0:
        raise DecompositionError(
            DecompositionErrorCode.PREPROCESSING_FAILED,
            "XDecomposer 插值后强度全零",
        )
    normalized = (interpolated / max_before_normalization).astype(np.float32)
    tensor = normalized.reshape(1, 1, XDECOMPOSER_OUTPUT_POINTS)
    output_sha = _sha256_array(tensor)
    metadata = XDecomposerPreprocessingMetadata(
        input_points=len(two_theta),
        finite_points=int(finite_mask.sum()),
        unique_points=int(np.unique(np.asarray(two_theta, dtype=np.float64)[finite_mask]).size),
        clipped_points=clipped_points,
        intensity_max_before_normalization=max_before_normalization,
        corrections=corrections,
        output_sha256=output_sha,
    )
    return PreprocessedDecompositionInput(
        tensor=tensor,
        two_theta=target_axis.astype(np.float32),
        intensity=normalized,
        metadata=metadata,
    )


def _sha256_array(array: npt.NDArray[np.float32]) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()
