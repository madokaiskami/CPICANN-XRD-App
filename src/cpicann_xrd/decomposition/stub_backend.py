"""Deterministic stub XDecomposer backend."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import numpy.typing as npt

from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.preprocessing import preprocess_for_xdecomposer
from cpicann_xrd.decomposition.schemas import (
    DecomposedComponent,
    XDecomposerRequest,
    XDecomposerResult,
)

STUB_MODEL_ID = "stub-xdecomposer-v1"


class StubDecompositionBackend:
    """Deterministic backend for product and client tests."""

    model_id = STUB_MODEL_ID

    def decompose(self, request: XDecomposerRequest) -> XDecomposerResult:
        """Return deterministic two-component decomposition."""
        preprocessed = preprocess_for_xdecomposer(request.two_theta, request.intensity)
        signal = preprocessed.intensity.astype(np.float32, copy=False)
        slots = min(request.max_sources, 2)
        components_raw = _split_signal(signal, slots)
        if components_raw.shape != (slots, signal.shape[0]):
            raise DecompositionError(
                DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
                "StubDecompositionBackend 输出 shape 异常",
                details={"shape": list(components_raw.shape)},
            )

        reconstruction = components_raw.sum(axis=0)
        residual = signal - reconstruction
        total_mass = float(np.sum(components_raw))
        components = []
        for index, pattern in enumerate(components_raw):
            active_probability = 0.95 if index == 0 else 0.65
            mass = float(np.sum(pattern))
            estimated_weight = 0.0 if total_mass <= 0 else mass / total_mass
            include_pattern = pattern.tolist() if request.return_component_patterns else None
            components.append(
                DecomposedComponent(
                    component_index=index + 1,
                    original_slot_index=index,
                    active_probability=active_probability,
                    is_active=active_probability >= request.activity_threshold,
                    estimated_weight=estimated_weight,
                    pattern_sha256=_sha256_float32(pattern),
                    pattern=include_pattern,
                )
            )

        return XDecomposerResult(
            sample_id=request.sample_id,
            model_id=self.model_id,
            components=components,
            reconstruction_error=_rmse(signal, reconstruction),
            residual_sha256=_sha256_float32(residual),
            reconstruction_sha256=_sha256_float32(reconstruction),
            preprocessing=preprocessed.metadata,
            warnings=["StubDecompositionBackend result; not a real XDecomposer prediction"],
        )


def _split_signal(signal: npt.NDArray[np.float32], slots: int) -> npt.NDArray[np.float32]:
    if slots <= 0:
        return np.zeros((0, signal.shape[0]), dtype=np.float32)
    if slots == 1:
        return signal.reshape(1, -1).astype(np.float32)
    axis = np.linspace(0.0, 1.0, signal.shape[0], dtype=np.float32)
    left_weight = 1.0 - axis
    right_weight = axis
    first = signal * left_weight
    second = signal * right_weight
    return np.stack([first, second], axis=0).astype(np.float32)


def _rmse(expected: npt.NDArray[np.float32], actual: npt.NDArray[np.float32]) -> float:
    return float(math.sqrt(float(np.mean((expected - actual) ** 2))))


def _sha256_float32(array: npt.NDArray[np.float32]) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()
