from __future__ import annotations

import numpy as np
import pytest

from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.preprocessing import preprocess_for_xdecomposer


def test_preprocessing_outputs_fixed_shape_and_hash() -> None:
    first = preprocess_for_xdecomposer([10.0, 45.0, 80.0], [0.0, 10.0, 0.0])
    second = preprocess_for_xdecomposer([10.0, 45.0, 80.0], [0.0, 10.0, 0.0])

    assert first.tensor.shape == (1, 1, 3500)
    assert first.tensor.dtype == np.float32
    assert float(first.tensor.max()) == pytest.approx(1.0)
    assert first.metadata.output_sha256 == second.metadata.output_sha256
    assert first.metadata.tensor_shape == (1, 1, 3500)


def test_preprocessing_sorts_and_averages_duplicate_angles() -> None:
    result = preprocess_for_xdecomposer(
        [80.0, 10.0, 45.0, 45.0],
        [0.0, 0.0, 4.0, 8.0],
    )

    assert result.metadata.unique_points == 3
    assert "two_theta_sorted" in result.metadata.corrections
    assert "duplicate_two_theta_averaged" in result.metadata.corrections
    assert result.intensity.shape == (3500,)


def test_preprocessing_rejects_all_zero_input() -> None:
    with pytest.raises(DecompositionError) as exc_info:
        preprocess_for_xdecomposer([10.0, 80.0], [0.0, 0.0])

    assert exc_info.value.code == DecompositionErrorCode.PREPROCESSING_FAILED


def test_preprocessing_rejects_empty_and_non_finite_input() -> None:
    with pytest.raises(DecompositionError) as empty_exc:
        preprocess_for_xdecomposer([], [])
    assert empty_exc.value.code == DecompositionErrorCode.INVALID_INPUT

    with pytest.raises(DecompositionError) as range_exc:
        preprocess_for_xdecomposer([float("nan"), 5.0], [1.0, 2.0])
    assert range_exc.value.code == DecompositionErrorCode.PREPROCESSING_FAILED


def test_preprocessing_clips_to_physical_range() -> None:
    result = preprocess_for_xdecomposer(
        [5.0, 10.0, 45.0, 80.0, 90.0],
        [100.0, 0.0, 10.0, 0.0, 100.0],
    )

    assert result.metadata.clipped_points == 3
    assert "two_theta_clipped_to_10_80" in result.metadata.corrections
