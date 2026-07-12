from __future__ import annotations

import numpy as np
import pytest

from cpicann_xrd.core.preprocessing import array_sha256, preprocess_spectrum
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.schemas import PreprocessingConfig, SpectrumData


def make_spectrum(
    *,
    two_theta: list[float] | None = None,
    intensity: list[float] | None = None,
) -> SpectrumData:
    return SpectrumData(
        sample_id="sample",
        source_filename="sample.txt",
        two_theta=two_theta or [10.0, 45.0, 80.0],
        intensity=intensity or [1.0, 10.0, 2.0],
    )


def test_preprocess_outputs_fixed_length_float32_model_input() -> None:
    result = preprocess_spectrum(make_spectrum())

    assert result.two_theta.shape == (4500,)
    assert result.intensity.shape == (4500,)
    assert result.model_input.shape == (1, 1, 4500)
    assert result.model_input.dtype == np.float32
    assert result.intensity.max() == pytest.approx(100.0)
    assert result.array_sha256 == array_sha256(result.model_input)


def test_preprocess_adds_legacy_boundaries_and_warnings() -> None:
    spectrum = make_spectrum(two_theta=[12.0, 40.0, 78.0], intensity=[1.0, 5.0, 2.0])

    result = preprocess_spectrum(spectrum)

    assert result.two_theta[0] == pytest.approx(10.0)
    assert result.two_theta[-1] == pytest.approx(80.0)
    assert any("Prepended 10.0" in warning for warning in result.warnings)
    assert any("Appended 80.0" in warning for warning in result.warnings)


def test_preprocess_supports_custom_point_count_for_tests() -> None:
    config = PreprocessingConfig(points=5)

    result = preprocess_spectrum(make_spectrum(), config)

    assert result.two_theta.tolist() == pytest.approx([10.0, 27.5, 45.0, 62.5, 80.0])
    assert result.model_input.shape == (1, 1, 5)


def test_preprocess_rejects_single_point_spectrum() -> None:
    spectrum = make_spectrum(two_theta=[10.0], intensity=[1.0])

    with pytest.raises(CpicannXrdError) as exc_info:
        preprocess_spectrum(spectrum)

    assert exc_info.value.error_code == ErrorCode.INSUFFICIENT_ANGLE_COVERAGE


def test_preprocess_rejects_non_positive_max_intensity() -> None:
    spectrum = make_spectrum(intensity=[0.0, 0.0, 0.0])

    with pytest.raises(CpicannXrdError) as exc_info:
        preprocess_spectrum(spectrum)

    assert exc_info.value.error_code == ErrorCode.PREPROCESSING_FAILED


def test_preprocessed_export_table_is_stable() -> None:
    result = preprocess_spectrum(make_spectrum(), PreprocessingConfig(points=3))

    assert result.export_table() == [
        {"two_theta": 10.0, "intensity": 10.0},
        {"two_theta": 45.0, "intensity": 100.0},
        {"two_theta": 80.0, "intensity": 20.0},
    ]
