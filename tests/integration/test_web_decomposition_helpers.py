"""Web helper tests for optional decomposition modes."""

from __future__ import annotations

from cpicann_xrd.decomposition.capabilities import build_capabilities
from cpicann_xrd.decomposition.schemas import (
    DecomposedComponent,
    XDecomposerPreprocessingMetadata,
)
from cpicann_xrd.web.service import component_pattern_csv, web_mode_options


def test_web_modes_default_to_single_phase_only() -> None:
    capabilities = build_capabilities(cpicann_available=True)
    options = web_mode_options(capabilities)

    assert [option.key for option in options] == [
        "single_phase",
        "decompose",
        "decompose_and_identify",
    ]
    assert options[0].enabled is True
    assert options[1].enabled is False
    assert options[2].enabled is False
    assert options[1].reason == "xdecomposer_disabled"
    assert options[2].reason == "xdecomposer_disabled"


def test_web_modes_enable_decomposition_when_stub_ready() -> None:
    capabilities = build_capabilities(cpicann_available=True, xdecomposer_backend="stub")
    options = web_mode_options(capabilities)

    assert all(option.enabled for option in options)
    assert [option.label for option in options] == [
        "单相物相识别",
        "多相谱图分解",
        "多相分解并识别",
    ]


def test_component_pattern_csv_uses_xdecomposer_axis() -> None:
    component = DecomposedComponent(
        component_index=0,
        original_slot_index=2,
        active_probability=0.9,
        is_active=True,
        estimated_weight=0.5,
        pattern_sha256="0" * 64,
        pattern=[1.0, 2.0, 3.0],
    )
    preprocessing = XDecomposerPreprocessingMetadata(
        input_points=3,
        finite_points=3,
        unique_points=3,
        clipped_points=0,
        intensity_max_before_normalization=3.0,
        output_points=3,
        tensor_shape=(1, 1, 3),
        output_sha256="1" * 64,
    )

    csv_text = component_pattern_csv(component, preprocessing).decode("utf-8")

    assert csv_text.splitlines() == [
        "two_theta,intensity",
        "10,1",
        "45,2",
        "80,3",
    ]
