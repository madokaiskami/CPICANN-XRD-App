"""Web helper tests for optional decomposition modes."""

from __future__ import annotations

from cpicann_xrd.decomposition.capabilities import build_capabilities
from cpicann_xrd.web.service import web_mode_options


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
