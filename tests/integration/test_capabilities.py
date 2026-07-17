"""Capability gate tests for optional XDecomposer integration."""

from __future__ import annotations

from pathlib import Path

from cpicann_xrd.decomposition.capabilities import (
    build_capabilities,
    evaluate_xdecomposer_capability,
)


def test_xdecomposer_disabled_by_default() -> None:
    capabilities = build_capabilities(cpicann_available=True)

    assert capabilities.cpicann.available is True
    assert capabilities.xdecomposer.enabled is False
    assert capabilities.xdecomposer.available is False
    assert capabilities.xdecomposer.reason == "xdecomposer_disabled"


def test_stub_backend_reports_ready() -> None:
    capabilities = build_capabilities(
        cpicann_available=True,
        xdecomposer_backend="stub",
    )

    assert capabilities.xdecomposer.enabled is True
    assert capabilities.xdecomposer.available is True
    assert capabilities.xdecomposer.reason == "stub_ready"


def test_assets_backend_without_manifest_is_not_ready() -> None:
    capability = evaluate_xdecomposer_capability(backend="assets")

    assert capability.enabled is True
    assert capability.available is False
    assert capability.reason == "assets_not_configured"


def test_assets_backend_invalid_manifest_is_structured(tmp_path: Path) -> None:
    manifest = tmp_path / "missing.yaml"

    capability = evaluate_xdecomposer_capability(backend="assets", manifest=manifest)

    assert capability.enabled is True
    assert capability.available is False
    assert capability.reason == "assets_invalid"
    assert capability.details["code"] == "XDECOMPOSER_ASSET_MANIFEST_INVALID"


def test_remote_backend_reports_service_unavailable() -> None:
    capability = evaluate_xdecomposer_capability(backend="remote")

    assert capability.enabled is True
    assert capability.available is False
    assert capability.reason == "service_unavailable"
