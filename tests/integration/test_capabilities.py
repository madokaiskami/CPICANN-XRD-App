"""Capability gate tests for optional XDecomposer integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

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


def test_remote_backend_reports_ready_from_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        assert url == "http://worker:8100/readyz"
        assert timeout == 2.0
        return _FakeResponse(
            {
                "status": "ready",
                "assets": {"model_id": "xdecomposer-local"},
            }
        )

    monkeypatch.setenv("CPICANN_XDECOMPOSER_SERVICE_URL", "http://worker:8100")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    capability = evaluate_xdecomposer_capability(backend="remote")

    assert capability.enabled is True
    assert capability.available is True
    assert capability.reason == "service_ready"
    assert capability.details["model_id"] == "xdecomposer-local"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")
