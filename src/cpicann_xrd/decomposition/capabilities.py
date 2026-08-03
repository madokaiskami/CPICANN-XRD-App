"""Capability reporting for optional XDecomposer features."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import Field

from cpicann_xrd.decomposition.assets import verify_xdecomposer_assets
from cpicann_xrd.exceptions import CpicannXrdError
from cpicann_xrd.schemas import StrictBaseModel

XDECOMPOSER_BACKEND_ENV = "CPICANN_XDECOMPOSER_BACKEND"
XDECOMPOSER_MANIFEST_ENV = "CPICANN_XDECOMPOSER_MANIFEST"
XDECOMPOSER_SERVICE_URL_ENV = "CPICANN_XDECOMPOSER_SERVICE_URL"
XDECOMPOSER_READY_TIMEOUT_ENV = "CPICANN_XDECOMPOSER_READY_TIMEOUT_SECONDS"
XDECOMPOSER_READY_RETRIES_ENV = "CPICANN_XDECOMPOSER_READY_RETRIES"


class BackendCapability(StrictBaseModel):
    """Availability of one runtime capability."""

    enabled: bool
    available: bool
    reason: str
    details: dict[str, str] = Field(default_factory=dict)


class Capabilities(StrictBaseModel):
    """Public capability response shared by CLI, API and Web helpers."""

    cpicann: BackendCapability
    xdecomposer: BackendCapability


def build_capabilities(
    *,
    cpicann_available: bool,
    xdecomposer_backend: str | None = None,
    xdecomposer_manifest: Path | None = None,
) -> Capabilities:
    """Return stable feature capabilities without changing default behavior."""
    return Capabilities(
        cpicann=BackendCapability(
            enabled=True,
            available=cpicann_available,
            reason="available" if cpicann_available else "cpicann_unavailable",
        ),
        xdecomposer=evaluate_xdecomposer_capability(
            backend=xdecomposer_backend,
            manifest=xdecomposer_manifest,
        ),
    )


def evaluate_xdecomposer_capability(
    *,
    backend: str | None = None,
    manifest: Path | None = None,
) -> BackendCapability:
    """Evaluate optional XDecomposer capability from explicit inputs or env."""
    backend_name = _normalize_backend(
        backend if backend is not None else os.environ.get(XDECOMPOSER_BACKEND_ENV, "disabled")
    )
    manifest_path = manifest or _manifest_from_env()
    if backend_name == "disabled":
        return BackendCapability(
            enabled=False,
            available=False,
            reason="xdecomposer_disabled",
        )
    if backend_name == "stub":
        return BackendCapability(
            enabled=True,
            available=True,
            reason="stub_ready",
            details={"backend": "stub"},
        )
    if backend_name == "assets":
        if manifest_path is None:
            return BackendCapability(
                enabled=True,
                available=False,
                reason="assets_not_configured",
            )
        try:
            result = verify_xdecomposer_assets(manifest_path, production=True)
        except CpicannXrdError as exc:
            return BackendCapability(
                enabled=True,
                available=False,
                reason="assets_invalid",
                details={"code": exc.error_code.value, "message": exc.message},
            )
        return BackendCapability(
            enabled=True,
            available=True,
            reason="assets_ready",
            details={"model_id": result.model_id},
        )
    if backend_name == "remote":
        return _remote_capability()
    return BackendCapability(
        enabled=True,
        available=False,
        reason="unsupported_xdecomposer_backend",
        details={"backend": backend_name},
    )


def _normalize_backend(
    value: str | None,
) -> Literal["disabled", "stub", "assets", "remote", "unsupported"]:
    normalized = (value or "disabled").strip().lower()
    if normalized in {"", "disabled", "off", "false", "0"}:
        return "disabled"
    if normalized in {"stub", "fake"}:
        return "stub"
    if normalized in {"assets", "real", "local"}:
        return "assets"
    if normalized in {"remote", "service"}:
        return "remote"
    return "unsupported"


def _manifest_from_env() -> Path | None:
    value = os.environ.get(XDECOMPOSER_MANIFEST_ENV)
    if not value:
        return None
    return Path(value)


def _remote_capability() -> BackendCapability:
    base_url = os.environ.get(XDECOMPOSER_SERVICE_URL_ENV, "http://127.0.0.1:8100").rstrip("/")
    timeout = _env_float(XDECOMPOSER_READY_TIMEOUT_ENV, default=10.0)
    retries = _env_int(XDECOMPOSER_READY_RETRIES_ENV, default=2)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(f"{base_url}/readyz", timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (OSError, TimeoutError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.25)
    else:
        return BackendCapability(
            enabled=True,
            available=False,
            reason="service_unavailable",
            details={
                "url": base_url,
                "timeout_seconds": str(timeout),
                "retries": str(retries),
                "reason": repr(last_error),
            },
        )
    if payload.get("status") == "ready":
        assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
        return BackendCapability(
            enabled=True,
            available=True,
            reason="service_ready",
            details={
                "url": base_url,
                "model_id": str(assets.get("model_id", "")),
            },
        )
    return BackendCapability(
        enabled=True,
        available=False,
        reason="service_not_ready",
        details={"url": base_url, "status": str(payload.get("status", ""))},
    )


def _env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default
