"""Health and readiness composition."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from xdecomposer_service.assets import inspect_assets
from xdecomposer_service.schemas import EnvironmentInfo, HealthResponse, InfoResponse, ReadyResponse
from xdecomposer_service.settings import Settings
from xdecomposer_service.upstream_adapter import inspect_upstream


def build_health() -> HealthResponse:
    """Return process liveness."""
    return HealthResponse()


def build_ready(settings: Settings) -> ReadyResponse:
    """Return readiness based on environment, asset and upstream checks."""
    environment = inspect_environment(settings.expected_python_minor)
    assets = inspect_assets(settings.manifest_path)
    upstream = inspect_upstream(settings.upstream_source_dir)
    warnings = []
    if not environment.python_matches_expected:
        warnings.append("python_version_mismatch")
    if not assets.assets_present:
        warnings.append("assets_not_ready")
    if settings.require_upstream_import and not upstream.importable:
        warnings.append("upstream_not_importable")
    ready = (
        assets.assets_present
        and environment.python_matches_expected
        and (upstream.importable or not settings.require_upstream_import)
    )
    return ReadyResponse(
        status="ready" if ready else "not_ready",
        environment=environment,
        assets=assets,
        upstream=upstream,
        warnings=warnings,
    )


def build_info(settings: Settings) -> InfoResponse:
    """Return runtime information."""
    return InfoResponse(
        environment=inspect_environment(settings.expected_python_minor),
        assets=inspect_assets(settings.manifest_path),
        upstream=inspect_upstream(settings.upstream_source_dir),
    )


def inspect_environment(expected_python_minor: str) -> EnvironmentInfo:
    """Inspect Python, optional PyTorch and accelerator state."""
    python_version = platform.python_version()
    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    torch_version = None
    torch_available = False
    cuda_available = False
    gpu_count = 0
    gpu_names: list[str] = []
    try:
        import torch  # type: ignore[import-not-found]

        torch_available = True
        torch_version = getattr(torch, "__version__", metadata.version("torch"))
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
        gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(gpu_count)]
    except Exception:
        torch_available = False

    return EnvironmentInfo(
        python=python_version,
        expected_python=expected_python_minor,
        python_matches_expected=python_minor == expected_python_minor,
        torch=torch_version,
        torch_available=torch_available,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
    )
