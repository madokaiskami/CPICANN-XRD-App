"""Model manifest and artifact loading helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
import yaml

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.hash import sha256_file
from cpicann_xrd.model.manifest import ModelManifest


def load_manifest(path: Path) -> ModelManifest:
    """Load a model manifest from YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model manifest must contain a mapping")
    return ModelManifest.model_validate(data)


def resolve_weight_path(manifest: ModelManifest, model_dir: Path) -> Path:
    """Resolve a manifest weight path under the configured model directory."""
    if manifest.weight_path is None:
        raise CpicannXrdError(
            ErrorCode.MODEL_NOT_INSTALLED,
            "模型 manifest 未配置权重文件",
            details={"model_id": manifest.model_id},
        )
    if manifest.weight_path.is_absolute():
        return manifest.weight_path
    return model_dir / manifest.model_id / manifest.weight_path


def verify_file_sha256(path: Path, expected_sha256: str | None, *, error_code: ErrorCode) -> str:
    """Verify a file hash when an expected value is configured."""
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise CpicannXrdError(
            error_code,
            "文件 SHA-256 校验失败",
            details={
                "path": str(path),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            },
        )
    return actual_sha256


def load_torch_checkpoint(path: Path, *, device: str) -> Any:
    """Load a trusted, already-hash-verified PyTorch checkpoint."""
    kwargs: dict[str, Any] = {"map_location": torch.device(device)}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)
