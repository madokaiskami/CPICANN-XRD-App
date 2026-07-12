"""Real CPICANN inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.cpicann_network import CPICANNSinglePhaseNet
from cpicann_xrd.model.loader import load_torch_checkpoint, resolve_weight_path, verify_file_sha256
from cpicann_xrd.model.manifest import ModelManifest
from cpicann_xrd.schemas import ModelInfo


class CPICANNBackend:
    """Load a verified CPICANN checkpoint and return raw logits."""

    def __init__(
        self,
        *,
        manifest: ModelManifest,
        model_dir: Path,
        device: str = "cpu",
    ) -> None:
        self._manifest = manifest
        self._device = torch.device(device)
        self._weight_path = resolve_weight_path(manifest, model_dir)
        if not self._weight_path.exists():
            raise CpicannXrdError(
                ErrorCode.MODEL_NOT_INSTALLED,
                "未找到 CPICANN 权重文件",
                details={
                    "model_id": manifest.model_id,
                    "expected_path": str(self._weight_path),
                },
            )
        if (
            manifest.weight_size_bytes is not None
            and self._weight_path.stat().st_size != manifest.weight_size_bytes
        ):
            raise CpicannXrdError(
                ErrorCode.MODEL_HASH_MISMATCH,
                "权重文件大小与 manifest 不一致",
                details={
                    "model_id": manifest.model_id,
                    "expected_size_bytes": manifest.weight_size_bytes,
                    "actual_size_bytes": self._weight_path.stat().st_size,
                },
            )
        self._weight_sha256 = verify_file_sha256(
            self._weight_path,
            manifest.weight_sha256,
            error_code=ErrorCode.MODEL_HASH_MISMATCH,
        )
        self._model = self._load_model()
        self._model_info = ModelInfo(
            model_id=manifest.model_id,
            backend="cpicann",
            num_classes=manifest.num_classes,
            preprocessing_version=manifest.preprocessing_version,
            device=str(self._device),
            source_revision=manifest.source_revision,
            weight_sha256=self._weight_sha256,
            catalog_sha256=manifest.catalog_sha256,
            metadata={"backend": "cpicann"},
        )

    @property
    def model_info(self) -> ModelInfo:
        """Return model metadata."""
        return self._model_info

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape `(batch, num_classes)`."""
        if x.ndim != 3:
            raise ValueError("CPICANN input tensor must have shape (batch, channels, points)")
        if x.shape[-1] != self._manifest.input_points:
            raise ValueError(f"CPICANN input must have {self._manifest.input_points} points")
        with torch.no_grad():
            logits = self._model(x.to(device=self._device, dtype=torch.float32))
        if logits.ndim != 2 or logits.shape[1] != self._manifest.num_classes:
            raise CpicannXrdError(
                ErrorCode.MODEL_OUTPUT_SIZE_MISMATCH,
                "模型输出类别数与 manifest 不一致",
                details={
                    "expected": self._manifest.num_classes,
                    "actual_shape": list(logits.shape),
                },
            )
        return torch.Tensor.cpu(logits)

    def _load_model(self) -> CPICANNSinglePhaseNet:
        checkpoint = load_torch_checkpoint(self._weight_path, device=str(self._device))
        state_dict = _extract_state_dict(checkpoint)
        model = CPICANNSinglePhaseNet(num_classes=self._manifest.num_classes)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        return model


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if all(isinstance(key, str) for key in checkpoint):
            tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
            if tensor_values:
                return checkpoint
    raise CpicannXrdError(
        ErrorCode.MODEL_NOT_INSTALLED,
        "权重文件不是可识别的 CPICANN state_dict",
    )
