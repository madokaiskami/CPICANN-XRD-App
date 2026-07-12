"""Deterministic fake inference backend for tests and demos."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from cpicann_xrd.schemas import ModelInfo


class FakeBackend:
    """Deterministic backend that never represents a real CPICANN model."""

    def __init__(
        self,
        *,
        num_classes: int = 8,
        model_id: str = "fake-cpicann",
        injected_logits: Sequence[float] | torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self._num_classes = num_classes
        self._model_info = ModelInfo(
            model_id=model_id,
            backend="fake",
            num_classes=num_classes,
            device=device,
            source_revision=None,
            weight_sha256=None,
            catalog_sha256=None,
            metadata={
                "backend": "fake",
                "purpose": "tests-and-demos-only",
            },
        )
        self._injected_logits = self._prepare_injected_logits(injected_logits)

    @property
    def model_info(self) -> ModelInfo:
        """Return fake backend metadata."""
        return self._model_info

    def _prepare_injected_logits(
        self,
        injected_logits: Sequence[float] | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if injected_logits is None:
            return None
        logits = torch.as_tensor(injected_logits, dtype=torch.float32)
        if logits.ndim != 1:
            raise ValueError("injected_logits must be one-dimensional")
        if logits.numel() != self._num_classes:
            raise ValueError("injected_logits length must match num_classes")
        return logits

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic logits with shape ``(batch, num_classes)``."""
        if x.ndim < 2:
            raise ValueError("input tensor must include a batch dimension")
        batch_size = x.shape[0]
        if self._injected_logits is not None:
            return self._injected_logits.to(device=x.device).repeat(batch_size, 1)

        flattened = x.to(dtype=torch.float32).reshape(batch_size, -1)
        means = flattened.mean(dim=1, keepdim=True)
        sums = flattened.sum(dim=1, keepdim=True)
        class_index = torch.arange(self._num_classes, dtype=torch.float32, device=x.device).reshape(
            1, -1
        )
        logits = torch.sin(class_index * 0.173 + means) + torch.cos(
            class_index * 0.071 + sums * 0.001
        )
        logits = logits + class_index / max(self._num_classes, 1)
        return logits
