"""Inference backend protocol."""

from __future__ import annotations

from typing import Protocol

import torch

from cpicann_xrd.schemas import ModelInfo


class InferenceBackend(Protocol):
    """Common protocol implemented by fake and real model backends."""

    @property
    def model_info(self) -> ModelInfo:
        """Return backend and model metadata."""
        ...

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for an input tensor."""
        ...
