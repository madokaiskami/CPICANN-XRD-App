"""Model registry primitives."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from cpicann_xrd.model.manifest import ModelManifest


class ModelRegistry:
    """In-memory registry of model manifests."""

    def __init__(self, manifests: Iterable[ModelManifest] | None = None) -> None:
        self._manifests: dict[str, ModelManifest] = {}
        for manifest in manifests or []:
            self.register(manifest)

    def register(self, manifest: ModelManifest) -> None:
        """Register or replace a model manifest by ID."""
        self._manifests[manifest.model_id] = manifest

    def get(self, model_id: str) -> ModelManifest:
        """Return a manifest by model ID."""
        try:
            return self._manifests[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model_id: {model_id}") from exc

    def list(self) -> list[ModelManifest]:
        """Return registered manifests sorted by model ID."""
        return [self._manifests[key] for key in sorted(self._manifests)]


def create_default_registry() -> ModelRegistry:
    """Create the Phase 2 registry with fake and planned CPICANN entries."""
    return ModelRegistry(
        [
            ModelManifest(
                model_id="fake-cpicann",
                backend="fake",
                num_classes=8,
                architecture="deterministic-fake",
            ),
            ModelManifest(
                model_id="cpicann-single-d1",
                backend="cpicann",
                num_classes=23073,
                architecture="CPICANN(embed_dim=128,nhead=8,layers=6)",
                source_url="https://huggingface.co/AI4Cryst/CPICANN",
                source_revision="3dbfaeab51d272e013d211c7f957760b46ab41cc",
                weight_path=Path("CPICANNsingle_phase_D1.pth"),
                weight_sha256="d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98",
            ),
        ]
    )
