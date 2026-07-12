"""Runtime backend and catalog construction shared by CLI and Web."""

from __future__ import annotations

from pathlib import Path

from cpicann_xrd.catalog.catalog import PhaseCatalog, load_catalog_from_manifest
from cpicann_xrd.model.cpicann_backend import CPICANNBackend
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.model.loader import load_manifest
from cpicann_xrd.model.protocol import InferenceBackend
from cpicann_xrd.settings import load_settings

DEFAULT_MODEL_MANIFEST = Path("configs/models/cpicann-single-d1.yaml")
DEFAULT_CATALOG_MANIFEST = Path("data/catalog/catalog_manifest.json")
FAKE_CATALOG_MANIFEST = Path("data/catalog/phase5_fixture_catalog_manifest.json")


def build_runtime(backend_name: str) -> tuple[InferenceBackend, PhaseCatalog]:
    """Build a backend and catalog for user-facing interfaces."""
    if backend_name == "fake":
        catalog, _ = load_catalog_from_manifest(FAKE_CATALOG_MANIFEST)
        backend: InferenceBackend = FakeBackend(
            num_classes=len(catalog),
            injected_logits=[0.0, 4.0, 3.0, -2.0, 1.0],
        )
        return backend, catalog
    if backend_name == "cpicann":
        settings = load_settings(cli_overrides={"backend": "cpicann"})
        manifest = load_manifest(DEFAULT_MODEL_MANIFEST)
        catalog, catalog_manifest = load_catalog_from_manifest(DEFAULT_CATALOG_MANIFEST)
        if manifest.catalog_sha256 != catalog_manifest.catalog_sha256:
            raise ValueError("模型 manifest 与 catalog manifest 的 SHA-256 不一致")
        backend = CPICANNBackend(
            manifest=manifest,
            model_dir=settings.model_dir,
            device=settings.device,
        )
        return backend, catalog
    raise ValueError("backend must be fake or cpicann")
