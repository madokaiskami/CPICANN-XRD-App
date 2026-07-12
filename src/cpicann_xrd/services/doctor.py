"""Runtime health checks for configured backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from cpicann_xrd.exceptions import CpicannXrdError
from cpicann_xrd.model.cpicann_backend import CPICANNBackend
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.model.loader import load_manifest
from cpicann_xrd.schemas import StrictBaseModel
from cpicann_xrd.settings import AppSettings


class DoctorResult(StrictBaseModel):
    """Doctor command result."""

    status: Literal["ok", "failed"]
    backend: Literal["fake", "cpicann"]
    model_id: str
    message: str
    details: dict[str, str | int | list[int]] = {}


def run_doctor(settings: AppSettings, *, manifest_path: Path | None = None) -> DoctorResult:
    """Run a lightweight backend check."""
    if settings.backend == "fake":
        fake_backend = FakeBackend(num_classes=8, model_id="fake-cpicann")
        tensor = torch.zeros((1, 1, 4500), dtype=torch.float32)
        logits = fake_backend.predict_logits(tensor)
        return DoctorResult(
            status="ok",
            backend="fake",
            model_id=fake_backend.model_info.model_id,
            message="FakeBackend 可用；该结果不代表真实 CPICANN 模型。",
            details={"logits_shape": list(logits.shape)},
        )

    manifest = load_manifest(manifest_path or Path("configs/models/cpicann-single-d1.yaml"))
    try:
        cpicann_backend = CPICANNBackend(
            manifest=manifest,
            model_dir=settings.model_dir,
            device=settings.device,
        )
        tensor = torch.zeros((1, 1, manifest.input_points), dtype=torch.float32)
        logits = cpicann_backend.predict_logits(tensor)
    except CpicannXrdError as exc:
        return DoctorResult(
            status="failed",
            backend="cpicann",
            model_id=settings.model_id,
            message=exc.message,
            details={"error_code": exc.error_code.value},
        )

    return DoctorResult(
        status="ok",
        backend="cpicann",
        model_id=cpicann_backend.model_info.model_id,
        message="CPICANNBackend 可用。",
        details={"logits_shape": list(logits.shape)},
    )
