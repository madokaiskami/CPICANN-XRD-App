from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.cpicann_backend import CPICANNBackend
from cpicann_xrd.model.cpicann_network import CPICANNSinglePhaseNet
from cpicann_xrd.model.loader import load_manifest, verify_file_sha256
from cpicann_xrd.model.manifest import ModelManifest


def test_manifest_loads_phase4_model_contract() -> None:
    manifest = load_manifest(Path("configs/models/cpicann-single-d1.yaml"))

    assert manifest.model_id == "cpicann-single-d1"
    assert manifest.num_classes == 23073
    assert manifest.input_points == 4500
    assert manifest.checkpoint_path == Path("CPICANNsingle_phase_D1.pth")
    assert (
        manifest.checkpoint_sha256
        == "d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98"
    )
    assert manifest.weight_path == Path("CPICANNsingle_phase_D1.state_dict.pth")
    assert (
        manifest.weight_sha256 == "0f3a452da5218df46eaa37f7d0dadb388e08cdeafa6e24e9d6e2e93512bb2e9a"
    )
    assert manifest.catalog_path == Path("data/catalog/cpicann_single_phase_d1_catalog.csv")
    assert (
        manifest.catalog_sha256
        == "393fd648778c2f788efed0d29141051d4214f89152e46ef262e7213bc164e51f"
    )


def test_missing_weight_raises_model_not_installed(tmp_path: Path) -> None:
    manifest = load_manifest(Path("configs/models/cpicann-single-d1.yaml"))

    with pytest.raises(CpicannXrdError) as exc_info:
        CPICANNBackend(manifest=manifest, model_dir=tmp_path, device="cpu")

    assert exc_info.value.error_code == ErrorCode.MODEL_NOT_INSTALLED
    assert "expected_path" in exc_info.value.details


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(CpicannXrdError) as exc_info:
        verify_file_sha256(path, "0" * 64, error_code=ErrorCode.MODEL_HASH_MISMATCH)

    assert exc_info.value.error_code == ErrorCode.MODEL_HASH_MISMATCH


def test_cpicann_network_output_shape_and_determinism() -> None:
    torch.manual_seed(0)
    model = CPICANNSinglePhaseNet(num_classes=5)
    model.eval()
    tensor = torch.zeros((1, 1, 4500), dtype=torch.float32)

    first = model(tensor)
    second = model(tensor)

    assert first.shape == (1, 5)
    assert torch.equal(first, second)


def test_backend_rejects_wrong_size_weight_before_pickle_load(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    weight_dir = model_dir / "tiny"
    weight_dir.mkdir(parents=True)
    (weight_dir / "tiny.pth").write_bytes(b"tiny")
    manifest = ModelManifest(
        model_id="tiny",
        backend="cpicann",
        num_classes=1,
        architecture="test",
        weight_path=Path("tiny.pth"),
        weight_sha256=None,
        weight_size_bytes=999,
    )

    with pytest.raises(CpicannXrdError) as exc_info:
        CPICANNBackend(manifest=manifest, model_dir=model_dir, device="cpu")

    assert exc_info.value.error_code == ErrorCode.MODEL_HASH_MISMATCH
