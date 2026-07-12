from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from cpicann_xrd.core.preprocessing import preprocess_spectrum
from cpicann_xrd.core.spectrum_io import read_spectrum_file
from cpicann_xrd.model.cpicann_backend import CPICANNBackend
from cpicann_xrd.model.loader import load_manifest

pytestmark = pytest.mark.model


def test_real_model_logits_shape_and_repeatability() -> None:
    model_dir = os.environ.get("CPICANN_MODEL_DIR")
    if not model_dir:
        pytest.skip("CPICANN_MODEL_DIR is not configured")
    manifest = load_manifest(Path("configs/models/cpicann-single-d1.yaml"))
    backend = CPICANNBackend(manifest=manifest, model_dir=Path(model_dir), device="cpu")
    tensor = torch.zeros((1, 1, manifest.input_points), dtype=torch.float32)

    first = backend.predict_logits(tensor)
    second = backend.predict_logits(tensor)

    assert first.shape == (1, manifest.num_classes)
    assert torch.equal(first, second)


def test_real_model_sample_smoke_uses_local_samples() -> None:
    model_dir = os.environ.get("CPICANN_MODEL_DIR")
    if not model_dir:
        pytest.skip("CPICANN_MODEL_DIR is not configured")

    sample_dir = Path("samples/CPICANN识别")
    expected = {
        "0-norm.txt": (
            "1be59cd3854ec938fa51cec4a76bee01fe0a9240c75929697ee6d8973f098c4f",
            [21637, 6002, 9994, 9477, 6100],
        ),
        "1-norm.txt": (
            "2e325067f109f5c747bbad611cfc2c3d668346d6ab9be43352b5bbaaba022125",
            [15237, 20319, 15147, 10815, 19897],
        ),
        "3-norm.txt": (
            "de58e6d8277bd5da70658ac9986a99f02859c91008ca3244763a895ccc269d86",
            [11272, 10803, 4143, 10245, 6100],
        ),
    }
    if not sample_dir.exists():
        pytest.skip(f"sample directory is not available: {sample_dir}")

    manifest = load_manifest(Path("configs/models/cpicann-single-d1.yaml"))
    backend = CPICANNBackend(manifest=manifest, model_dir=Path(model_dir), device="cpu")

    for filename, (input_sha256, expected_top5) in expected.items():
        result = read_spectrum_file(sample_dir / filename)
        assert result.status == "success"
        assert result.spectrum is not None
        assert result.rows_read == 3500

        preprocessed = preprocess_spectrum(result.spectrum)
        assert preprocessed.model_input.shape == (1, 1, manifest.input_points)
        assert preprocessed.array_sha256 == input_sha256

        logits = backend.predict_logits(torch.from_numpy(preprocessed.model_input))
        probabilities = torch.softmax(logits[0], dim=0)
        _, top_indices = torch.topk(probabilities, k=5)

        assert [int(index.item()) for index in top_indices] == expected_top5
