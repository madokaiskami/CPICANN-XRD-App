from __future__ import annotations

import torch
from pytest import approx, raises

from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.services.predictor import PredictionService


def test_fake_backend_is_deterministic_for_same_input() -> None:
    backend = FakeBackend(num_classes=6)
    tensor = torch.linspace(0, 1, 4500, dtype=torch.float32).reshape(1, 1, 4500)

    first = backend.predict_logits(tensor)
    second = backend.predict_logits(tensor)

    assert first.shape == (1, 6)
    assert torch.equal(first, second)
    assert backend.model_info.backend == "fake"
    assert backend.model_info.metadata["backend"] == "fake"


def test_fake_backend_supports_injected_logits() -> None:
    backend = FakeBackend(num_classes=4, injected_logits=[0.1, 0.2, 2.0, -1.0])
    tensor = torch.zeros((2, 1, 4500), dtype=torch.float32)

    logits = backend.predict_logits(tensor)

    assert logits.shape == (2, 4)
    assert torch.equal(logits[0], torch.tensor([0.1, 0.2, 2.0, -1.0]))
    assert torch.equal(logits[0], logits[1])


def test_fake_backend_rejects_bad_injected_logits() -> None:
    with raises(ValueError, match="length"):
        FakeBackend(num_classes=3, injected_logits=[1.0, 2.0])


def test_fake_backend_drives_minimal_prediction_service() -> None:
    backend = FakeBackend(num_classes=4, injected_logits=[0.0, 3.0, 1.0, -2.0])
    service = PredictionService(backend)
    tensor = torch.ones((1, 1, 4500), dtype=torch.float32)

    prediction = service.predict_tensor(
        sample_id="sample",
        source_filename="sample.xy",
        tensor=tensor,
        top_k=2,
    )

    assert prediction.backend == "fake"
    assert prediction.model_id == "fake-cpicann"
    assert prediction.returned_top_k == 2
    assert [item.class_index for item in prediction.predictions] == [1, 2]
    assert prediction.predictions[0].filtered_confidence == approx(
        prediction.predictions[0].unfiltered_probability
    )
    assert prediction.warnings == ["FakeBackend result; not a real CPICANN prediction"]
