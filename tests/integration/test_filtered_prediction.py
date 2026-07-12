from __future__ import annotations

import pytest
import torch

from cpicann_xrd.catalog.catalog import load_catalog
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.schemas import FilterSpec
from cpicann_xrd.services.predictor import PredictionService


def test_prediction_service_uses_catalog_filter_and_masked_confidence() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    backend = FakeBackend(num_classes=5, injected_logits=[0.0, 4.0, 3.0, -2.0, 1.0])
    service = PredictionService(backend, catalog=catalog)

    prediction = service.predict_tensor(
        sample_id="sample",
        source_filename="sample.xy",
        tensor=torch.zeros((1, 1, 4500), dtype=torch.float32),
        top_k=2,
        filter_spec=FilterSpec(allowed_elements={"Li", "Zr", "O"}),
    )

    assert prediction.candidate_count_before_filter == 5
    assert prediction.candidate_count_after_filter == 3
    assert prediction.returned_top_k == 2
    assert [item.class_index for item in prediction.predictions] == [1, 4]
    assert [item.filtered_rank for item in prediction.predictions] == [1, 2]
    assert [item.global_rank for item in prediction.predictions] == [1, 3]
    assert prediction.predictions[0].phase.cod_id == "FIXTURE-00001"
    assert prediction.predictions[0].phase.formula == "ZrO2"
    assert prediction.predictions[0].filtered_confidence == pytest.approx(
        float(torch.softmax(torch.tensor([0.0, 4.0, 1.0]), dim=0)[1])
    )
