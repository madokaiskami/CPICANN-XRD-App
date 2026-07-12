from __future__ import annotations

import pytest
import torch

from cpicann_xrd.catalog.catalog import load_catalog
from cpicann_xrd.catalog.confidence import rank_predictions_from_logits
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.schemas import FilterSpec


def test_filtered_confidence_uses_masked_logits_not_probability_resoftmax() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    logits = torch.tensor([0.0, 4.0, 3.0, -2.0, 1.0])
    predictions, candidate_count, warnings = rank_predictions_from_logits(
        logits,
        catalog=catalog,
        filter_spec=FilterSpec(allowed_elements={"Li", "Zr", "O"}),
        top_k=3,
    )

    valid_logits = torch.tensor([0.0, 4.0, 1.0])
    expected_filtered = torch.softmax(valid_logits, dim=0)
    unfiltered = torch.softmax(logits, dim=0)

    assert candidate_count == 3
    assert warnings == []
    assert [item.class_index for item in predictions] == [1, 4, 0]
    assert sum(item.filtered_confidence for item in predictions) == pytest.approx(1.0)
    assert predictions[0].filtered_confidence == pytest.approx(float(expected_filtered[1]))
    assert predictions[0].unfiltered_probability == pytest.approx(float(unfiltered[1]))
    assert predictions[0].global_rank == 1
    assert predictions[1].global_rank == 3


def test_confidence_reports_empty_filter_result() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    with pytest.raises(CpicannXrdError) as exc_info:
        rank_predictions_from_logits(
            torch.zeros(5),
            catalog=catalog,
            filter_spec=FilterSpec(include_must={"Pb"}),
            top_k=5,
        )

    assert exc_info.value.error_code == ErrorCode.NO_CANDIDATES_AFTER_FILTER


def test_top_k_greater_than_candidates_returns_all_with_warning() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    predictions, candidate_count, warnings = rank_predictions_from_logits(
        torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0]),
        catalog=catalog,
        filter_spec=FilterSpec(include_must={"Li", "Zr", "O"}, allowed_elements={"Li", "Zr", "O"}),
        top_k=5,
    )

    assert candidate_count == 1
    assert len(predictions) == 1
    assert predictions[0].class_index == 0
    assert predictions[0].filtered_confidence == pytest.approx(1.0)
    assert warnings == ["候选数少于 Top-K，已返回全部过滤后候选。"]


def test_masked_softmax_is_numerically_stable() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    predictions, _, _ = rank_predictions_from_logits(
        torch.tensor([10000.0, 9999.0, -10000.0, 9998.0, -9999.0]),
        catalog=catalog,
        filter_spec=FilterSpec(include_must={"Zr", "O"}),
        top_k=4,
    )

    assert sum(item.filtered_confidence for item in predictions) == pytest.approx(1.0)
    assert all(item.filtered_confidence >= 0 for item in predictions)
