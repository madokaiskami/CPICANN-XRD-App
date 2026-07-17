"""Stub integration tests for XDecomposer -> CPICANN orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cpicann_xrd.catalog.catalog import load_catalog_from_manifest
from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.orchestration import (
    identify_decomposed_components,
    restore_component_spectrum,
)
from cpicann_xrd.decomposition.schemas import XDecomposerRequest
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.schemas import FilterSpec, ModelInfo

CATALOG_MANIFEST = Path("data/catalog/phase5_fixture_catalog_manifest.json")


def test_two_components_receive_top_k_predictions() -> None:
    catalog, manifest = load_catalog_from_manifest(CATALOG_MANIFEST)
    result = identify_decomposed_components(
        request=_mixture_request(),
        decomposition_backend=StubDecompositionBackend(),
        phase_backend=FakeBackend(num_classes=len(catalog), injected_logits=[5, 4, 3, 2, 1]),
        catalog=catalog,
        catalog_manifest=manifest,
        top_k=3,
    )

    assert result.status == "success"
    assert [component.status for component in result.components] == ["success", "success"]
    assert all(component.cpicann is not None for component in result.components)
    assert all(component.cpicann.returned_top_k == 3 for component in result.components)
    assert all(component.catalog is not None for component in result.components)
    assert "conditional confidence" in result.report_markdown


def test_element_filtering_renormalizes_filtered_confidence() -> None:
    catalog, manifest = load_catalog_from_manifest(CATALOG_MANIFEST)
    result = identify_decomposed_components(
        request=_mixture_request(),
        decomposition_backend=StubDecompositionBackend(),
        phase_backend=FakeBackend(num_classes=len(catalog), injected_logits=[1, 2, 3, 4, 5]),
        catalog=catalog,
        catalog_manifest=manifest,
        top_k=10,
        filter_spec=FilterSpec(allowed_elements={"Li", "O", "Zr"}),
    )

    for component in result.components:
        assert component.cpicann is not None
        assert component.cpicann.candidate_count_after_filter == 3
        assert component.cpicann.returned_top_k == 3
        assert all("Hf" not in item.phase.elements for item in component.cpicann.predictions)
        filtered_sum = sum(item.filtered_confidence for item in component.cpicann.predictions)
        assert filtered_sum == pytest.approx(1.0)


def test_component_candidate_empty_does_not_drop_other_components() -> None:
    catalog, manifest = load_catalog_from_manifest(CATALOG_MANIFEST)
    result = identify_decomposed_components(
        request=_mixture_request(),
        decomposition_backend=StubDecompositionBackend(),
        phase_backend=FakeBackend(num_classes=len(catalog), injected_logits=[1, 2, 3, 4, 5]),
        catalog=catalog,
        catalog_manifest=manifest,
        top_k=3,
        component_filter_specs={
            1: FilterSpec(include_must={"Li"}),
            2: FilterSpec(include_must={"Na"}),
        },
    )

    assert result.status == "partial"
    assert result.components[0].status == "success"
    assert result.components[1].status == "failed"
    assert result.components[1].error is not None
    assert result.components[1].error.code == ErrorCode.NO_CANDIDATES_AFTER_FILTER.value


def test_component_candidate_count_less_than_top_k_is_reported() -> None:
    catalog, manifest = load_catalog_from_manifest(CATALOG_MANIFEST)
    result = identify_decomposed_components(
        request=_mixture_request(),
        decomposition_backend=StubDecompositionBackend(),
        phase_backend=FakeBackend(num_classes=len(catalog), injected_logits=[1, 2, 3, 4, 5]),
        catalog=catalog,
        catalog_manifest=manifest,
        top_k=5,
        filter_spec=FilterSpec(include_must={"Hf"}),
    )

    for component in result.components:
        assert component.cpicann is not None
        assert component.cpicann.candidate_count_after_filter == 1
        assert component.cpicann.returned_top_k == 1
        assert component.cpicann.warnings == ["候选数少于 Top-K，已返回全部过滤后候选。"]


def test_xdecomposer_backend_unavailable_is_structured() -> None:
    with pytest.raises(DecompositionError) as exc_info:
        identify_decomposed_components(
            request=_mixture_request(),
            decomposition_backend=UnavailableDecompositionBackend(),
            phase_backend=FakeBackend(num_classes=5),
            top_k=3,
        )

    assert exc_info.value.code == DecompositionErrorCode.BACKEND_UNAVAILABLE


def test_cpicann_backend_unavailable_records_component_failures() -> None:
    result = identify_decomposed_components(
        request=_mixture_request(),
        decomposition_backend=StubDecompositionBackend(),
        phase_backend=UnavailablePhaseBackend(),
        top_k=3,
    )

    assert result.status == "failed"
    assert [component.status for component in result.components] == ["failed", "failed"]
    assert all(component.error is not None for component in result.components)
    assert {component.error.code for component in result.components if component.error} == {
        ErrorCode.MODEL_NOT_INSTALLED.value
    }


def test_single_component_failure_keeps_remaining_diagnostics() -> None:
    result = identify_decomposed_components(
        request=_mixture_request(),
        decomposition_backend=StubDecompositionBackend(),
        phase_backend=FailsFirstPhaseBackend(),
        top_k=2,
    )

    assert result.status == "partial"
    assert [component.status for component in result.components] == ["failed", "success"]
    assert result.components[0].error is not None
    assert result.components[1].cpicann is not None
    assert "RuntimeError" in result.components[0].error.code


def test_component_axis_restore_and_cpicann_hash_are_deterministic() -> None:
    request = _mixture_request()
    decomposition = StubDecompositionBackend().decompose(
        request.model_copy(update={"return_component_patterns": True})
    )
    restored = restore_component_spectrum(
        sample_id=request.sample_id,
        source_filename=request.source_filename or request.sample_id,
        component=decomposition.components[0],
        preprocessing=decomposition.preprocessing,
    )

    assert len(restored.two_theta) == 3500
    assert restored.two_theta[0] == pytest.approx(10.0)
    assert restored.two_theta[-1] == pytest.approx(80.0)

    kwargs = {
        "request": request,
        "decomposition_backend": StubDecompositionBackend(),
        "phase_backend": FakeBackend(num_classes=5),
        "top_k": 2,
    }
    first = identify_decomposed_components(**kwargs)
    second = identify_decomposed_components(**kwargs)

    assert [component.xdecomposer_axis.points for component in first.components] == [3500, 3500]
    assert [component.cpicann_input_sha256 for component in first.components] == [
        component.cpicann_input_sha256 for component in second.components
    ]
    assert all(
        component.cpicann_preprocessing_version == "legacy-cpicann-v1"
        for component in first.components
    )


def _mixture_request() -> XDecomposerRequest:
    two_theta = np.linspace(10.0, 80.0, 801, dtype=np.float32)
    phase_a = np.exp(-0.5 * ((two_theta - 28.0) / 1.8) ** 2)
    phase_b = np.exp(-0.5 * ((two_theta - 52.0) / 2.4) ** 2)
    mixture = 0.7 * phase_a + 0.3 * phase_b
    return XDecomposerRequest(
        sample_id="synthetic-mixture",
        source_filename="synthetic-mixture.xy",
        two_theta=[float(value) for value in two_theta],
        intensity=[float(value) for value in mixture],
        max_sources=2,
        activity_threshold=0.5,
    )


class UnavailableDecompositionBackend:
    """Backend that simulates an unavailable XDecomposer worker."""

    def decompose(self, request: XDecomposerRequest) -> object:
        raise DecompositionError(
            DecompositionErrorCode.BACKEND_UNAVAILABLE,
            "XDecomposer worker is unavailable",
            details={"sample_id": request.sample_id},
        )


class UnavailablePhaseBackend:
    """Backend that simulates an unavailable CPICANN model."""

    @property
    def model_info(self) -> ModelInfo:
        return _phase_model_info()

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        raise CpicannXrdError(ErrorCode.MODEL_NOT_INSTALLED, "CPICANN model is unavailable")


class FailsFirstPhaseBackend:
    """Backend that fails one component and succeeds for the next."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_info(self) -> ModelInfo:
        return _phase_model_info()

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("component-specific backend failure")
        return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)


def _phase_model_info() -> ModelInfo:
    return ModelInfo(
        model_id="test-cpicann",
        backend="fake",
        num_classes=5,
        preprocessing_version="legacy-cpicann-v1",
    )
