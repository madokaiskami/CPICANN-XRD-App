from __future__ import annotations

from cpicann_xrd.decomposition.schemas import XDecomposerRequest
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend


def test_stub_backend_is_deterministic_and_limited_by_max_sources() -> None:
    backend = StubDecompositionBackend()
    request = XDecomposerRequest(
        sample_id="mix-001",
        two_theta=[10.0, 20.0, 45.0, 80.0],
        intensity=[0.0, 2.0, 10.0, 0.0],
        max_sources=4,
    )

    first = backend.decompose(request)
    second = backend.decompose(request)

    assert first.model_dump() == second.model_dump()
    assert len(first.components) == 2
    assert first.reconstruction_error < 1e-6
    assert first.preprocessing.output_points == 3500
    assert first.components[0].estimated_weight + first.components[1].estimated_weight == 1.0


def test_stub_backend_respects_one_source_request() -> None:
    backend = StubDecompositionBackend()
    request = XDecomposerRequest(
        sample_id="mix-001",
        two_theta=[10.0, 45.0, 80.0],
        intensity=[0.0, 10.0, 0.0],
        max_sources=1,
        return_component_patterns=True,
    )

    result = backend.decompose(request)

    assert len(result.components) == 1
    assert result.components[0].pattern is not None
    assert len(result.components[0].pattern) == 3500


def test_activity_threshold_controls_active_flag() -> None:
    backend = StubDecompositionBackend()
    request = XDecomposerRequest(
        sample_id="mix-001",
        two_theta=[10.0, 45.0, 80.0],
        intensity=[0.0, 10.0, 0.0],
        activity_threshold=0.8,
    )

    result = backend.decompose(request)

    assert [component.is_active for component in result.components] == [True, False]
