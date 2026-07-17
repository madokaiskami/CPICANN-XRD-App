from __future__ import annotations

import pytest
from pydantic import ValidationError

from cpicann_xrd.decomposition.schemas import (
    DecomposedComponent,
    XDecomposerPreprocessingMetadata,
    XDecomposerRequest,
    XDecomposerResult,
)

SHA = "0" * 64


def test_request_validates_lengths_and_parameters() -> None:
    request = XDecomposerRequest(
        sample_id="sample-001",
        two_theta=[10.0, 20.0],
        intensity=[1.0, 2.0],
        max_sources=4,
        activity_threshold=0.5,
    )

    assert request.sample_id == "sample-001"
    assert request.reference_top_k == 5


def test_request_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        XDecomposerRequest(sample_id="bad", two_theta=[10.0], intensity=[1.0, 2.0])


def test_result_schema_round_trips() -> None:
    metadata = XDecomposerPreprocessingMetadata(
        input_points=2,
        finite_points=2,
        unique_points=2,
        clipped_points=2,
        intensity_max_before_normalization=1.0,
        output_sha256=SHA,
    )
    result = XDecomposerResult(
        sample_id="sample-001",
        model_id="stub-xdecomposer-v1",
        components=[
            DecomposedComponent(
                component_index=1,
                original_slot_index=0,
                active_probability=0.9,
                is_active=True,
                estimated_weight=1.0,
                pattern_sha256=SHA,
            )
        ],
        reconstruction_error=0.0,
        residual_sha256=SHA,
        reconstruction_sha256=SHA,
        preprocessing=metadata,
    )

    parsed = XDecomposerResult.model_validate_json(result.model_dump_json())

    assert parsed.backend == "xdecomposer"
    assert parsed.preprocessing_version == "xdecomposer-v1"
    assert parsed.components[0].is_active is True
