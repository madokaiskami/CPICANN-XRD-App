from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cpicann_xrd.exceptions import ErrorCode
from cpicann_xrd.schemas import (
    DiagnosticRecord,
    FilterSpec,
    ModelInfo,
    PhaseRecord,
    PredictionItem,
    PreprocessingConfig,
    RunMetadata,
    SamplePrediction,
    SpectrumData,
)


def test_spectrum_data_validates_lengths_and_finite_values() -> None:
    spectrum = SpectrumData(
        sample_id="sample-1",
        source_filename="sample.xy",
        two_theta=[10.0, 10.1],
        intensity=[1.0, 2.0],
        sha256="a" * 64,
    )

    assert spectrum.sample_id == "sample-1"

    with pytest.raises(ValidationError):
        SpectrumData(
            sample_id="bad",
            source_filename="bad.xy",
            two_theta=[10.0],
            intensity=[1.0, 2.0],
        )

    with pytest.raises(ValidationError):
        SpectrumData(
            sample_id="bad",
            source_filename="bad.xy",
            two_theta=[10.0],
            intensity=[float("nan")],
        )


def test_filter_spec_normalizes_and_serializes_elements_deterministically() -> None:
    spec = FilterSpec(include_must="Zr O", allowed_elements={"Li", "O", "Zr"})

    assert spec.include_must == frozenset({"Zr", "O"})
    assert spec.allowed_elements == frozenset({"Li", "O", "Zr"})
    assert json.loads(spec.model_dump_json()) == {
        "include_must": ["O", "Zr"],
        "allowed_elements": ["Li", "O", "Zr"],
    }

    with pytest.raises(ValidationError):
        FilterSpec(include_must=["not-an-element"])


def test_phase_record_and_prediction_schema_round_trip_json() -> None:
    phase = PhaseRecord(
        class_index=3378,
        cod_id="9004484",
        formula="Pb4 S4 O16",
        reduced_formula="PbSO4",
        elements={"Pb", "S", "O"},
        space_group="Pnma",
        space_group_number=62,
    )
    item = PredictionItem(
        filtered_rank=1,
        global_rank=3,
        class_index=3378,
        phase=phase,
        raw_logit=12.5,
        unfiltered_probability=0.2,
        filtered_confidence=0.7,
    )
    sample = SamplePrediction(
        sample_id="pbso4",
        source_filename="PbSO4.csv",
        status="success",
        backend="fake",
        model_id="fake-cpicann",
        preprocessing_version="legacy-cpicann-v1",
        candidate_count_before_filter=8,
        candidate_count_after_filter=8,
        requested_top_k=1,
        returned_top_k=1,
        predictions=[item],
    )

    payload = json.loads(sample.model_dump_json())

    assert payload["predictions"][0]["phase"]["elements"] == ["O", "Pb", "S"]
    assert payload["returned_top_k"] == 1

    with pytest.raises(ValidationError):
        SamplePrediction(
            sample_id="bad",
            source_filename="bad.csv",
            status="success",
            backend="fake",
            model_id="fake-cpicann",
            preprocessing_version="legacy-cpicann-v1",
            candidate_count_before_filter=1,
            candidate_count_after_filter=1,
            requested_top_k=1,
            returned_top_k=0,
            predictions=[item],
        )


def test_diagnostic_and_run_metadata_are_serializable() -> None:
    diagnostic = DiagnosticRecord(
        source_filename="image.png",
        status="ignored",
        stage="input",
        error_code=ErrorCode.UNSUPPORTED_EXTENSION,
        message="不支持的文件类型",
        ignored_reason="unsupported extension",
    )
    metadata = RunMetadata(
        run_id="run-001",
        application={"version": "0.1.0.dev0", "git_commit": "UNKNOWN"},
        model=ModelInfo(model_id="fake-cpicann", backend="fake", num_classes=8),
        runtime={"python": "3.11", "device": "cpu"},
        preprocessing=PreprocessingConfig(),
        top_k=5,
        inputs=[{"source_filename": "image.png"}],
    )

    assert json.loads(diagnostic.model_dump_json())["error_code"] == "UNSUPPORTED_EXTENSION"
    assert json.loads(metadata.model_dump_json())["model"]["backend"] == "fake"


def test_preprocessing_config_rejects_invalid_angle_range() -> None:
    with pytest.raises(ValidationError):
        PreprocessingConfig(two_theta_min=80.0, two_theta_max=10.0)
