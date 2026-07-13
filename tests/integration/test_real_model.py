from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from fastapi import Request
from typer.testing import CliRunner

from cpicann_xrd.api import main
from cpicann_xrd.api import service as api_service
from cpicann_xrd.catalog.catalog import load_catalog_from_manifest
from cpicann_xrd.core.preprocessing import preprocess_spectrum
from cpicann_xrd.core.spectrum_io import read_spectrum_file
from cpicann_xrd.model.cpicann_backend import CPICANNBackend
from cpicann_xrd.model.loader import load_manifest
from cpicann_xrd.schemas import FilterSpec, SamplePrediction
from cpicann_xrd.services.batch_runner import BatchRunResult, run_batch
from cpicann_xrd.services.predictor import PredictionService
from cpicann_xrd.services.runtime import build_runtime
from cpicann_xrd.web.service import stage_uploaded_files

pytestmark = pytest.mark.model

MODEL_DIR = Path("models")
MODEL_MANIFEST = Path("configs/models/cpicann-single-d1.yaml")
CATALOG_MANIFEST = Path("data/catalog/catalog_manifest.json")
SAMPLE_DIR = Path("samples/CPICANN识别")
GOLDEN_PATH = Path("tests/golden/real_model_samples.json")
FILTERED_CASE = "zr_o_allowed_li_zr_o"


@dataclass(frozen=True)
class WebUpload:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


@dataclass(frozen=True)
class ApiUpload:
    filename: str
    data: bytes

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.data
        return self.data[:size]


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_runtime() -> tuple[Any, Any]:
    _require_real_assets()
    return build_runtime("cpicann")


def test_real_model_logits_shape_and_repeatability() -> None:
    _require_real_assets()
    manifest = load_manifest(MODEL_MANIFEST)
    backend = CPICANNBackend(manifest=manifest, model_dir=MODEL_DIR, device="cpu")
    tensor = torch.zeros((1, 1, manifest.input_points), dtype=torch.float32)

    first = backend.predict_logits(tensor)
    second = backend.predict_logits(tensor)

    assert first.shape == (1, manifest.num_classes)
    assert torch.equal(first, second)


def test_real_model_golden_baseline_matches_cpu_repeatably(
    golden: dict[str, Any],
    real_runtime: tuple[Any, Any],
) -> None:
    backend, catalog = real_runtime
    model_golden = golden["model"]
    model_info = backend.model_info
    catalog_manifest = load_catalog_from_manifest(CATALOG_MANIFEST)[1]

    assert model_info.model_id == model_golden["model_id"]
    assert model_info.source_revision == model_golden["source_revision"]
    assert model_info.weight_sha256 == model_golden["weight_sha256"]
    assert model_info.catalog_sha256 == model_golden["catalog_sha256"]
    assert catalog_manifest.catalog_sha256 == model_golden["catalog_sha256"]
    assert catalog_manifest.record_count == model_golden["catalog_record_count"]

    service = PredictionService(backend, catalog=catalog)
    for sample_golden in golden["samples"]:
        tensor = _load_sample_tensor(sample_golden)

        first = _predict_all_cases(service, tensor, sample_golden)
        second = _predict_all_cases(service, tensor, sample_golden)

        assert _prediction_signature(first) == _prediction_signature(second)
        _assert_predictions_match_golden(first, sample_golden, golden["tolerances"])


def test_real_model_cli_web_api_results_are_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden: dict[str, Any],
    real_runtime: tuple[Any, Any],
) -> None:
    _require_real_assets()
    expected = _expected_case_signature(golden, FILTERED_CASE)
    sample_paths = _sample_paths(golden)
    filter_spec = FilterSpec(
        include_must=frozenset({"Zr", "O"}),
        allowed_elements=frozenset({"Li", "Zr", "O"}),
    )

    backend, catalog = real_runtime
    web_prepared = stage_uploaded_files(
        [WebUpload(path.name, path.read_bytes()) for path in sample_paths],
        input_dir=tmp_path / "web-inputs",
    )
    web_result = run_batch(
        input_paths=web_prepared.input_paths,
        output_root=tmp_path / "web-runs",
        backend=backend,
        catalog=catalog,
        filter_spec=filter_spec,
        top_k=5,
        run_id="web-real-model",
    )

    cli_result = CliRunner().invoke(
        main_cli_app(),
        [
            "batch",
            "--backend",
            "cpicann",
            "--input",
            str(sample_paths[0]),
            "--input",
            str(sample_paths[1]),
            "--input",
            str(sample_paths[2]),
            "--include-must",
            "Zr",
            "--include-must",
            "O",
            "--allowed-elements",
            "Li",
            "--allowed-elements",
            "Zr",
            "--allowed-elements",
            "O",
            "--top-k",
            "5",
            "--output",
            str(tmp_path / "cli-runs"),
            "--json",
        ],
    )

    monkeypatch.setattr(api_service, "DEFAULT_API_RUN_ROOT", tmp_path / "api-runs")
    main._cached_runtime.cache_clear()
    api_result = asyncio.run(
        main._run_uploaded_batch(
            request=_request("phase11-api-001"),
            uploads=[ApiUpload(path.name, path.read_bytes()) for path in sample_paths],
            backend_name="cpicann",
            include_must=["Zr", "O"],
            allowed_elements=["Li", "Zr", "O"],
            top_k=5,
        )
    )
    main._cached_runtime.cache_clear()

    assert cli_result.exit_code == 0, cli_result.output
    cli_run_dir = Path(json.loads(cli_result.stdout)["run_dir"])

    assert _batch_signature(web_result) == expected
    assert _csv_run_signature(cli_run_dir) == expected
    assert _sample_prediction_signature(api_result.predictions) == expected


def main_cli_app() -> Any:
    from cpicann_xrd.cli import app

    return app


def _require_real_assets() -> None:
    if not (MODEL_DIR / "cpicann-single-d1" / "CPICANNsingle_phase_D1.state_dict.pth").exists():
        pytest.skip("real CPICANN model weights are not available")
    if not SAMPLE_DIR.exists():
        pytest.skip(f"sample directory is not available: {SAMPLE_DIR}")


def _sample_paths(golden: dict[str, Any]) -> list[Path]:
    return [SAMPLE_DIR / sample["source_filename"] for sample in golden["samples"]]


def _load_sample_tensor(sample_golden: dict[str, Any]) -> torch.Tensor:
    path = SAMPLE_DIR / sample_golden["source_filename"]
    read_result = read_spectrum_file(path)
    assert read_result.status == "success"
    assert read_result.spectrum is not None
    assert read_result.spectrum.sha256 == sample_golden["input_sha256"]
    assert len(read_result.spectrum.two_theta) == sample_golden["rows_read"]

    preprocessed = preprocess_spectrum(read_result.spectrum)
    assert preprocessed.array_sha256 == sample_golden["preprocessed_sha256"]
    return torch.from_numpy(preprocessed.model_input)


def _predict_all_cases(
    service: PredictionService,
    tensor: torch.Tensor,
    sample_golden: dict[str, Any],
) -> dict[str, SamplePrediction]:
    return {
        case_name: service.predict_tensor(
            sample_id=Path(sample_golden["source_filename"]).stem,
            source_filename=sample_golden["source_filename"],
            tensor=tensor,
            top_k=5,
            filter_spec=_filter_spec(case_golden["filter"]),
        )
        for case_name, case_golden in sample_golden["cases"].items()
    }


def _filter_spec(filter_golden: dict[str, Any]) -> FilterSpec:
    allowed_elements = filter_golden["allowed_elements"]
    return FilterSpec(
        include_must=frozenset(filter_golden["include_must"]),
        allowed_elements=None if allowed_elements is None else frozenset(allowed_elements),
    )


def _assert_predictions_match_golden(
    predictions_by_case: dict[str, SamplePrediction],
    sample_golden: dict[str, Any],
    tolerances: dict[str, float],
) -> None:
    for case_name, prediction in predictions_by_case.items():
        case_golden = sample_golden["cases"][case_name]
        assert prediction.candidate_count_before_filter == 23073
        assert (
            prediction.candidate_count_after_filter == case_golden["candidate_count_after_filter"]
        )
        assert prediction.returned_top_k == 5
        assert prediction.filter_spec.model_dump(mode="json") == case_golden["filter"]
        for item, expected in zip(prediction.predictions, case_golden["top5"], strict=True):
            (
                filtered_rank,
                global_rank,
                class_index,
                cod_id,
                formula,
                reduced_formula,
                elements,
                space_group,
                space_group_number,
                raw_logit,
                probability,
                confidence,
            ) = expected
            assert item.filtered_rank == filtered_rank
            assert item.global_rank == global_rank
            assert item.class_index == class_index
            assert item.phase.cod_id == cod_id
            assert item.phase.formula == formula
            assert item.phase.reduced_formula == reduced_formula
            assert sorted(item.phase.elements) == elements
            assert item.phase.space_group == space_group
            assert item.phase.space_group_number == space_group_number
            assert item.raw_logit == pytest.approx(raw_logit, abs=tolerances["raw_logit_abs"])
            assert item.unfiltered_probability == pytest.approx(
                probability,
                abs=tolerances["probability_abs"],
            )
            assert item.filtered_confidence == pytest.approx(
                confidence,
                abs=tolerances["confidence_abs"],
            )


def _prediction_signature(
    predictions_by_case: dict[str, SamplePrediction],
) -> dict[str, list[tuple[int, str, int, float]]]:
    return {
        case_name: [
            (
                item.class_index,
                item.phase.cod_id,
                item.global_rank,
                item.filtered_confidence,
            )
            for item in prediction.predictions
        ]
        for case_name, prediction in predictions_by_case.items()
    }


def _expected_case_signature(
    golden: dict[str, Any],
    case_name: str,
) -> dict[str, list[tuple[int, str, int, int]]]:
    return {
        sample["source_filename"]: [
            (int(row[2]), str(row[3]), int(row[1]), int(row[8]))
            for row in sample["cases"][case_name]["top5"]
        ]
        for sample in golden["samples"]
    }


def _batch_signature(result: BatchRunResult) -> dict[str, list[tuple[int, str, int, int]]]:
    assert result.counts == {"success": 3, "failed": 0, "ignored": 0}
    return _sample_prediction_signature(result.predictions)


def _sample_prediction_signature(
    predictions: list[SamplePrediction],
) -> dict[str, list[tuple[int, str, int, int]]]:
    return {
        prediction.source_filename: [
            (
                item.class_index,
                item.phase.cod_id,
                item.global_rank,
                item.phase.space_group_number or 0,
            )
            for item in prediction.predictions
        ]
        for prediction in predictions
    }


def _csv_run_signature(run_dir: Path) -> dict[str, list[tuple[int, str, int, int]]]:
    result: dict[str, list[tuple[int, str, int, int]]] = {}
    for sample_dir in sorted((run_dir / "samples").iterdir()):
        rows = _read_csv(sample_dir / "prediction_top5.csv")
        source_filename = f"{sample_dir.name}.txt"
        result[source_filename] = [
            (
                int(row["class_index"]),
                row["cod_id"],
                int(row["global_rank"]),
                int(row["space_group_number"]),
            )
            for row in rows
        ]
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _request(request_id: str) -> Request:
    return cast(Request, SimpleNamespace(state=SimpleNamespace(request_id=request_id)))
