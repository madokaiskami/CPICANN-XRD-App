from __future__ import annotations

import csv
import zipfile

import torch

from cpicann_xrd.catalog.catalog import load_catalog
from cpicann_xrd.core.preprocessing import preprocess_spectrum
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.reports.exporters import (
    prediction_rows,
    write_csv_atomic,
    write_observed_png,
    write_zip_bundle,
)
from cpicann_xrd.reports.markdown_report import DISCLAIMER, render_summary_report
from cpicann_xrd.schemas import FilterSpec, SpectrumData
from cpicann_xrd.services.batch_runner import PREDICTION_FIELDNAMES
from cpicann_xrd.services.predictor import PredictionService


def test_prediction_csv_rows_and_atomic_csv(tmp_path) -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    prediction = PredictionService(
        FakeBackend(num_classes=5, injected_logits=[0, 4, 3, -2, 1]),
        catalog=catalog,
    ).predict_tensor(
        sample_id="sample",
        source_filename="sample.xy",
        tensor=torch.zeros((1, 1, 4500), dtype=torch.float32),
        top_k=2,
        filter_spec=FilterSpec(),
    )
    path = tmp_path / "prediction_top5.csv"

    write_csv_atomic(path, PREDICTION_FIELDNAMES, prediction_rows(prediction))

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["cod_id"] == "FIXTURE-00001"
    assert rows[0]["filtered_rank"] == "1"


def test_markdown_report_contains_required_chinese_disclaimer() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")
    prediction = PredictionService(
        FakeBackend(num_classes=5, injected_logits=[0, 4, 3, -2, 1]),
        catalog=catalog,
    ).predict_tensor(
        sample_id="sample",
        source_filename="sample.xy",
        tensor=torch.zeros((1, 1, 4500), dtype=torch.float32),
        top_k=1,
    )
    report = render_summary_report(
        run_id="run-001",
        model_info=FakeBackend(num_classes=5).model_info,
        preprocessing_version="legacy-cpicann-v1",
        filter_spec=FilterSpec(),
        top_k=1,
        predictions=[prediction],
        diagnostics=[],
        counts={"success": 1, "failed": 0, "ignored": 0},
    )

    assert "## 运行摘要" in report
    assert DISCLAIMER in report


def test_png_and_zip_exporters_create_readable_files(tmp_path) -> None:
    spectrum = SpectrumData(
        sample_id="sample",
        source_filename="sample.xy",
        two_theta=[10.0, 20.0, 30.0],
        intensity=[1.0, 5.0, 2.0],
    )
    png_path = tmp_path / "observed_xrd.png"
    write_observed_png(png_path, spectrum)

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    write_csv_atomic(tmp_path / "table.csv", ["a"], [{"a": 1}])
    zip_path = tmp_path / "bundle.zip"
    write_zip_bundle(tmp_path, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        assert "table.csv" in archive.namelist()
        assert "bundle.zip" not in archive.namelist()


def test_preprocessing_fixture_still_generates_table_data() -> None:
    spectrum = SpectrumData(
        sample_id="sample",
        source_filename="sample.xy",
        two_theta=[10.0, 20.0, 80.0],
        intensity=[1.0, 5.0, 2.0],
    )
    preprocessed = preprocess_spectrum(spectrum)

    assert len(preprocessed.export_table()) == 4500
