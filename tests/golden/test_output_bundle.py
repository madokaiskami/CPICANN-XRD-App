from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from cpicann_xrd.catalog.catalog import load_catalog
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.reports.markdown_report import DISCLAIMER
from cpicann_xrd.services.batch_runner import run_batch


def test_output_bundle_has_stable_phase6_structure(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "0-norm.txt").write_text("10 1\n20 3\n80 2\n", encoding="utf-8")
    (input_dir / "observed.png").write_bytes(b"ignored")
    (input_dir / "archive.rar").write_bytes(b"ignored")
    (input_dir / "bad.csv").write_text("header\nnot numeric\n", encoding="utf-8")

    result = run_batch(
        input_paths=[input_dir],
        output_root=tmp_path / "runs",
        run_id="golden-run",
        backend=FakeBackend(num_classes=5, injected_logits=[0, 4, 3, -2, 1]),
        catalog=load_catalog("data/catalog/phase5_fixture_catalog.csv"),
        top_k=5,
    )

    expected_paths = {
        "summary.csv",
        "summary_report.md",
        "run_metadata.json",
        "diagnostics.csv",
        "input_manifest.csv",
        "result_bundle.zip",
        "samples/0-norm/prediction_top5.csv",
        "samples/0-norm/observed_xrd.png",
        "samples/0-norm/preprocessed_xrd.csv",
    }
    actual_paths = {
        str(path.relative_to(result.run_dir))
        for path in result.run_dir.rglob("*")
        if path.is_file()
    }

    assert expected_paths <= actual_paths
    assert DISCLAIMER in (result.run_dir / "summary_report.md").read_text(encoding="utf-8")
    with (result.run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["top1_cod_id"] == "FIXTURE-00001"
    with zipfile.ZipFile(result.run_dir / "result_bundle.zip") as archive:
        assert expected_paths - {"result_bundle.zip"} <= set(archive.namelist())
