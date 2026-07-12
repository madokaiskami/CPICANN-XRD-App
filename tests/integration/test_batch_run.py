from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from cpicann_xrd.catalog.catalog import load_catalog
from cpicann_xrd.model.fake_backend import FakeBackend
from cpicann_xrd.services.batch_runner import run_batch


def test_batch_run_isolates_errors_and_writes_standard_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_spectrum(input_dir / "0-norm.txt", scale=1.0)
    _write_spectrum(input_dir / "1-norm.txt", scale=2.0)
    _write_spectrum(input_dir / "3-norm.txt", scale=3.0)
    (input_dir / "observed.png").write_bytes(b"not model input")
    (input_dir / "archive.rar").write_bytes(b"not model input")
    (input_dir / "bad.csv").write_text("two_theta,intensity\nbad,row\n", encoding="utf-8")

    result = run_batch(
        input_paths=[input_dir],
        output_root=tmp_path / "runs",
        run_id="run-001",
        backend=FakeBackend(num_classes=5, injected_logits=[0, 4, 3, -2, 1]),
        catalog=load_catalog("data/catalog/phase5_fixture_catalog.csv"),
        top_k=5,
    )

    assert result.counts == {"success": 3, "failed": 1, "ignored": 2}
    run_dir = result.run_dir
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "summary_report.md").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "diagnostics.csv").exists()
    assert (run_dir / "input_manifest.csv").exists()
    assert (run_dir / "result_bundle.zip").exists()

    sample_dirs = sorted((run_dir / "samples").iterdir())
    assert [path.name for path in sample_dirs] == ["0-norm", "1-norm", "3-norm"]
    for sample_dir in sample_dirs:
        assert (sample_dir / "prediction_top5.csv").exists()
        assert (sample_dir / "observed_xrd.png").read_bytes().startswith(b"\x89PNG")
        assert (sample_dir / "preprocessed_xrd.csv").exists()

    summary_rows = _read_csv(run_dir / "summary.csv")
    assert [row["status"] for row in summary_rows].count("success") == 3
    assert [row["status"] for row in summary_rows].count("ignored") == 2
    assert [row["status"] for row in summary_rows].count("failed") == 1
    assert all(row["top1_cod_id"] == "FIXTURE-00001" for row in summary_rows[:3])

    diagnostics_rows = _read_csv(run_dir / "diagnostics.csv")
    assert any(row["error_code"] == "UNSUPPORTED_EXTENSION" for row in diagnostics_rows)
    assert any(row["error_code"] == "NO_VALID_NUMERIC_ROWS" for row in diagnostics_rows)

    metadata_text = (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["run_id"] == "run-001"
    assert metadata["model"]["backend"] == "fake"
    assert str(tmp_path) not in metadata_text

    with zipfile.ZipFile(run_dir / "result_bundle.zip") as archive:
        names = set(archive.namelist())
    assert "summary.csv" in names
    assert "samples/0-norm/prediction_top5.csv" in names


def _write_spectrum(path: Path, *, scale: float) -> None:
    rows = [
        (10.0, 1.0 * scale),
        (20.0, 4.0 * scale),
        (40.0, 2.0 * scale),
        (80.0, 3.0 * scale),
    ]
    path.write_text(
        "\n".join(f"{angle}\t{intensity}" for angle, intensity in rows) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
