from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from cpicann_xrd.schemas import FilterSpec
from cpicann_xrd.services.batch_runner import run_batch
from cpicann_xrd.services.runtime import build_runtime
from cpicann_xrd.web.app import _markdown_table
from cpicann_xrd.web.service import available_elements, stage_uploaded_files


@dataclass(frozen=True)
class FakeUpload:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def test_stage_uploaded_files_limits_and_deduplicates_names(tmp_path: Path) -> None:
    prepared = stage_uploaded_files(
        [
            FakeUpload("../0-norm.txt", b"10 1\n20 2\n"),
            FakeUpload("0-norm.txt", b"10 3\n20 4\n"),
            FakeUpload(".hidden", b"10 5\n20 6\n"),
            FakeUpload("too-large.xy", b"1234567890123"),
        ],
        input_dir=tmp_path / "inputs",
        max_bytes=12,
    )

    assert [path.name for path in prepared.input_paths] == [
        "0-norm.txt",
        "0-norm_2.txt",
        "hidden",
    ]
    assert [path.parent for path in prepared.input_paths] == [prepared.input_dir] * 3
    assert prepared.input_paths[0].read_bytes() == b"10 1\n20 2\n"
    assert prepared.warnings == ["too-large.xy 超过大小限制，已跳过。"]


def test_stage_uploaded_files_rejects_too_many_files(tmp_path: Path) -> None:
    uploads = [FakeUpload(f"{index}.txt", b"10 1\n") for index in range(3)]

    with pytest.raises(ValueError, match="最多允许上传 2 个文件"):
        stage_uploaded_files(uploads, input_dir=tmp_path / "inputs", max_files=2)


def test_available_elements_excludes_fake_placeholder() -> None:
    elements = available_elements()

    assert elements == sorted(elements)
    assert "Li" in elements
    assert "O" in elements
    assert "Zr" in elements
    assert "X" not in elements


def test_fake_web_service_path_records_unsupported_files(tmp_path: Path) -> None:
    uploaded_files = [
        _example_upload("0-norm.txt"),
        _example_upload("1-norm.txt"),
        _example_upload("3-norm.txt"),
        FakeUpload("observed.png", b"not a spectrum"),
    ]
    prepared = stage_uploaded_files(uploaded_files, input_dir=tmp_path / "inputs")
    backend, catalog = build_runtime("fake")

    result = run_batch(
        input_paths=prepared.input_paths,
        output_root=tmp_path / "runs",
        backend=backend,
        catalog=catalog,
        filter_spec=FilterSpec(
            include_must=frozenset({"Zr", "O"}),
            allowed_elements=frozenset({"Li", "Zr", "O"}),
        ),
        top_k=5,
        run_id="web-run-001",
    )

    assert result.counts == {"success": 3, "failed": 0, "ignored": 1}
    assert prepared.warnings == []
    assert (result.run_dir / "summary.csv").exists()
    assert (result.run_dir / "summary_report.md").exists()
    assert (result.run_dir / "result_bundle.zip").exists()
    assert all(
        (result.run_dir / "samples" / item.sample_id / "observed_xrd.png").exists()
        for item in result.predictions
    )
    assert all(item.predictions[0].phase.cod_id == "FIXTURE-00001" for item in result.predictions)
    assert all(item.predictions[0].filtered_confidence > 0 for item in result.predictions)

    summary_rows = _read_csv(result.run_dir / "summary.csv")
    assert [row["status"] for row in summary_rows].count("success") == 3
    assert [row["status"] for row in summary_rows].count("ignored") == 1

    diagnostics_rows = _read_csv(result.run_dir / "diagnostics.csv")
    assert any(
        row["source_filename"] == "observed.png"
        and row["status"] == "ignored"
        and row["error_code"] == "UNSUPPORTED_EXTENSION"
        for row in diagnostics_rows
    )


def test_web_prediction_table_includes_space_group() -> None:
    markdown = _markdown_table(
        [
            {
                "COD ID": "FIXTURE-00001",
                "模型预测物相": "ZrO2",
                "空间群": "P4_2/nmc",
                "空间群编号": 137,
            }
        ]
    )

    assert "空间群" in markdown
    assert "空间群编号" in markdown
    assert "P4_2/nmc" in markdown
    assert "137" in markdown


def _example_upload(name: str) -> FakeUpload:
    return FakeUpload(name=name, data=(Path("examples/spectra") / name).read_bytes())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
