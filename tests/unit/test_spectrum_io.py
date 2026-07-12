from __future__ import annotations

from pathlib import Path

import pytest

from cpicann_xrd.core.spectrum_io import read_spectrum_file
from cpicann_xrd.exceptions import ErrorCode


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_two_column_txt_with_header(tmp_path: Path) -> None:
    path = write_text(tmp_path / "sample.txt", "angle intensity\n10 1\n20 5\n80 2\n")

    result = read_spectrum_file(path)

    assert result.status == "success"
    assert result.spectrum is not None
    assert result.spectrum.two_theta == [10.0, 20.0, 80.0]
    assert result.spectrum.intensity == [1.0, 5.0, 2.0]
    assert result.rows_read == 4
    assert result.rows_invalid == 0
    assert any("Skipped header" in diagnostic.message for diagnostic in result.diagnostics)


def test_reads_three_column_txt_as_difference(tmp_path: Path) -> None:
    path = write_text(tmp_path / "sample.txt", "10 11 1\n20 25 5\n80 12 2\n")

    result = read_spectrum_file(path)

    assert result.status == "success"
    assert result.spectrum is not None
    assert result.spectrum.intensity == [10.0, 20.0, 10.0]


def test_reads_csv_with_header_and_without_header(tmp_path: Path) -> None:
    with_header = write_text(
        tmp_path / "with_header.csv", "two_theta,intensity\n10,1\n20,2\n80,3\n"
    )
    no_header = write_text(tmp_path / "no_header.csv", "10,1\n20,2\n80,3\n")

    header_result = read_spectrum_file(with_header)
    no_header_result = read_spectrum_file(no_header)

    assert header_result.status == "success"
    assert header_result.spectrum is not None
    assert header_result.spectrum.two_theta == [10.0, 20.0, 80.0]
    assert no_header_result.status == "success"
    assert no_header_result.spectrum is not None
    assert no_header_result.spectrum.two_theta == [10.0, 20.0, 80.0]


def test_reads_tab_delimited_xy_and_uppercase_extension(tmp_path: Path) -> None:
    xy_path = write_text(tmp_path / "sample.xy", "10\t1\n20\t2\n80\t3\n")
    upper_path = write_text(tmp_path / "upper.TXT", "10 1\n20 2\n80 3\n")

    xy_result = read_spectrum_file(xy_path)
    upper_result = read_spectrum_file(upper_path)

    assert xy_result.status == "success"
    assert upper_result.status == "success"


@pytest.mark.parametrize("filename", ["image.png", "archive.rar"])
def test_unsupported_files_are_ignored(tmp_path: Path, filename: str) -> None:
    path = write_text(tmp_path / filename, "not a spectrum")

    result = read_spectrum_file(path)

    assert result.status == "ignored"
    assert result.spectrum is None
    assert result.diagnostics[0].error_code == ErrorCode.UNSUPPORTED_EXTENSION


def test_empty_file_fails_with_error_code(tmp_path: Path) -> None:
    path = write_text(tmp_path / "empty.csv", "")

    result = read_spectrum_file(path)

    assert result.status == "failed"
    assert result.diagnostics[-1].error_code == ErrorCode.EMPTY_FILE


def test_non_numeric_file_fails_with_error_code(tmp_path: Path) -> None:
    path = write_text(tmp_path / "bad.txt", "hello world\nfoo bar\n")

    result = read_spectrum_file(path)

    assert result.status == "failed"
    assert result.diagnostics[-1].error_code == ErrorCode.NO_VALID_NUMERIC_ROWS


def test_descending_angles_are_sorted_and_diagnosed(tmp_path: Path) -> None:
    path = write_text(tmp_path / "descending.txt", "80 3\n20 2\n10 1\n")

    result = read_spectrum_file(path)

    assert result.status == "success"
    assert result.spectrum is not None
    assert result.spectrum.two_theta == [10.0, 20.0, 80.0]
    assert any("Sorted rows" in diagnostic.message for diagnostic in result.diagnostics)


def test_duplicate_angles_are_aggregated_by_mean(tmp_path: Path) -> None:
    path = write_text(tmp_path / "duplicates.txt", "10 1\n20 2\n20 6\n80 3\n")

    result = read_spectrum_file(path)

    assert result.status == "success"
    assert result.spectrum is not None
    assert result.spectrum.two_theta == [10.0, 20.0, 80.0]
    assert result.spectrum.intensity == [1.0, 4.0, 3.0]
    assert any("duplicate angle" in diagnostic.message for diagnostic in result.diagnostics)


@pytest.mark.parametrize("value", ["NaN", "Inf", "-Inf"])
def test_non_finite_values_fail(tmp_path: Path, value: str) -> None:
    path = write_text(tmp_path / "non_finite.txt", f"10 1\n20 {value}\n80 3\n")

    result = read_spectrum_file(path)

    assert result.status == "failed"
    assert result.diagnostics[-1].error_code == ErrorCode.NON_FINITE_VALUES
