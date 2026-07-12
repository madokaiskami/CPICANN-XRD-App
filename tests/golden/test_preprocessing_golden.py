from __future__ import annotations

from pathlib import Path

import pytest

from cpicann_xrd.core.preprocessing import preprocess_spectrum
from cpicann_xrd.core.spectrum_io import read_spectrum_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECTRA_DIR = PROJECT_ROOT / "examples" / "spectra"

GOLDEN = {
    "0-norm.txt": {
        "valid_rows": 8,
        "angle_min": 10.0,
        "angle_max": 80.0,
        "shape": (1, 1, 4500),
        "dtype": "float32",
        "sha256": "7c47a32fce25d5ac460f5548145d9181b22ef69f9c661819e2b253a89d1bc569",
        "warnings": [],
    },
    "1-norm.txt": {
        "valid_rows": 8,
        "angle_min": 10.0,
        "angle_max": 80.0,
        "shape": (1, 1, 4500),
        "dtype": "float32",
        "sha256": "c735bd05fb0ca43a4d10150ead2b0bc30f041717557e72bf346e0ca03749a46f",
        "warnings": [],
    },
    "3-norm.txt": {
        "valid_rows": 8,
        "angle_min": 12.0,
        "angle_max": 78.0,
        "shape": (1, 1, 4500),
        "dtype": "float32",
        "sha256": "b1a984c78af38aa6d130873c4f8aa5ddd199a1b1994853a033653f5a8be6e454",
        "warnings": [
            "Prepended 10.0 degree boundary using first intensity",
            "Appended 80.0 degree boundary using last intensity",
        ],
    },
}


@pytest.mark.parametrize("filename", sorted(GOLDEN))
def test_preprocessing_golden_for_synthetic_norm_fixtures(filename: str) -> None:
    # These are small redistributable Phase 3 fixtures because upstream 0/1/3-norm
    # files were not available during Phase 1 audit.
    expected = GOLDEN[filename]

    read_result = read_spectrum_file(SPECTRA_DIR / filename)

    assert read_result.status == "success"
    assert read_result.spectrum is not None
    assert len(read_result.spectrum.two_theta) == expected["valid_rows"]
    assert read_result.spectrum.two_theta[0] == expected["angle_min"]
    assert read_result.spectrum.two_theta[-1] == expected["angle_max"]

    preprocessed = preprocess_spectrum(read_result.spectrum)

    assert preprocessed.model_input.shape == expected["shape"]
    assert str(preprocessed.model_input.dtype) == expected["dtype"]
    assert preprocessed.array_sha256 == expected["sha256"]
    assert list(preprocessed.warnings) == expected["warnings"]
