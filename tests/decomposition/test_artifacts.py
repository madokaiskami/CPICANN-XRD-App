from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cpicann_xrd.decomposition.artifacts import SUMMARY_FIELDS, write_decomposition_artifacts
from cpicann_xrd.decomposition.schemas import XDecomposerRequest
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend


def test_decomposition_artifacts_complete_tree(tmp_path: Path) -> None:
    request = _request()
    result = StubDecompositionBackend().decompose(
        request.model_copy(update={"return_component_patterns": True})
    )

    paths = write_decomposition_artifacts(
        run_dir=tmp_path / "run",
        request=request,
        result=result,
        source_sha256=_source_sha(request),
        source_filename="mixture.xy",
        git_commit="abc123",
        upstream_commit="upstream123",
        checkpoint_hashes={"separator": "0" * 64, "mae": "1" * 64},
        reference_bank_hash="2" * 64,
        elapsed_seconds=1.25,
    )

    expected = [
        paths.summary_csv,
        paths.metadata_json,
        paths.report_md,
        paths.observed_png,
        paths.reconstruction_png,
        paths.residual_png,
        paths.component_overview_png,
        paths.diagnostics_csv,
        paths.bundle_zip,
        paths.run_dir / "components" / "component_01.csv",
        paths.run_dir / "components" / "component_01.png",
        paths.run_dir / "components" / "component_02.csv",
        paths.run_dir / "components" / "component_02.png",
    ]
    for path in expected:
        assert path.is_file(), path


def test_decomposition_summary_schema_is_fixed(tmp_path: Path) -> None:
    request = _request()
    result = StubDecompositionBackend().decompose(
        request.model_copy(update={"return_component_patterns": True})
    )

    paths = write_decomposition_artifacts(
        run_dir=tmp_path / "run",
        request=request,
        result=result,
        source_sha256=_source_sha(request),
        source_filename="mixture.xy",
        git_commit="abc123",
    )

    with paths.summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == SUMMARY_FIELDS
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "mix-001"


def test_decomposition_metadata_json_is_parseable(tmp_path: Path) -> None:
    request = _request()
    result = StubDecompositionBackend().decompose(
        request.model_copy(update={"return_component_patterns": True})
    )

    paths = write_decomposition_artifacts(
        run_dir=tmp_path / "run",
        request=request,
        result=result,
        source_sha256=_source_sha(request),
        source_filename="mixture.xy",
        git_commit="abc123",
    )

    payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

    assert payload["sample_id"] == "mix-001"
    assert payload["preprocessing_protocol"] == "xdecomposer-v1"
    assert payload["xdecomposer_axis"] == [10.0, 80.0, 3500]
    assert payload["model_id"] == "stub-xdecomposer-v1"


def _request() -> XDecomposerRequest:
    return XDecomposerRequest(
        sample_id="mix-001",
        source_filename="mixture.xy",
        two_theta=[10.0, 25.0, 45.0, 65.0, 80.0],
        intensity=[0.0, 3.0, 10.0, 4.0, 0.0],
        max_sources=4,
        return_component_patterns=True,
    )


def _source_sha(request: XDecomposerRequest) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{x},{y}" for x, y in zip(request.two_theta, request.intensity, strict=True)
        ).encode("utf-8")
    ).hexdigest()
