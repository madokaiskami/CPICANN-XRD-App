from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from cpicann_xrd.decomposition.artifacts import write_decomposition_artifacts
from cpicann_xrd.decomposition.schemas import XDecomposerRequest
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend


def test_decomposition_bundle_contains_outputs_but_no_weights(tmp_path: Path) -> None:
    request = XDecomposerRequest(
        sample_id="mix-001",
        two_theta=[10.0, 45.0, 80.0],
        intensity=[0.0, 10.0, 0.0],
        return_component_patterns=True,
    )
    result = StubDecompositionBackend().decompose(request)

    paths = write_decomposition_artifacts(
        run_dir=tmp_path / "run",
        request=request,
        result=result,
        source_sha256=hashlib.sha256(b"input").hexdigest(),
        source_filename="mixture.xy",
        git_commit="abc123",
    )

    with zipfile.ZipFile(paths.bundle_zip) as archive:
        names = set(archive.namelist())

    assert "decomposition_summary.csv" in names
    assert "decomposition_metadata.json" in names
    assert "decomposition_report.md" in names
    assert "components/component_01.csv" in names
    assert "components/component_01.png" in names
    assert not any(name.endswith((".pt", ".pth", ".ckpt", ".onnx", ".env")) for name in names)
    assert not any("token" in name.lower() for name in names)
