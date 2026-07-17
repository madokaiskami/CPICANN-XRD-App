from __future__ import annotations

import hashlib
from pathlib import Path

from cpicann_xrd.decomposition.artifacts import write_decomposition_artifacts
from cpicann_xrd.decomposition.schemas import XDecomposerRequest
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend


def test_decomposition_report_contains_scientific_limitations(tmp_path: Path) -> None:
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

    report = paths.report_md.read_text(encoding="utf-8")

    assert "XDecomposer 多相分解报告" in report
    assert "不是 Rietveld 定量相含量" in report
    assert "不等同于已经确认的物相" in report
    assert "component" in report
    assert "CPICANN" in report
