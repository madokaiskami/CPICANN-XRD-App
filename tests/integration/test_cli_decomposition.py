"""CLI tests for explicit optional decomposition entry points."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from cpicann_xrd.cli import app


def test_cli_decompose_disabled_returns_structured_diagnostic() -> None:
    result = CliRunner().invoke(
        app,
        [
            "decompose",
            "--input",
            "examples/spectra/0-norm.txt",
            "--json",
        ],
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 1
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "INVALID_REQUEST"
    assert "xdecomposer_disabled" in payload["error"]["message"]
    assert "Traceback" not in result.stdout


def test_cli_decompose_stub_returns_components() -> None:
    result = CliRunner().invoke(
        app,
        [
            "decompose",
            "--input",
            "examples/spectra/0-norm.txt",
            "--xdecomposer-backend",
            "stub",
            "--json",
        ],
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["model_id"] == "stub-xdecomposer-v1"
    assert payload["preprocessing_version"] == "xdecomposer-v1"
    assert [component["component_index"] for component in payload["components"]] == [1, 2]


def test_cli_decompose_and_identify_stub_matches_orchestration_shape() -> None:
    result = CliRunner().invoke(
        app,
        [
            "decompose-and-identify",
            "--input",
            "examples/spectra/0-norm.txt",
            "--xdecomposer-backend",
            "stub",
            "--backend",
            "fake",
            "--allowed-elements",
            "Li",
            "--allowed-elements",
            "O",
            "--allowed-elements",
            "Zr",
            "--top-k",
            "2",
            "--json",
        ],
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["mode"] == "multiphase-identification"
    assert payload["status"] == "success"
    assert len(payload["components"]) == 2
    assert all(component["cpicann"]["returned_top_k"] == 2 for component in payload["components"])
    assert all(
        component["cpicann"]["candidate_count_after_filter"] == 3
        for component in payload["components"]
    )
    assert "conditional confidence" in payload["report_markdown"]
