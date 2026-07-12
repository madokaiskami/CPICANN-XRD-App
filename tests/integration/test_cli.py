from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cpicann_xrd.cli import PARTIAL_SUCCESS_EXIT_CODE, _normalize_cli_args, app


def test_predict_fake_writes_outputs_and_json(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "predict",
            "--backend",
            "fake",
            "--input",
            "examples/spectra/0-norm.txt",
            "--include-must",
            "Zr",
            "--include-must",
            "O",
            "--allowed-elements",
            "Li",
            "--allowed-elements",
            "Zr",
            "--allowed-elements",
            "O",
            "--top-k",
            "5",
            "--output",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "predict"
    assert payload["status"] == "ok"
    run_dir = Path(payload["run_dir"])
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "samples" / "0-norm" / "prediction_top5.csv").exists()


def test_batch_fake_writes_outputs(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "batch",
            "--backend",
            "fake",
            "--input",
            "examples/spectra",
            "--top-k",
            "5",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "状态：ok" in result.stdout
    assert "成功：3" in result.stdout


def test_batch_partial_success_has_stable_exit_code(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "sample.xy").write_text("10 1\n20 2\n80 3\n", encoding="utf-8")
    (input_dir / "image.png").write_bytes(b"ignored")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "--backend",
            "fake",
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == PARTIAL_SUCCESS_EXIT_CODE
    assert "状态：partial_success" in result.stdout


def test_invalid_backend_returns_parameter_error_code(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--backend",
            "invalid",
            "--input",
            "examples/spectra/0-norm.txt",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "backend must be fake or cpicann" in result.stderr


def test_models_list_json() -> None:
    result = CliRunner().invoke(app, ["models", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"][0]["model_id"] == "cpicann-single-d1"


def test_console_arg_normalization_supports_plan_style_elements() -> None:
    assert _normalize_cli_args(
        [
            "predict",
            "--include-must",
            "Zr",
            "O",
            "--allowed-elements",
            "Li",
            "Zr",
            "O",
        ]
    ) == [
        "predict",
        "--include-must",
        "Zr",
        "--include-must",
        "O",
        "--allowed-elements",
        "Li",
        "--allowed-elements",
        "Zr",
        "--allowed-elements",
        "O",
    ]
