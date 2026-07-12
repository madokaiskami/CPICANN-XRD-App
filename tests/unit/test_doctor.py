from __future__ import annotations

from typer.testing import CliRunner

from cpicann_xrd.cli import app
from cpicann_xrd.services.doctor import run_doctor
from cpicann_xrd.settings import AppSettings


def test_doctor_fake_backend_returns_ok() -> None:
    result = run_doctor(AppSettings(backend="fake"))

    assert result.status == "ok"
    assert result.backend == "fake"
    assert result.details["logits_shape"] == [1, 8]


def test_cli_doctor_fake_backend() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--backend", "fake"])

    assert result.exit_code == 0
    assert "状态：ok" in result.stdout
    assert "后端：fake" in result.stdout


def test_cli_doctor_fake_json() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--backend", "fake", "--json"])

    assert result.exit_code == 0
    assert '"status":"ok"' in result.stdout
