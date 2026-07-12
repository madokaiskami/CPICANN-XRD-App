from typer.testing import CliRunner

from cpicann_xrd.cli import app
from cpicann_xrd.version import __version__


def test_version_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"cpicann-xrd {__version__}"
