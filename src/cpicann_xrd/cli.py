"""Command line entry points for CPICANN-XRD."""

from __future__ import annotations

from typing import Annotated

import typer

from cpicann_xrd.services.doctor import run_doctor
from cpicann_xrd.settings import load_settings
from cpicann_xrd.version import __version__

app = typer.Typer(
    add_completion=False,
    help="CPICANN-XRD phase-identification application.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"cpicann-xrd {__version__}")
        raise typer.Exit(code=0)


@app.callback(invoke_without_command=True)
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Show application version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run the CPICANN-XRD command line interface."""
    _ = version


@app.command()
def doctor(
    backend: Annotated[
        str,
        typer.Option("--backend", help="Backend to check: fake or cpicann."),
    ] = "fake",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Check runtime configuration and selected backend."""
    settings = load_settings(cli_overrides={"backend": backend})
    result = run_doctor(settings)
    if json_output:
        typer.echo(result.model_dump_json())
    else:
        typer.echo(f"状态：{result.status}")
        typer.echo(f"后端：{result.backend}")
        typer.echo(f"模型：{result.model_id}")
        typer.echo(f"信息：{result.message}")
    if result.status != "ok":
        raise typer.Exit(code=1)


def main() -> None:
    """Console script entry point."""
    app()
