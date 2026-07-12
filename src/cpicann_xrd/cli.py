"""Command line entry points for CPICANN-XRD."""

from __future__ import annotations

from typing import Annotated

import typer

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


def main() -> None:
    """Console script entry point."""
    app()
