"""Command line entry points for CPICANN-XRD."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from cpicann_xrd.catalog.catalog import load_catalog_from_manifest
from cpicann_xrd.exceptions import CpicannXrdError
from cpicann_xrd.model.loader import load_manifest
from cpicann_xrd.schemas import FilterSpec
from cpicann_xrd.services.batch_runner import BatchRunResult, run_batch
from cpicann_xrd.services.doctor import run_doctor
from cpicann_xrd.services.runtime import (
    DEFAULT_CATALOG_MANIFEST,
    DEFAULT_MODEL_MANIFEST,
    build_runtime,
)
from cpicann_xrd.settings import load_settings
from cpicann_xrd.version import __version__

PARTIAL_SUCCESS_EXIT_CODE = 3

app = typer.Typer(
    add_completion=False,
    help="CPICANN-XRD phase-identification application.",
)
models_app = typer.Typer(add_completion=False, help="模型资产管理。")
app.add_typer(models_app, name="models")


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


@app.command()
def predict(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, readable=True, help="单个谱图输入文件。"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="输出 runs 根目录。"),
    ] = Path("runs"),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Backend: fake or cpicann."),
    ] = "fake",
    include_must: Annotated[
        list[str] | None,
        typer.Option("--include-must", help="候选相必须包含的元素，可重复传入。"),
    ] = None,
    allowed_elements: Annotated[
        list[str] | None,
        typer.Option("--allowed-elements", help="候选相允许出现的元素范围，可重复传入。"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="返回候选数量。")] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """对单个谱图执行预测并写出标准结果目录。"""
    result = _run_cli_batch(
        input_paths=[input_path],
        output=output,
        backend_name=backend,
        include_must=include_must,
        allowed_elements=allowed_elements,
        top_k=top_k,
    )
    _emit_run_result(result, json_output=json_output, mode="predict")
    _exit_for_counts(result.counts)


@app.command()
def batch(
    input_paths: Annotated[
        list[Path],
        typer.Option("--input", exists=True, readable=True, help="输入文件或目录，可重复传入。"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="输出 runs 根目录。"),
    ] = Path("runs"),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Backend: fake or cpicann."),
    ] = "fake",
    include_must: Annotated[
        list[str] | None,
        typer.Option("--include-must", help="候选相必须包含的元素，可重复传入。"),
    ] = None,
    allowed_elements: Annotated[
        list[str] | None,
        typer.Option("--allowed-elements", help="候选相允许出现的元素范围，可重复传入。"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="返回候选数量。")] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """对目录或多个文件执行批量预测并写出标准结果目录。"""
    result = _run_cli_batch(
        input_paths=input_paths,
        output=output,
        backend_name=backend,
        include_must=include_must,
        allowed_elements=allowed_elements,
        top_k=top_k,
    )
    _emit_run_result(result, json_output=json_output, mode="batch")
    _exit_for_counts(result.counts)


@models_app.command("list")
def models_list(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """列出内置模型配置。"""
    manifest = load_manifest(DEFAULT_MODEL_MANIFEST)
    payload = {
        "models": [
            {
                "model_id": manifest.model_id,
                "backend": manifest.backend,
                "num_classes": manifest.num_classes,
                "catalog_sha256": manifest.catalog_sha256,
            }
        ]
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    typer.echo("可用模型：")
    typer.echo(f"- {manifest.model_id} ({manifest.backend}, classes={manifest.num_classes})")


@models_app.command("verify")
def models_verify(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """验证已提交的真实 catalog manifest。"""
    catalog, manifest = load_catalog_from_manifest(DEFAULT_CATALOG_MANIFEST)
    payload = {
        "model_id": manifest.model_id,
        "catalog_sha256": manifest.catalog_sha256,
        "record_count": len(catalog),
        "status": "ok",
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo("状态：ok")
        typer.echo(f"模型：{manifest.model_id}")
        typer.echo(f"Catalog 记录数：{len(catalog)}")
        typer.echo(f"Catalog SHA-256：{manifest.catalog_sha256}")


@models_app.command("install")
def models_install() -> None:
    """提示用户显式安装真实模型权重。"""
    typer.echo("不会默认下载模型。请显式使用 scripts/download_model.py 或放置本地权重后验证。")
    raise typer.Exit(code=1)


def _run_cli_batch(
    *,
    input_paths: list[Path],
    output: Path,
    backend_name: str,
    include_must: list[str] | None,
    allowed_elements: list[str] | None,
    top_k: int,
) -> BatchRunResult:
    try:
        backend, catalog = build_runtime(backend_name)
        return run_batch(
            input_paths=input_paths,
            output_root=output,
            backend=backend,
            catalog=catalog,
            filter_spec=FilterSpec(
                include_must=frozenset(include_must or []),
                allowed_elements=None if allowed_elements is None else frozenset(allowed_elements),
            ),
            top_k=top_k,
        )
    except CpicannXrdError as exc:
        typer.echo(f"{exc.error_code.value}: {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _emit_run_result(result: BatchRunResult, *, json_output: bool, mode: str) -> None:
    payload = {
        "mode": mode,
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "counts": result.counts,
        "status": _status_from_counts(result.counts),
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    typer.echo(f"状态：{payload['status']}")
    typer.echo(f"Run ID：{result.run_id}")
    typer.echo(f"输出目录：{result.run_dir}")
    typer.echo(f"成功：{result.counts['success']}")
    typer.echo(f"失败：{result.counts['failed']}")
    typer.echo(f"忽略：{result.counts['ignored']}")
    if any(prediction.backend == "fake" for prediction in result.predictions):
        typer.echo("提示：--backend fake 仅用于测试和演示，不代表真实 CPICANN 预测。")


def _status_from_counts(counts: dict[str, int]) -> str:
    if counts["failed"] == 0 and counts["ignored"] == 0:
        return "ok"
    if counts["success"] > 0:
        return "partial_success"
    return "failed"


def _exit_for_counts(counts: dict[str, int]) -> None:
    status = _status_from_counts(counts)
    if status == "ok":
        return
    if status == "partial_success":
        raise typer.Exit(code=PARTIAL_SUCCESS_EXIT_CODE)
    raise typer.Exit(code=1)


def main() -> None:
    """Console script entry point."""
    app(args=_normalize_cli_args(sys.argv[1:]))


def _normalize_cli_args(args: list[str]) -> list[str]:
    """Support plan-style multi-value element options in console entrypoints."""
    multi_value_options = {"--include-must", "--allowed-elements"}
    normalized: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg not in multi_value_options:
            normalized.append(arg)
            index += 1
            continue
        option = arg
        index += 1
        values: list[str] = []
        while index < len(args) and not args[index].startswith("-"):
            values.append(args[index])
            index += 1
        if not values:
            normalized.append(option)
        for value in values:
            normalized.extend([option, value])
    return normalized
