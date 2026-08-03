"""Streamlit web application for local CPICANN-XRD use."""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import streamlit as st

from cpicann_xrd.catalog.catalog import PhaseCatalog
from cpicann_xrd.core.spectrum_io import read_spectrum_file
from cpicann_xrd.decomposition.artifacts import (
    DecompositionArtifactPaths,
    write_decomposition_artifacts,
)
from cpicann_xrd.decomposition.capabilities import build_capabilities
from cpicann_xrd.decomposition.client import XDecomposerHttpClient
from cpicann_xrd.decomposition.exceptions import DecompositionError
from cpicann_xrd.decomposition.orchestration import (
    MultiphaseIdentificationResult,
    identify_decomposed_components,
)
from cpicann_xrd.decomposition.schemas import XDecomposerRequest, XDecomposerResult
from cpicann_xrd.exceptions import CpicannXrdError
from cpicann_xrd.model.protocol import InferenceBackend
from cpicann_xrd.schemas import FilterSpec
from cpicann_xrd.services.batch_runner import BatchRunResult, run_batch
from cpicann_xrd.services.runtime import build_runtime
from cpicann_xrd.web.service import (
    UploadedFileLike,
    available_elements,
    component_pattern_csv,
    stage_uploaded_files,
    web_mode_options,
)

WEB_BACKEND_NAME = "cpicann"


@dataclass(frozen=True)
class WebDecompositionRun:
    """One Web decomposition result with downloadable artifacts."""

    result: XDecomposerResult
    artifacts: DecompositionArtifactPaths | None


@dataclass(frozen=True)
class WebIdentificationRun:
    """One Web decomposition plus CPICANN identification result."""

    result: MultiphaseIdentificationResult


def main() -> None:
    """Render and run the Streamlit application."""
    st.set_page_config(page_title="CPICANN-XRD", layout="wide")
    st.title("CPICANN-XRD")
    st.caption("单相 CPICANN 候选物相排序工具")

    with st.sidebar:
        backend_name = WEB_BACKEND_NAME
        model_status = _model_status(backend_name)
        capabilities = build_capabilities(cpicann_available=_backend_available(backend_name))
        mode_options = web_mode_options(capabilities)
        enabled_mode_options = [option for option in mode_options if option.enabled]
        mode_label = st.radio(
            "模式",
            [option.label for option in enabled_mode_options],
            index=0,
        )
        selected_mode = next(
            option for option in enabled_mode_options if option.label == mode_label
        )
        for option in mode_options:
            if not option.enabled:
                st.caption(f"{option.label}：{option.reason}")
        st.write(f"模型状态：{model_status}")
        elements = available_elements()
        include_must = st.multiselect("必须包含元素", elements, default=[])
        allowed_elements = st.multiselect("允许元素范围", elements, default=[])
        top_k = st.number_input("Top-K", min_value=1, max_value=50, value=5, step=1)
        max_sources = st.number_input(
            "XDecomposer components", min_value=1, max_value=16, value=4, step=1
        )
        activity_threshold = st.slider(
            "Activity threshold", min_value=0.0, max_value=1.0, value=0.5
        )

    uploaded_files = st.file_uploader(
        "上传谱图文件",
        accept_multiple_files=True,
        help="支持 .txt、.csv、.xy；其他文件会记录为不支持输入。",
    )
    st.info("样品名称来自文件名；模型预测物相是单相候选排序。过滤后条件置信度不是实际多相含量。")
    if not uploaded_files:
        st.stop()

    if selected_mode.key == "decompose":
        if st.button("运行多相分解", type="primary"):
            try:
                decomposition_runs = _run_decomposition_files(
                    uploaded_files=cast("list[UploadedFileLike]", list(uploaded_files)),
                    max_sources=int(max_sources),
                    activity_threshold=float(activity_threshold),
                )
            except (DecompositionError, CpicannXrdError, ValueError) as exc:
                st.error(str(exc))
                return
            st.session_state["web_decomposition_runs"] = decomposition_runs
            st.session_state.pop("web_identification_runs", None)
            _render_decomposition_results(decomposition_runs)
        elif "web_decomposition_results" in st.session_state:
            # Backward compatibility for sessions opened before this deployment.
            _render_decomposition_results(
                [
                    WebDecompositionRun(result=result, artifacts=None)
                    for result in cast(
                        "list[XDecomposerResult]", st.session_state["web_decomposition_results"]
                    )
                ]
            )
        elif "web_decomposition_runs" in st.session_state:
            _render_decomposition_results(
                cast("list[WebDecompositionRun]", st.session_state["web_decomposition_runs"])
            )
        return

    if selected_mode.key == "decompose_and_identify":
        if st.button("运行多相分解并识别", type="primary"):
            try:
                identification_runs = _run_identification_files(
                    uploaded_files=cast("list[UploadedFileLike]", list(uploaded_files)),
                    max_sources=int(max_sources),
                    activity_threshold=float(activity_threshold),
                    backend_name=backend_name,
                    include_must=include_must,
                    allowed_elements=allowed_elements or None,
                    top_k=int(top_k),
                )
            except (DecompositionError, CpicannXrdError, ValueError) as exc:
                st.error(str(exc))
                return
            st.session_state["web_identification_runs"] = identification_runs
            st.session_state.pop("web_decomposition_runs", None)
            _render_identification_results(identification_runs)
        elif "web_identification_runs" in st.session_state:
            _render_identification_results(
                cast("list[WebIdentificationRun]", st.session_state["web_identification_runs"])
            )
        return

    if st.button("运行识别", type="primary"):
        try:
            result = _run_uploaded_files(
                uploaded_files=cast("list[UploadedFileLike]", list(uploaded_files)),
                backend_name=backend_name,
                include_must=include_must,
                allowed_elements=allowed_elements or None,
                top_k=int(top_k),
            )
        except (CpicannXrdError, ValueError) as exc:
            st.error(str(exc))
            return
        _render_result(result)
    elif "web_result" in st.session_state:
        _render_result(cast("BatchRunResult", st.session_state["web_result"]))


@st.cache_resource(show_spinner=False)
def _cached_runtime(backend_name: str) -> tuple[InferenceBackend, PhaseCatalog]:
    return build_runtime(backend_name)


def _model_status(backend_name: str) -> str:
    try:
        backend, _ = _cached_runtime(backend_name)
    except Exception:
        return "真实模型未就绪"
    return f"{backend.model_info.model_id} 可用"


def _backend_available(backend_name: str) -> bool:
    try:
        _cached_runtime(backend_name)
    except Exception:
        return False
    return True


def _run_uploaded_files(
    *,
    uploaded_files: list[UploadedFileLike],
    backend_name: str,
    include_must: list[str],
    allowed_elements: list[str] | None,
    top_k: int,
) -> BatchRunResult:
    backend, catalog = _cached_runtime(backend_name)
    previous_temp_dir = st.session_state.get("web_temp_dir")
    if previous_temp_dir:
        shutil.rmtree(previous_temp_dir, ignore_errors=True)
    temp_root = Path(tempfile.mkdtemp(prefix="cpicann-web-"))
    atexit.register(shutil.rmtree, temp_root, ignore_errors=True)
    st.session_state["web_temp_dir"] = str(temp_root)
    prepared = stage_uploaded_files(uploaded_files, input_dir=temp_root / "inputs")
    result = run_batch(
        input_paths=prepared.input_paths,
        output_root=temp_root / "runs",
        backend=backend,
        catalog=catalog,
        filter_spec=FilterSpec(
            include_must=frozenset(include_must),
            allowed_elements=None if allowed_elements is None else frozenset(allowed_elements),
        ),
        top_k=top_k,
    )
    stored_result = _snapshot_run_result(result)
    st.session_state["web_result"] = stored_result
    st.session_state["upload_warnings"] = prepared.warnings
    return stored_result


def _run_decomposition_files(
    *,
    uploaded_files: list[UploadedFileLike],
    max_sources: int,
    activity_threshold: float,
) -> list[WebDecompositionRun]:
    previous_temp_dir = st.session_state.get("web_temp_dir")
    if previous_temp_dir:
        shutil.rmtree(previous_temp_dir, ignore_errors=True)
    temp_root = Path(tempfile.mkdtemp(prefix="cpicann-web-xdecomposer-"))
    atexit.register(shutil.rmtree, temp_root, ignore_errors=True)
    st.session_state["web_temp_dir"] = str(temp_root)
    prepared = stage_uploaded_files(uploaded_files, input_dir=temp_root / "inputs")
    st.session_state["upload_warnings"] = prepared.warnings
    client = _xdecomposer_client()
    results: list[WebDecompositionRun] = []
    for input_path in prepared.input_paths:
        read_result = read_spectrum_file(input_path)
        if read_result.spectrum is None:
            message = "; ".join(diagnostic.message for diagnostic in read_result.diagnostics)
            raise ValueError(message or f"failed to read {input_path.name}")
        spectrum = read_result.spectrum
        request = XDecomposerRequest(
            sample_id=spectrum.sample_id,
            source_filename=spectrum.source_filename,
            two_theta=spectrum.two_theta,
            intensity=spectrum.intensity,
            max_sources=max_sources,
            activity_threshold=activity_threshold,
            return_component_patterns=True,
        )
        result = client.decompose(request)
        artifacts = write_decomposition_artifacts(
            run_dir=temp_root / "runs" / spectrum.sample_id,
            request=request,
            result=result,
            source_sha256=spectrum.sha256 or _sha256_file(input_path),
            source_filename=spectrum.source_filename,
            git_commit=os.environ.get("CPICANN_GIT_COMMIT", "UNKNOWN"),
            device=os.environ.get("XDECOMPOSER_DEVICE", "remote"),
        )
        results.append(WebDecompositionRun(result=result, artifacts=artifacts))
    return results


def _run_identification_files(
    *,
    uploaded_files: list[UploadedFileLike],
    max_sources: int,
    activity_threshold: float,
    backend_name: str,
    include_must: list[str],
    allowed_elements: list[str] | None,
    top_k: int,
) -> list[WebIdentificationRun]:
    backend, catalog = _cached_runtime(backend_name)
    previous_temp_dir = st.session_state.get("web_temp_dir")
    if previous_temp_dir:
        shutil.rmtree(previous_temp_dir, ignore_errors=True)
    temp_root = Path(tempfile.mkdtemp(prefix="cpicann-web-xdecomposer-identify-"))
    atexit.register(shutil.rmtree, temp_root, ignore_errors=True)
    st.session_state["web_temp_dir"] = str(temp_root)
    prepared = stage_uploaded_files(uploaded_files, input_dir=temp_root / "inputs")
    st.session_state["upload_warnings"] = prepared.warnings
    client = _xdecomposer_client()
    results: list[WebIdentificationRun] = []
    for input_path in prepared.input_paths:
        read_result = read_spectrum_file(input_path)
        if read_result.spectrum is None:
            message = "; ".join(diagnostic.message for diagnostic in read_result.diagnostics)
            raise ValueError(message or f"failed to read {input_path.name}")
        spectrum = read_result.spectrum
        result = identify_decomposed_components(
            request=XDecomposerRequest(
                sample_id=spectrum.sample_id,
                source_filename=spectrum.source_filename,
                two_theta=spectrum.two_theta,
                intensity=spectrum.intensity,
                max_sources=max_sources,
                activity_threshold=activity_threshold,
            ),
            decomposition_backend=client,
            phase_backend=backend,
            catalog=catalog,
            top_k=top_k,
            filter_spec=FilterSpec(
                include_must=frozenset(include_must),
                allowed_elements=None if allowed_elements is None else frozenset(allowed_elements),
            ),
        )
        results.append(WebIdentificationRun(result=result))
    return results


def _xdecomposer_client() -> XDecomposerHttpClient:
    return XDecomposerHttpClient(
        base_url=os.environ.get("CPICANN_XDECOMPOSER_SERVICE_URL", "http://127.0.0.1:8100"),
        timeout_seconds=float(os.environ.get("CPICANN_XDECOMPOSER_TIMEOUT_SECONDS", "300")),
        retries=int(os.environ.get("CPICANN_XDECOMPOSER_RETRIES", "1")),
    )


def _snapshot_run_result(result: BatchRunResult) -> BatchRunResult:
    return BatchRunResult(
        run_id=result.run_id,
        run_dir=result.run_dir,
        counts=result.counts,
        predictions=result.predictions,
        diagnostics=result.diagnostics,
    )


def _render_result(result: BatchRunResult) -> None:
    st.subheader("运行摘要")
    cols = st.columns(3)
    cols[0].metric("成功", result.counts["success"])
    cols[1].metric("失败", result.counts["failed"])
    cols[2].metric("忽略", result.counts["ignored"])

    for warning in st.session_state.get("upload_warnings", []):
        st.warning(warning)

    for prediction in result.predictions:
        st.markdown(f"### {prediction.sample_id} · {prediction.source_filename}")
        st.caption("下表为单相候选排序，不是多相含量拟合结果。")
        rows = [
            {
                "过滤后排序": item.filtered_rank,
                "全局排序": item.global_rank,
                "类别索引": item.class_index,
                "COD ID": item.phase.cod_id,
                "模型预测物相": item.phase.formula,
                "空间群": item.phase.space_group,
                "空间群编号": item.phase.space_group_number,
                "原始 logit": item.raw_logit,
                "过滤前全局概率": item.unfiltered_probability,
                "过滤后条件置信度": item.filtered_confidence,
            }
            for item in prediction.predictions
        ]
        st.markdown(_markdown_table(rows))
        png_path = result.run_dir / "samples" / prediction.sample_id / "observed_xrd.png"
        if png_path.exists():
            st.image(str(png_path), caption="observed XRD")

    diagnostics = [
        {
            "source_filename": diagnostic.source_filename,
            "status": diagnostic.status,
            "stage": diagnostic.stage,
            "error_code": diagnostic.error_code.value if diagnostic.error_code else "",
            "message": diagnostic.message,
        }
        for diagnostic in result.diagnostics
        if diagnostic.status in {"failed", "ignored"}
    ]
    if diagnostics:
        st.subheader("诊断提示")
        st.markdown(_markdown_table(diagnostics))

    st.subheader("下载")
    _download_file("summary.csv", result.run_dir / "summary.csv", "text/csv")
    _download_file("summary_report.md", result.run_dir / "summary_report.md", "text/markdown")
    _download_file("result_bundle.zip", result.run_dir / "result_bundle.zip", "application/zip")
    st.caption(
        "本结果为 CPICANN 单相分类模型在当前元素约束条件下生成的候选物相排序。"
        "过滤后置信度为候选集合内的相对条件分数，不代表实际物相含量。"
    )


def _render_decomposition_results(runs: list[WebDecompositionRun]) -> None:
    st.subheader("XDecomposer 多相分解")
    for warning in st.session_state.get("upload_warnings", []):
        st.warning(warning)
    for run in runs:
        result = run.result
        st.markdown(f"### {result.sample_id}")
        st.caption(
            "Component 是 XDecomposer 分解输出，不等同于已确认物相；"
            "estimated weight 不是 Rietveld 定量相含量。"
        )
        st.metric("Reconstruction error", f"{result.reconstruction_error:.6g}")
        rows = [
            {
                "component": component.component_index,
                "slot": component.original_slot_index,
                "active": component.is_active,
                "activity_probability": component.active_probability,
                "estimated_weight": component.estimated_weight,
                "pattern_sha256": component.pattern_sha256[:12],
            }
            for component in result.components
        ]
        st.markdown(_markdown_table(rows))
        st.markdown("#### 下载")
        if run.artifacts is not None:
            _download_file(
                f"{result.sample_id}_decomposition_summary.csv",
                run.artifacts.summary_csv,
                "text/csv",
            )
            _download_file(
                f"{result.sample_id}_decomposition_report.md",
                run.artifacts.report_md,
                "text/markdown",
            )
            _download_file(
                f"{result.sample_id}_result_bundle.zip",
                run.artifacts.bundle_zip,
                "application/zip",
            )
        for component in result.components:
            if component.pattern is None:
                continue
            slot = component.original_slot_index
            st.download_button(
                label=f"下载 slot {slot} CSV",
                data=component_pattern_csv(component, result.preprocessing),
                file_name=f"{result.sample_id}_slot_{slot}.csv",
                mime="text/csv",
            )


def _render_identification_results(runs: list[WebIdentificationRun]) -> None:
    st.subheader("XDecomposer 多相分解并识别")
    for warning in st.session_state.get("upload_warnings", []):
        st.warning(warning)
    for run in runs:
        result = run.result
        st.markdown(f"### {result.sample_id}")
        st.metric("成功识别 component", result.successful_component_count)
        st.metric("失败 component", result.failed_component_count)
        st.caption("每个 component 先由 XDecomposer 分解，再用 CPICANN 单相模型进行候选排序。")
        rows = []
        for component in result.components:
            top_prediction = None
            if component.cpicann is not None and component.cpicann.predictions:
                top_prediction = component.cpicann.predictions[0]
            rows.append(
                {
                    "component": component.component_index,
                    "slot": component.original_slot_index,
                    "status": component.status,
                    "active": component.is_active,
                    "estimated_weight": component.estimated_weight,
                    "top_formula": "" if top_prediction is None else top_prediction.phase.formula,
                    "space_group": ""
                    if top_prediction is None
                    else top_prediction.phase.space_group,
                    "confidence": ""
                    if top_prediction is None
                    else top_prediction.filtered_confidence,
                    "error": "" if component.error is None else component.error.message,
                }
            )
        st.markdown(_markdown_table(rows))
        st.download_button(
            label=f"下载 {result.sample_id} 识别报告",
            data=result.report_markdown.encode("utf-8"),
            file_name=f"{result.sample_id}_multiphase_identification.md",
            mime="text/markdown",
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_file(label: str, path: Path, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=f"下载 {label}",
            data=path.read_bytes(),
            file_name=label,
            mime=mime,
        )


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(_format_table_cell(row.get(header, "")) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def _format_table_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
