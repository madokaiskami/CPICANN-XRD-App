"""FastAPI application for CPICANN-XRD."""

from __future__ import annotations

import json
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from cpicann_xrd.api.service import api_run_root, resolve_run_dir, stage_api_uploads
from cpicann_xrd.catalog.catalog import PhaseCatalog, load_catalog_from_manifest
from cpicann_xrd.core.spectrum_io import read_spectrum_file
from cpicann_xrd.decomposition.capabilities import Capabilities, build_capabilities
from cpicann_xrd.decomposition.exceptions import DecompositionError
from cpicann_xrd.decomposition.orchestration import (
    MultiphaseIdentificationResult,
    identify_decomposed_components,
)
from cpicann_xrd.decomposition.schemas import XDecomposerRequest, XDecomposerResult
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend
from cpicann_xrd.exceptions import CpicannXrdError
from cpicann_xrd.model.protocol import InferenceBackend
from cpicann_xrd.schemas import DiagnosticRecord, FilterSpec, ModelInfo, SamplePrediction
from cpicann_xrd.services.batch_runner import BatchRunResult, run_batch
from cpicann_xrd.services.runtime import DEFAULT_CATALOG_MANIFEST, build_runtime
from cpicann_xrd.version import __version__

API_TIMEOUT_SECONDS = 120

app = FastAPI(
    title="CPICANN-XRD API",
    version=__version__,
    description="Synchronous local API for small CPICANN-XRD prediction batches.",
)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach a stable request id to every API request and response."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    request_id: str


class ReadyResponse(BaseModel):
    status: Literal["ok"]
    request_id: str
    backends: dict[str, bool]


class ModelEntry(BaseModel):
    backend: Literal["fake", "cpicann"]
    available: bool
    model: ModelInfo | None = None
    message: str


class ModelsResponse(BaseModel):
    request_id: str
    models: list[ModelEntry]


class RunArtifacts(BaseModel):
    summary_csv: bool
    summary_report_md: bool
    result_bundle_zip: bool
    diagnostics_csv: bool


class RunResponse(BaseModel):
    request_id: str
    run_id: str
    status: Literal["ok", "partial_success", "failed", "available"]
    counts: dict[str, int]
    predictions: list[SamplePrediction] = Field(default_factory=list)
    diagnostics: list[DiagnosticRecord] = Field(default_factory=list)
    artifacts: RunArtifacts
    timeout_seconds: int = API_TIMEOUT_SECONDS
    note: str


class RunLookupResponse(BaseModel):
    request_id: str
    run_id: str
    status: Literal["available"]
    counts: dict[str, int]
    artifacts: RunArtifacts


@app.exception_handler(CpicannXrdError)
async def cpicann_error_handler(request: Request, exc: CpicannXrdError) -> JSONResponse:
    return _error_response(
        request,
        status_code=status.HTTP_400_BAD_REQUEST,
        code=exc.error_code.value,
        message=exc.message,
        details=exc.details,
    )


@app.exception_handler(DecompositionError)
async def decomposition_error_handler(request: Request, exc: DecompositionError) -> JSONResponse:
    return _error_response(
        request,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code=exc.code.value,
        message=exc.message,
        details=exc.details,
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    message = str(exc)
    status_code = (
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        if "超过大小限制" in message
        else status.HTTP_400_BAD_REQUEST
    )
    return _error_response(
        request,
        status_code=status_code,
        code="INVALID_REQUEST",
        message=message,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="请求参数无效",
        details={"errors": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return _error_response(
        request,
        status_code=exc.status_code,
        code="NOT_FOUND" if exc.status_code == status.HTTP_404_NOT_FOUND else "HTTP_ERROR",
        message=detail,
    )


@app.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    """Return process liveness."""
    return HealthResponse(status="ok", request_id=_request_id(request))


@app.get("/readyz", response_model=ReadyResponse)
def readyz(request: Request) -> ReadyResponse:
    """Return readiness for configured synchronous backends."""
    backends = {"fake": _backend_available("fake"), "cpicann": _backend_available("cpicann")}
    if not any(backends.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no backend available",
        )
    return ReadyResponse(status="ok", request_id=_request_id(request), backends=backends)


@app.get("/capabilities", response_model=Capabilities)
def capabilities() -> Capabilities:
    """Return CPICANN and optional XDecomposer capability state."""
    return build_capabilities(cpicann_available=_backend_available("cpicann"))


@app.get("/v1/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    """List supported backends and model metadata."""
    entries: list[ModelEntry] = []
    for backend_name in ("fake", "cpicann"):
        try:
            backend, _ = _cached_runtime(backend_name)
        except Exception:
            entries.append(
                ModelEntry(
                    backend=backend_name,
                    available=False,
                    model=None,
                    message="unavailable",
                )
            )
            continue
        entries.append(
            ModelEntry(
                backend=backend_name,
                available=True,
                model=backend.model_info,
                message="available",
            )
        )
    return ModelsResponse(request_id=_request_id(request), models=entries)


@app.post("/v1/predict", response_model=RunResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Single .txt/.csv/.xy spectrum file.")],
    backend: Annotated[str, Form(description="Backend: fake or cpicann.")] = "fake",
    include_must: Annotated[list[str] | None, Form()] = None,
    allowed_elements: Annotated[list[str] | None, Form()] = None,
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
) -> RunResponse:
    """Run one synchronous prediction and write a standard result bundle."""
    return await _run_uploaded_batch(
        request=request,
        uploads=[file],
        backend_name=backend,
        include_must=include_must,
        allowed_elements=allowed_elements,
        top_k=top_k,
    )


@app.post("/v1/batch", response_model=RunResponse)
async def batch(
    request: Request,
    files: Annotated[list[UploadFile], File(description="One or more spectrum files.")],
    backend: Annotated[str, Form(description="Backend: fake or cpicann.")] = "fake",
    include_must: Annotated[list[str] | None, Form()] = None,
    allowed_elements: Annotated[list[str] | None, Form()] = None,
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
) -> RunResponse:
    """Run a small synchronous batch and write a standard result bundle."""
    return await _run_uploaded_batch(
        request=request,
        uploads=files,
        backend_name=backend,
        include_must=include_must,
        allowed_elements=allowed_elements,
        top_k=top_k,
    )


@app.post("/v1/decompose", response_model=XDecomposerResult)
async def decompose(
    file: Annotated[UploadFile, File(description="Single .txt/.csv/.xy spectrum file.")],
    xdecomposer_backend: Annotated[
        str,
        Form(description="XDecomposer backend: disabled or stub."),
    ] = "disabled",
    max_sources: Annotated[int, Form(ge=1, le=16)] = 4,
    activity_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
    reference_top_k: Annotated[int, Form(ge=0, le=100)] = 5,
    return_component_patterns: Annotated[bool, Form()] = False,
) -> XDecomposerResult:
    """Explicit optional decomposition endpoint. Disabled unless a backend is selected."""
    backend = _decomposition_backend_or_503(xdecomposer_backend)
    spectrum = await _read_one_upload(file)
    return backend.decompose(
        XDecomposerRequest(
            sample_id=spectrum.sample_id,
            source_filename=spectrum.source_filename,
            two_theta=spectrum.two_theta,
            intensity=spectrum.intensity,
            max_sources=max_sources,
            activity_threshold=activity_threshold,
            reference_top_k=reference_top_k,
            return_component_patterns=return_component_patterns,
        )
    )


@app.post("/v1/decompose-and-identify", response_model=MultiphaseIdentificationResult)
async def decompose_and_identify(
    file: Annotated[UploadFile, File(description="Single .txt/.csv/.xy spectrum file.")],
    xdecomposer_backend: Annotated[
        str,
        Form(description="XDecomposer backend: disabled or stub."),
    ] = "disabled",
    backend: Annotated[str, Form(description="CPICANN backend: fake or cpicann.")] = "fake",
    include_must: Annotated[list[str] | None, Form()] = None,
    allowed_elements: Annotated[list[str] | None, Form()] = None,
    max_sources: Annotated[int, Form(ge=1, le=16)] = 4,
    activity_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
) -> MultiphaseIdentificationResult:
    """Explicit optional decomposition plus CPICANN ranking endpoint."""
    decomposition_backend = _decomposition_backend_or_503(xdecomposer_backend)
    spectrum = await _read_one_upload(file)
    phase_backend, catalog = _cached_runtime(backend)
    _, catalog_manifest = load_catalog_from_manifest(DEFAULT_CATALOG_MANIFEST)
    return identify_decomposed_components(
        request=XDecomposerRequest(
            sample_id=spectrum.sample_id,
            source_filename=spectrum.source_filename,
            two_theta=spectrum.two_theta,
            intensity=spectrum.intensity,
            max_sources=max_sources,
            activity_threshold=activity_threshold,
        ),
        decomposition_backend=decomposition_backend,
        phase_backend=phase_backend,
        catalog=catalog,
        catalog_manifest=catalog_manifest,
        top_k=top_k,
        filter_spec=FilterSpec(
            include_must=frozenset(include_must or []),
            allowed_elements=None if allowed_elements is None else frozenset(allowed_elements),
        ),
    )


@app.get("/v1/runs/{run_id}", response_model=RunLookupResponse)
def get_run(request: Request, run_id: str) -> RunLookupResponse:
    """Return metadata for one API-created run without exposing filesystem paths."""
    run_dir = _existing_run_dir(run_id)
    return RunLookupResponse(
        request_id=_request_id(request),
        run_id=run_id,
        status="available",
        counts=_counts_from_metadata(run_dir),
        artifacts=_artifacts(run_dir),
    )


@app.get("/v1/runs/{run_id}/download")
def download_run(run_id: str) -> FileResponse:
    """Download the standard ZIP bundle for one API-created run."""
    run_dir = _existing_run_dir(run_id)
    zip_path = run_dir / "result_bundle.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run bundle not found")
    return FileResponse(zip_path, media_type="application/zip", filename=f"{run_id}.zip")


async def _run_uploaded_batch(
    *,
    request: Request,
    uploads: list[UploadFile],
    backend_name: str,
    include_must: list[str] | None,
    allowed_elements: list[str] | None,
    top_k: int,
) -> RunResponse:
    backend, catalog = _cached_runtime(backend_name)
    run_root = api_run_root()
    with tempfile.TemporaryDirectory(prefix="cpicann-api-inputs-") as temp_dir:
        prepared = await stage_api_uploads(uploads, input_dir=Path(temp_dir) / "inputs")
        result = run_batch(
            input_paths=prepared.input_paths,
            output_root=run_root,
            backend=backend,
            catalog=catalog,
            filter_spec=FilterSpec(
                include_must=frozenset(include_must or []),
                allowed_elements=None if allowed_elements is None else frozenset(allowed_elements),
            ),
            top_k=top_k,
        )
    return _run_response(request, result)


@lru_cache(maxsize=2)
def _cached_runtime(backend_name: str) -> tuple[InferenceBackend, PhaseCatalog]:
    return build_runtime(backend_name)


def _backend_available(backend_name: str) -> bool:
    try:
        _cached_runtime(backend_name)
    except Exception:
        return False
    return True


def _decomposition_backend_or_503(backend_name: str) -> StubDecompositionBackend:
    capabilities = build_capabilities(
        cpicann_available=_backend_available("cpicann"),
        xdecomposer_backend=backend_name,
    )
    if not capabilities.xdecomposer.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"XDecomposer unavailable: {capabilities.xdecomposer.reason}",
        )
    if backend_name.strip().lower() in {"stub", "fake"}:
        return StubDecompositionBackend()
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"XDecomposer unavailable: {capabilities.xdecomposer.reason}",
    )


async def _read_one_upload(file: UploadFile) -> Any:
    with tempfile.TemporaryDirectory(prefix="cpicann-api-decomposition-") as temp_dir:
        prepared = await stage_api_uploads([file], input_dir=Path(temp_dir) / "inputs")
        read_result = read_spectrum_file(prepared.input_paths[0])
    if read_result.spectrum is None:
        message = "; ".join(diagnostic.message for diagnostic in read_result.diagnostics)
        raise ValueError(message or "failed to read spectrum")
    return read_result.spectrum


def _run_response(request: Request, result: BatchRunResult) -> RunResponse:
    return RunResponse(
        request_id=_request_id(request),
        run_id=result.run_id,
        status=_status_from_counts(result.counts),
        counts=result.counts,
        predictions=result.predictions,
        diagnostics=list(_non_success_diagnostics(result)),
        artifacts=_artifacts(result.run_dir),
        note="同步接口面向小批量请求；建议客户端超时不低于 120 秒。",
    )


def _status_from_counts(counts: dict[str, int]) -> Literal["ok", "partial_success", "failed"]:
    if counts["failed"] == 0 and counts["ignored"] == 0:
        return "ok"
    if counts["success"] > 0:
        return "partial_success"
    return "failed"


def _non_success_diagnostics(result: BatchRunResult) -> list[DiagnosticRecord]:
    return [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.status in {"failed", "ignored"}
    ]


def _artifacts(run_dir: Path) -> RunArtifacts:
    return RunArtifacts(
        summary_csv=(run_dir / "summary.csv").exists(),
        summary_report_md=(run_dir / "summary_report.md").exists(),
        result_bundle_zip=(run_dir / "result_bundle.zip").exists(),
        diagnostics_csv=(run_dir / "diagnostics.csv").exists(),
    )


def _existing_run_dir(run_id: str) -> Path:
    run_dir = resolve_run_dir(run_id)
    if not run_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return run_dir


def _counts_from_metadata(run_dir: Path) -> dict[str, int]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {"success": 0, "failed": 0, "ignored": 0}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime = metadata.get("runtime", {})
    return {
        "success": int(runtime.get("success_count", 0)),
        "failed": int(runtime.get("failed_count", 0)),
        "ignored": int(runtime.get("ignored_count", 0)),
    }


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    payload = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=request_id,
            details=details or {},
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))
