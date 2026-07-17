"""FastAPI entry point for the isolated XDecomposer service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from xdecomposer_service.backend import RealXDecomposerBackend, XDecomposerBackendError
from xdecomposer_service.health import build_health, build_info, build_ready
from xdecomposer_service.schemas import (
    HealthResponse,
    InfoResponse,
    ReadyResponse,
    XDecomposerRequest,
    XDecomposerResult,
)
from xdecomposer_service.settings import load_settings

app = FastAPI(title="XDecomposer Runtime Service", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Process liveness endpoint."""
    return build_health()


@app.get("/readyz", response_model=ReadyResponse)
def readyz() -> ReadyResponse:
    """Readiness endpoint."""
    return build_ready(load_settings())


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Runtime environment and asset information endpoint."""
    return build_info(load_settings())


@app.post("/decompose", response_model=XDecomposerResult)
def decompose(request: XDecomposerRequest) -> XDecomposerResult:
    """Run one real XDecomposer decomposition."""
    try:
        return RealXDecomposerBackend(load_settings()).decompose(request)
    except XDecomposerBackendError as exc:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
        if exc.code in {"manifest_invalid", "manifest_unreadable"}:
            code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if exc.code == "output_shape_mismatch":
            code = status.HTTP_502_BAD_GATEWAY
        raise HTTPException(
            status_code=code,
            detail={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_input", "message": str(exc), "details": {}},
        ) from exc
