"""FastAPI entry point for the isolated XDecomposer service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

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
from xdecomposer_service.settings import Settings, load_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    preload_model_if_configured()
    yield


app = FastAPI(title="XDecomposer Runtime Service", version="0.1.0", lifespan=lifespan)


def preload_model_if_configured() -> None:
    """Optionally load the model once during container startup."""
    settings = load_settings()
    if not settings.preload_model:
        return
    try:
        _backend_for_settings(settings).warm_up()
    except XDecomposerBackendError:
        logger.exception("XDecomposer model preload failed")


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
        return _backend_for_settings(load_settings()).decompose(request)
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


@lru_cache(maxsize=4)
def _backend_for_settings(settings: Settings) -> RealXDecomposerBackend:
    return RealXDecomposerBackend(settings)
