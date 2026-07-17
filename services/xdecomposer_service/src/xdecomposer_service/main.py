"""FastAPI entry point for the isolated XDecomposer service."""

from __future__ import annotations

from fastapi import FastAPI

from xdecomposer_service.health import build_health, build_info, build_ready
from xdecomposer_service.schemas import HealthResponse, InfoResponse, ReadyResponse
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
