"""API tests for explicit optional decomposition endpoints."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException, Request, UploadFile

from cpicann_xrd.api import main


def test_openapi_schema_contains_decomposition_routes() -> None:
    schema = main.app.openapi()

    assert "/capabilities" in schema["paths"]
    assert "/v1/decompose" in schema["paths"]
    assert "/v1/decompose-and-identify" in schema["paths"]


def test_api_capabilities_default_xdecomposer_disabled() -> None:
    capabilities = main.capabilities()

    assert capabilities.cpicann.enabled is True
    assert capabilities.xdecomposer.enabled is False
    assert capabilities.xdecomposer.available is False
    assert capabilities.xdecomposer.reason == "xdecomposer_disabled"


def test_api_decompose_disabled_maps_to_stable_error() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            main.decompose(
                file=cast(UploadFile, _upload("0-norm.txt")),
                xdecomposer_backend="disabled",
            )
        )

    response = asyncio.run(main.http_error_handler(_request("req-xd-disabled"), exc_info.value))
    payload = json.loads(response.body)
    assert response.status_code == 503
    assert payload["error"]["request_id"] == "req-xd-disabled"
    assert payload["error"]["code"] == "HTTP_ERROR"
    assert "xdecomposer_disabled" in payload["error"]["message"]
    assert "Traceback" not in response.body.decode()


def test_api_decompose_stub_returns_decomposition() -> None:
    result = asyncio.run(
        main.decompose(
            file=cast(UploadFile, _upload("0-norm.txt")),
            xdecomposer_backend="stub",
            return_component_patterns=True,
        )
    )

    assert result.model_id == "stub-xdecomposer-v1"
    assert result.preprocessing_version == "xdecomposer-v1"
    assert [component.component_index for component in result.components] == [1, 2]
    assert all(component.pattern is not None for component in result.components)


def test_api_decompose_and_identify_stub_uses_orchestration_service() -> None:
    result = asyncio.run(
        main.decompose_and_identify(
            file=cast(UploadFile, _upload("0-norm.txt")),
            xdecomposer_backend="stub",
            backend="fake",
            allowed_elements=["Li", "O", "Zr"],
            top_k=2,
        )
    )

    assert result.mode == "multiphase-identification"
    assert result.status == "success"
    assert len(result.components) == 2
    assert all(component.cpicann is not None for component in result.components)
    assert all(component.cpicann.returned_top_k == 2 for component in result.components)


@dataclass
class FakeUpload:
    filename: str
    data: bytes

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.data
        return self.data[:size]


def _upload(name: str) -> FakeUpload:
    return FakeUpload(filename=name, data=(Path("examples/spectra") / name).read_bytes())


def _request(request_id: str) -> Request:
    return cast(Request, SimpleNamespace(state=SimpleNamespace(request_id=request_id)))
