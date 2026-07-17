from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import pytest

from cpicann_xrd.decomposition.client import XDecomposerHttpClient
from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.schemas import XDecomposerRequest
from cpicann_xrd.decomposition.stub_backend import StubDecompositionBackend


def test_http_client_parses_worker_result() -> None:
    request = XDecomposerRequest(
        sample_id="sample-001",
        two_theta=[10.0, 45.0, 80.0],
        intensity=[0.0, 10.0, 0.0],
    )
    expected = StubDecompositionBackend().decompose(request).model_dump()
    client = XDecomposerHttpClient(base_url="http://worker", opener=_fake_opener(expected))

    result = client.decompose(request)

    assert result.sample_id == "sample-001"
    assert result.model_id == "stub-xdecomposer-v1"


def test_http_client_maps_connection_failure() -> None:
    request = XDecomposerRequest(
        sample_id="sample-001",
        two_theta=[10.0, 45.0, 80.0],
        intensity=[0.0, 10.0, 0.0],
    )

    def failing_opener(_request: urllib.request.Request, *, timeout: float) -> Any:
        raise urllib.error.URLError("connection refused")

    client = XDecomposerHttpClient(
        base_url="http://worker",
        timeout_seconds=0.1,
        opener=failing_opener,
    )

    with pytest.raises(DecompositionError) as exc_info:
        client.decompose(request)

    assert exc_info.value.code == DecompositionErrorCode.BACKEND_UNAVAILABLE


def test_http_client_rejects_invalid_worker_schema() -> None:
    request = XDecomposerRequest(
        sample_id="sample-001",
        two_theta=[10.0, 45.0, 80.0],
        intensity=[0.0, 10.0, 0.0],
    )
    client = XDecomposerHttpClient(
        base_url="http://worker", opener=_fake_opener({"unexpected": True})
    )

    with pytest.raises(DecompositionError) as exc_info:
        client.decompose(request)

    assert exc_info.value.code == DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH


def _fake_opener(payload: dict[str, Any]) -> Any:
    def opener(_request: urllib.request.Request, *, timeout: float) -> _FakeResponse:
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    return opener


@dataclass
class _FakeResponse:
    body: bytes

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body
