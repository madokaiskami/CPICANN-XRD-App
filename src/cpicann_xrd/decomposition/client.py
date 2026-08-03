"""HTTP client for an isolated XDecomposer worker."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from cpicann_xrd.decomposition.exceptions import DecompositionError, DecompositionErrorCode
from cpicann_xrd.decomposition.schemas import XDecomposerRequest, XDecomposerResult


@dataclass(frozen=True)
class XDecomposerHttpClient:
    """Small stdlib HTTP client with timeout and retry controls."""

    base_url: str
    timeout_seconds: float = 300.0
    retries: int = 1
    opener: Any = urllib.request.urlopen

    def decompose(self, request: XDecomposerRequest) -> XDecomposerResult:
        """POST a decomposition request and parse the worker response."""
        payload = request.model_dump_json().encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/decompose"
        last_error: Exception | None = None
        for _attempt in range(self.retries + 1):
            try:
                http_request = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.opener(http_request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                raw = json.loads(body)
                return XDecomposerResult.model_validate(raw)
            except TimeoutError as exc:
                last_error = exc
                continue
            except urllib.error.URLError as exc:
                last_error = exc
                continue
            except json.JSONDecodeError as exc:
                raise DecompositionError(
                    DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
                    "XDecomposer worker 返回非 JSON 响应",
                    details={"url": url},
                ) from exc
            except ValidationError as exc:
                raise DecompositionError(
                    DecompositionErrorCode.OUTPUT_SHAPE_MISMATCH,
                    "XDecomposer worker 响应 schema 无效",
                    details={"url": url, "errors": exc.errors(include_url=False)},
                ) from exc

        code = _map_transport_error(last_error)
        raise DecompositionError(
            code,
            "XDecomposer worker 请求失败",
            details={
                "url": url,
                "timeout_seconds": self.timeout_seconds,
                "retries": self.retries,
                "reason": repr(last_error),
            },
        )


def _map_transport_error(error: Exception | None) -> DecompositionErrorCode:
    if isinstance(error, TimeoutError):
        return DecompositionErrorCode.INFERENCE_TIMEOUT
    if isinstance(error, urllib.error.URLError):
        reason: Any = getattr(error, "reason", None)
        if isinstance(reason, TimeoutError):
            return DecompositionErrorCode.INFERENCE_TIMEOUT
    return DecompositionErrorCode.BACKEND_UNAVAILABLE
