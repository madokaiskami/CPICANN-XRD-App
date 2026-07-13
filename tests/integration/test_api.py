from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import Request

from cpicann_xrd.api import main
from cpicann_xrd.api import service as api_service


def test_openapi_schema_contains_phase9_routes() -> None:
    schema = main.app.openapi()

    assert schema["info"]["title"] == "CPICANN-XRD API"
    assert "/healthz" in schema["paths"]
    assert "/readyz" in schema["paths"]
    assert "/v1/models" in schema["paths"]
    assert "/v1/predict" in schema["paths"]
    assert "/v1/batch" in schema["paths"]
    assert "/v1/runs/{run_id}" in schema["paths"]
    assert "/v1/runs/{run_id}/download" in schema["paths"]


def test_healthz_and_models_do_not_expose_paths() -> None:
    request = _request("req-test-001")

    health = main.healthz(request)
    models = main.models(request)

    assert health.status == "ok"
    assert health.request_id == "req-test-001"
    fake = next(item for item in models.models if item.backend == "fake")
    assert fake.available is True
    assert fake.model is not None
    assert "/home/" not in models.model_dump_json()


def test_predict_fake_writes_run_and_supports_download(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _set_api_run_root(tmp_path, monkeypatch)

    result = asyncio.run(
        main._run_uploaded_batch(
            request=_request("req-predict-001"),
            uploads=[_upload("0-norm.txt", Path("examples/spectra/0-norm.txt").read_bytes())],
            backend_name="fake",
            include_must=["Zr", "O"],
            allowed_elements=["Li", "Zr", "O"],
            top_k=5,
        )
    )

    assert result.status == "ok"
    assert result.counts == {"success": 1, "failed": 0, "ignored": 0}
    assert result.predictions[0].sample_id == "0-norm"
    assert result.predictions[0].predictions[0].phase.space_group == "P4_2/nmc"
    assert result.artifacts.summary_csv is True
    assert "/tmp/" not in result.model_dump_json()

    lookup = main.get_run(_request("req-lookup-001"), result.run_id)
    assert lookup.counts["success"] == 1
    download = main.download_run(result.run_id)
    assert Path(str(download.path)).read_bytes().startswith(b"PK")


def test_batch_fake_records_unsupported_files(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _set_api_run_root(tmp_path, monkeypatch)

    result = asyncio.run(
        main._run_uploaded_batch(
            request=_request("req-batch-001"),
            uploads=[
                _upload("0-norm.txt", Path("examples/spectra/0-norm.txt").read_bytes()),
                _upload("1-norm.txt", Path("examples/spectra/1-norm.txt").read_bytes()),
                _upload("3-norm.txt", Path("examples/spectra/3-norm.txt").read_bytes()),
                _upload("observed.png", b"not a spectrum"),
            ],
            backend_name="fake",
            include_must=None,
            allowed_elements=["Li", "Zr", "O"],
            top_k=5,
        )
    )

    assert result.status == "partial_success"
    assert result.counts == {"success": 3, "failed": 0, "ignored": 1}
    assert any(
        diagnostic.source_filename == "observed.png"
        and diagnostic.error_code is not None
        and diagnostic.error_code.value == "UNSUPPORTED_EXTENSION"
        for diagnostic in result.diagnostics
    )


def test_api_error_payload_is_stable_and_path_safe() -> None:
    response = asyncio.run(
        main.value_error_handler(
            _request("req-error-001"),
            ValueError("backend must be fake or cpicann"),
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 400
    assert payload["error"]["request_id"] == "req-error-001"
    assert payload["error"]["code"] == "INVALID_REQUEST"
    assert "Traceback" not in response.body.decode()
    assert "/home/" not in response.body.decode()


def test_upload_size_and_run_id_controls(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _set_api_run_root(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="超过大小限制"):
        asyncio.run(
            api_service.stage_api_uploads(
                [_upload("../sample.txt", b"12345")],
                input_dir=tmp_path / "inputs",
                max_bytes=4,
            )
        )

    prepared = asyncio.run(
        api_service.stage_api_uploads(
            [_upload("../sample.txt", b"10 1\n"), _upload("sample.txt", b"10 2\n")],
            input_dir=tmp_path / "inputs-ok",
        )
    )
    assert [path.name for path in prepared.input_paths] == ["sample.txt", "sample_2.txt"]


def test_run_id_traversal_is_rejected(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _set_api_run_root(tmp_path, monkeypatch)

    response = asyncio.run(
        main.value_error_handler(_request("req-run-001"), ValueError("invalid run_id"))
    )

    assert response.status_code == 400
    assert json.loads(response.body)["error"]["code"] == "INVALID_REQUEST"


def _request(request_id: str) -> Request:
    return cast(Request, SimpleNamespace(state=SimpleNamespace(request_id=request_id)))


@dataclass
class FakeUpload:
    filename: str
    data: bytes

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.data
        return self.data[:size]


def _upload(name: str, data: bytes) -> Any:
    return FakeUpload(filename=name, data=data)


def _set_api_run_root(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(api_service, "DEFAULT_API_RUN_ROOT", tmp_path / "api-runs")
    main._cached_runtime.cache_clear()
