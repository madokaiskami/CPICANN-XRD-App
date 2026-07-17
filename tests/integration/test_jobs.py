"""Asynchronous API job tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi import Request

from cpicann_xrd.api import main
from cpicann_xrd.api import service as api_service
from cpicann_xrd.api.jobs import (
    InMemoryJobStore,
    JobCreateRequest,
    JobExecutionContext,
    JobState,
    wait_for_terminal,
)


def test_openapi_schema_contains_job_routes() -> None:
    schema = main.app.openapi()

    assert "/v1/jobs" in schema["paths"]
    assert "/v1/jobs/{job_id}" in schema["paths"]
    assert "/v1/jobs/{job_id}/result" in schema["paths"]
    assert "/v1/jobs/{job_id}/download" in schema["paths"]
    assert "/v1/jobs/{job_id}/cancel" in schema["paths"]
    assert "/metrics" in schema["paths"]


def test_api_capabilities_job_lifecycle(tmp_path: Path, monkeypatch: Any) -> None:
    _set_api_run_root(tmp_path, monkeypatch)
    snapshot = main.create_job(_request("req-job-001"), JobCreateRequest())

    terminal = wait_for_terminal(main._job_store(), snapshot.job_id)
    result = main.get_job_result(snapshot.job_id)
    download = main.download_job(snapshot.job_id)
    metrics = main.metrics()

    assert snapshot.status == JobState.QUEUED
    assert terminal.status == JobState.SUCCEEDED
    assert result.status == JobState.SUCCEEDED
    assert result.result["capabilities"]["xdecomposer"]["reason"] == "xdecomposer_disabled"
    assert Path(str(download.path)).read_bytes().startswith(b"PK")
    assert 'cpicann_xrd_jobs{status="succeeded"}' in metrics.body.decode()


def test_job_queue_respects_max_concurrency(tmp_path: Path) -> None:
    store = InMemoryJobStore(root=tmp_path / "jobs", max_concurrent=1)
    started = threading.Event()
    release = threading.Event()

    first = store.submit(
        mode="blocking",
        request_id="req-1",
        runner=lambda context: _blocking_runner(context, started, release),
    )
    assert started.wait(timeout=2.0)

    second = store.submit(
        mode="quick",
        request_id="req-2",
        runner=lambda context: {"job_id": context.job_id},
    )

    assert store.get(first.job_id).status in {JobState.VALIDATING, JobState.REPORTING}
    assert store.get(second.job_id).status == JobState.QUEUED

    release.set()
    assert wait_for_terminal(store, first.job_id).status == JobState.SUCCEEDED
    assert wait_for_terminal(store, second.job_id).status == JobState.SUCCEEDED
    store.shutdown()


def test_job_cancel_sets_terminal_state(tmp_path: Path) -> None:
    store = InMemoryJobStore(root=tmp_path / "jobs", max_concurrent=1)
    started = threading.Event()

    snapshot = store.submit(
        mode="cancellable",
        request_id="req-cancel",
        runner=lambda context: _cancellable_runner(context, started),
    )
    assert started.wait(timeout=2.0)

    cancelled = store.cancel(snapshot.job_id)
    terminal = wait_for_terminal(store, snapshot.job_id)

    assert cancelled.status == JobState.CANCELLED
    assert terminal.status == JobState.CANCELLED
    store.shutdown()


def _blocking_runner(
    context: JobExecutionContext,
    started: threading.Event,
    release: threading.Event,
) -> dict[str, str]:
    context.set_status(JobState.REPORTING)
    started.set()
    release.wait(timeout=2.0)
    return {"job_id": context.job_id}


def _cancellable_runner(
    context: JobExecutionContext,
    started: threading.Event,
) -> dict[str, str]:
    context.set_status(JobState.PREPROCESSING)
    started.set()
    while not context.cancel_event.is_set():
        time.sleep(0.01)
    return {"cancelled": "true"}


def _set_api_run_root(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(api_service, "DEFAULT_API_RUN_ROOT", tmp_path / "api-runs")
    main._job_store.cache_clear()


def _request(request_id: str) -> Request:
    return cast(Request, SimpleNamespace(state=SimpleNamespace(request_id=request_id)))
