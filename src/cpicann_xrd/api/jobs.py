"""In-process asynchronous job store for API task orchestration."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from cpicann_xrd.schemas import StrictBaseModel


class JobState(StrEnum):
    """Stable asynchronous job states."""

    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    DECOMPOSING = "decomposing"
    IDENTIFYING = "identifying"
    REPORTING = "reporting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class JobCreateRequest(StrictBaseModel):
    """Minimal job creation request for opt-in asynchronous API work."""

    mode: Literal["capabilities"] = "capabilities"
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=3600.0)


class JobAuditEntry(StrictBaseModel):
    """One structured job audit event."""

    at: datetime
    job_id: str
    request_id: str
    event: str
    status: JobState
    details: dict[str, str] = Field(default_factory=dict)


class JobSnapshot(StrictBaseModel):
    """Public job metadata snapshot."""

    job_id: str
    mode: str
    request_id: str
    status: JobState
    created_at: datetime
    updated_at: datetime
    timeout_seconds: float
    result_available: bool
    download_available: bool
    error: dict[str, str] | None = None
    audit: list[JobAuditEntry] = Field(default_factory=list)


class JobResultEnvelope(StrictBaseModel):
    """Stored JSON job result."""

    job_id: str
    status: JobState
    result: dict[str, Any]


@dataclass(frozen=True)
class JobExecutionContext:
    """Context passed to a job runner."""

    job_id: str
    request_id: str
    job_dir: Path
    cancel_event: threading.Event
    set_status: Callable[[JobState], None]


JobRunner = Callable[[JobExecutionContext], dict[str, Any]]


@dataclass
class _JobRecord:
    job_id: str
    mode: str
    request_id: str
    status: JobState
    created_at: datetime
    updated_at: datetime
    timeout_seconds: float
    job_dir: Path
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result_path: Path | None = None
    download_path: Path | None = None
    error: dict[str, str] | None = None
    audit: list[JobAuditEntry] = field(default_factory=list)
    future: Future[None] | None = None


class InMemoryJobStore:
    """Small thread-pool-backed job store suitable for local API tests."""

    def __init__(
        self,
        *,
        root: Path,
        max_concurrent: int = 1,
        timeout_seconds: float = 120.0,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.root = root
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.ttl_seconds = ttl_seconds
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        *,
        mode: str,
        request_id: str,
        runner: JobRunner,
        timeout_seconds: float | None = None,
    ) -> JobSnapshot:
        """Submit a job and return its initial queued snapshot."""
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        now = _now()
        job = _JobRecord(
            job_id=job_id,
            mode=mode,
            request_id=request_id,
            status=JobState.QUEUED,
            created_at=now,
            updated_at=now,
            timeout_seconds=timeout_seconds or self.timeout_seconds,
            job_dir=job_dir,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._audit(job, "submitted")
            job.future = self._executor.submit(self._run_job, job_id, runner)
            return self._snapshot(job)

    def get(self, job_id: str) -> JobSnapshot:
        """Return a job snapshot."""
        with self._lock:
            return self._snapshot(self._require_job(job_id))

    def cancel(self, job_id: str) -> JobSnapshot:
        """Request cancellation for a queued or running job."""
        with self._lock:
            job = self._require_job(job_id)
            job.cancel_event.set()
            if job.status not in TERMINAL_JOB_STATES:
                self._set_status_locked(job, JobState.CANCELLED, event="cancelled")
            return self._snapshot(job)

    def result(self, job_id: str) -> JobResultEnvelope:
        """Return stored JSON result for a succeeded job."""
        with self._lock:
            job = self._require_job(job_id)
            if job.result_path is None or not job.result_path.is_file():
                raise KeyError("job result is not available")
            payload = json.loads(job.result_path.read_text(encoding="utf-8"))
            return JobResultEnvelope.model_validate(payload)

    def download_path(self, job_id: str) -> Path:
        """Return a ZIP archive containing isolated job outputs."""
        with self._lock:
            job = self._require_job(job_id)
            if job.status != JobState.SUCCEEDED:
                raise KeyError("job download is not available")
            if job.download_path is None or not job.download_path.is_file():
                job.download_path = self._write_download_zip(job)
            return job.download_path

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Delete terminal jobs older than the configured TTL."""
        active_now = now or _now()
        removed = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status not in TERMINAL_JOB_STATES:
                    continue
                age = (active_now - job.updated_at).total_seconds()
                if age < self.ttl_seconds:
                    continue
                shutil.rmtree(job.job_dir, ignore_errors=True)
                del self._jobs[job_id]
                removed += 1
        return removed

    def metrics(self) -> dict[str, int]:
        """Return Prometheus-friendly status counts."""
        with self._lock:
            counts = {state.value: 0 for state in JobState}
            for job in self._jobs.values():
                counts[job.status.value] += 1
            counts["total"] = len(self._jobs)
            return counts

    def shutdown(self) -> None:
        """Stop the backing executor."""
        self._executor.shutdown(wait=True)

    def _run_job(self, job_id: str, runner: JobRunner) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_event.is_set():
                self._set_status_locked(job, JobState.CANCELLED, event="cancelled")
                return
            self._set_status_locked(job, JobState.VALIDATING)
        started = time.monotonic()
        context = JobExecutionContext(
            job_id=job_id,
            request_id=job.request_id,
            job_dir=job.job_dir,
            cancel_event=job.cancel_event,
            set_status=lambda status: self._set_status(job_id, status),
        )
        try:
            result = runner(context)
        except Exception as exc:  # noqa: BLE001 - jobs must persist structured failure state.
            with self._lock:
                active_job = self._jobs.get(job_id)
                if active_job is not None and active_job.status not in TERMINAL_JOB_STATES:
                    active_job.error = {"code": exc.__class__.__name__, "message": str(exc)}
                    self._set_status_locked(active_job, JobState.FAILED, event="failed")
            return

        elapsed = time.monotonic() - started
        with self._lock:
            active_job = self._jobs.get(job_id)
            if active_job is None:
                return
            if active_job.cancel_event.is_set():
                self._set_status_locked(active_job, JobState.CANCELLED, event="cancelled")
                return
            if elapsed > active_job.timeout_seconds:
                active_job.error = {
                    "code": "JOB_TIMEOUT",
                    "message": f"job exceeded timeout_seconds={active_job.timeout_seconds:g}",
                }
                self._set_status_locked(active_job, JobState.FAILED, event="timeout")
                return
            self._write_result(active_job, result)
            self._set_status_locked(active_job, JobState.SUCCEEDED, event="succeeded")

    def _set_status(self, job_id: str, status: JobState) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_JOB_STATES:
                return
            self._set_status_locked(job, status)

    def _set_status_locked(
        self,
        job: _JobRecord,
        status: JobState,
        *,
        event: str | None = None,
    ) -> None:
        job.status = status
        job.updated_at = _now()
        self._audit(job, event or f"status:{status.value}")

    def _audit(
        self,
        job: _JobRecord,
        event: str,
        *,
        details: dict[str, str] | None = None,
    ) -> None:
        entry = JobAuditEntry(
            at=_now(),
            job_id=job.job_id,
            request_id=job.request_id,
            event=event,
            status=job.status,
            details=details or {},
        )
        job.audit.append(entry)
        audit_path = job.job_dir / "audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")

    def _write_result(self, job: _JobRecord, result: dict[str, Any]) -> None:
        payload = JobResultEnvelope(job_id=job.job_id, status=JobState.SUCCEEDED, result=result)
        result_path = job.job_dir / "result.json"
        result_path.write_text(payload.model_dump_json(), encoding="utf-8")
        job.result_path = result_path

    def _write_download_zip(self, job: _JobRecord) -> Path:
        zip_path = job.job_dir / "job_result.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(job.job_dir.rglob("*")):
                if path == zip_path or not path.is_file():
                    continue
                archive.write(path, path.relative_to(job.job_dir))
        return zip_path

    def _require_job(self, job_id: str) -> _JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError("job not found") from exc

    def _snapshot(self, job: _JobRecord) -> JobSnapshot:
        return JobSnapshot(
            job_id=job.job_id,
            mode=job.mode,
            request_id=job.request_id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            timeout_seconds=job.timeout_seconds,
            result_available=job.result_path is not None and job.result_path.is_file(),
            download_available=job.status == JobState.SUCCEEDED,
            error=job.error,
            audit=list(job.audit),
        )


def wait_for_terminal(
    store: InMemoryJobStore,
    job_id: str,
    *,
    timeout_seconds: float = 5.0,
) -> JobSnapshot:
    """Poll a job until it reaches a terminal state."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = store.get(job_id)
        if snapshot.status in TERMINAL_JOB_STATES:
            return snapshot
        time.sleep(0.01)
    raise TimeoutError(f"job {job_id} did not reach a terminal state")


def _now() -> datetime:
    return datetime.now(UTC)
