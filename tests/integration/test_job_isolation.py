"""Job output isolation and cleanup tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

from cpicann_xrd.api.jobs import (
    InMemoryJobStore,
    JobExecutionContext,
    JobState,
    wait_for_terminal,
)


def test_job_outputs_are_isolated_and_downloadable(tmp_path: Path) -> None:
    store = InMemoryJobStore(root=tmp_path / "jobs", max_concurrent=2)
    first = store.submit(mode="write", request_id="req-a", runner=_write_marker)
    second = store.submit(mode="write", request_id="req-b", runner=_write_marker)

    first_done = wait_for_terminal(store, first.job_id)
    second_done = wait_for_terminal(store, second.job_id)

    first_zip = store.download_path(first.job_id)
    second_zip = store.download_path(second.job_id)

    assert first_done.status == JobState.SUCCEEDED
    assert second_done.status == JobState.SUCCEEDED
    assert first.job_id != second.job_id
    assert first_zip.parent != second_zip.parent
    assert (first_zip.parent / "marker.txt").read_text(encoding="utf-8") == first.job_id
    assert (second_zip.parent / "marker.txt").read_text(encoding="utf-8") == second.job_id
    assert _zip_text(first_zip, "marker.txt") == first.job_id
    assert _zip_text(second_zip, "marker.txt") == second.job_id
    store.shutdown()


def test_terminal_jobs_are_cleaned_after_ttl(tmp_path: Path) -> None:
    store = InMemoryJobStore(root=tmp_path / "jobs", max_concurrent=1, ttl_seconds=0.0)
    snapshot = store.submit(mode="write", request_id="req-cleanup", runner=_write_marker)
    terminal = wait_for_terminal(store, snapshot.job_id)
    job_dir = tmp_path / "jobs" / snapshot.job_id

    removed = store.cleanup_expired()

    assert terminal.status == JobState.SUCCEEDED
    assert removed == 1
    assert not job_dir.exists()
    store.shutdown()


def _write_marker(context: JobExecutionContext) -> dict[str, str]:
    context.set_status(JobState.REPORTING)
    (context.job_dir / "marker.txt").write_text(context.job_id, encoding="utf-8")
    return {"marker": context.job_id}


def _zip_text(zip_path: Path, name: str) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(name).decode()
