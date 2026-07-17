"""Job timeout tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cpicann_xrd.api.jobs import (
    InMemoryJobStore,
    JobExecutionContext,
    JobState,
    wait_for_terminal,
)


def test_job_timeout_records_structured_failure(tmp_path: Path) -> None:
    store = InMemoryJobStore(root=tmp_path / "jobs", max_concurrent=1, timeout_seconds=0.01)
    snapshot = store.submit(
        mode="slow",
        request_id="req-timeout",
        runner=_slow_runner,
    )

    terminal = wait_for_terminal(store, snapshot.job_id)

    assert terminal.status == JobState.FAILED
    assert terminal.error is not None
    assert terminal.error["code"] == "JOB_TIMEOUT"
    with pytest.raises(KeyError, match="job result is not available"):
        store.result(snapshot.job_id)
    store.shutdown()


def _slow_runner(context: JobExecutionContext) -> dict[str, str]:
    context.set_status(JobState.DECOMPOSING)
    time.sleep(0.05)
    return {"finished": "too-late"}
