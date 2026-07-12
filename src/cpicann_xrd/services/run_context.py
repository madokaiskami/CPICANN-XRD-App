"""Run directory and sample-id management."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class RunContext:
    """State for one batch run output tree."""

    output_root: Path
    run_id: str
    run_dir: Path
    _seen_sample_ids: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(cls, output_root: Path, *, run_id: str | None = None) -> RunContext:
        """Create a run context and its output directory."""
        active_run_id = run_id or create_run_id()
        run_dir = output_root / active_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "samples").mkdir()
        return cls(output_root=output_root, run_id=active_run_id, run_dir=run_dir)

    def allocate_sample_id(self, source_filename: str) -> str:
        """Return a filesystem-safe, unique sample id for a source filename."""
        stem = Path(source_filename).stem
        base = safe_sample_id(stem)
        count = self._seen_sample_ids.get(base, 0) + 1
        self._seen_sample_ids[base] = count
        if count == 1:
            return base
        return f"{base}_{count}"

    def sample_dir(self, sample_id: str) -> Path:
        """Create and return the output directory for one sample."""
        path = self.run_dir / "samples" / sample_id
        path.mkdir(parents=True, exist_ok=False)
        return path


def create_run_id(now: datetime | None = None) -> str:
    """Create a sortable run id."""
    active_now = now or datetime.now(UTC)
    timestamp = active_now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def safe_sample_id(value: str) -> str:
    """Normalize an arbitrary sample id for directory names."""
    normalized = SAFE_ID_RE.sub("_", value.strip()).strip("._-")
    return normalized or "sample"
