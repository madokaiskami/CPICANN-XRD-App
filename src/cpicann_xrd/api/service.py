"""FastAPI-facing service helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SAFE_RUN_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
DEFAULT_API_RUN_ROOT = Path(os.environ.get("CPICANN_API_RUN_ROOT", "/tmp/cpicann-xrd-api-runs"))
MAX_API_UPLOAD_FILES = 20
MAX_API_UPLOAD_BYTES = 10 * 1024 * 1024


class ApiUploadLike(Protocol):
    """Minimal protocol shared by FastAPI uploads and tests."""

    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class PreparedApiInputs:
    """Uploaded files staged for API batch processing."""

    input_dir: Path
    input_paths: list[Path]


async def stage_api_uploads(
    uploads: Sequence[ApiUploadLike],
    *,
    input_dir: Path,
    max_files: int = MAX_API_UPLOAD_FILES,
    max_bytes: int = MAX_API_UPLOAD_BYTES,
) -> PreparedApiInputs:
    """Write API uploads into an isolated directory with size and name controls."""
    if not uploads:
        raise ValueError("at least one file is required")
    if len(uploads) > max_files:
        raise ValueError(f"最多允许上传 {max_files} 个文件")
    input_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: list[Path] = []
    seen_names: dict[str, int] = {}
    for upload in uploads:
        data = await upload.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"{upload.filename or 'uploaded'} 超过大小限制")
        safe_name = safe_upload_name(upload.filename or "uploaded", seen_names)
        destination = input_dir / safe_name
        destination.write_bytes(data)
        staged_paths.append(destination)
    return PreparedApiInputs(input_dir=input_dir, input_paths=staged_paths)


def safe_upload_name(name: str, seen_names: dict[str, int] | None = None) -> str:
    """Return a basename-only upload filename and deduplicate it within one request."""
    raw_name = Path(name).name.strip() or "uploaded"
    if raw_name.startswith("."):
        raw_name = raw_name.lstrip(".") or "uploaded"
    seen = seen_names if seen_names is not None else {}
    count = seen.get(raw_name, 0) + 1
    seen[raw_name] = count
    if count == 1:
        return raw_name
    path = Path(raw_name)
    return f"{path.stem}_{count}{path.suffix}"


def api_run_root() -> Path:
    """Return the API run root without exposing it through responses."""
    return DEFAULT_API_RUN_ROOT


def resolve_run_dir(run_id: str, *, run_root: Path | None = None) -> Path:
    """Resolve a run id under the API run root, rejecting traversal and separators."""
    if not run_id or any(char not in SAFE_RUN_ID_CHARS for char in run_id):
        raise ValueError("invalid run_id")
    root = run_root or api_run_root()
    run_dir = root / run_id
    resolved_root = root.resolve()
    resolved_run_dir = run_dir.resolve()
    if resolved_root not in resolved_run_dir.parents and resolved_run_dir != resolved_root:
        raise ValueError("invalid run_id")
    return run_dir
