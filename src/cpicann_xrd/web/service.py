"""Streamlit-facing service helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cpicann_xrd.catalog.elements import VALID_ELEMENT_SYMBOLS

MAX_UPLOAD_FILES = 20
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class UploadedFileLike(Protocol):
    """Minimal protocol shared by Streamlit uploads and tests."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class PreparedWebInputs:
    """Uploaded files staged for batch processing."""

    input_dir: Path
    input_paths: list[Path]
    warnings: list[str]


def available_elements() -> list[str]:
    """Return user-selectable real element symbols."""
    return sorted(element for element in VALID_ELEMENT_SYMBOLS if element != "X")


def stage_uploaded_files(
    uploaded_files: list[UploadedFileLike],
    *,
    input_dir: Path,
    max_files: int = MAX_UPLOAD_FILES,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> PreparedWebInputs:
    """Write uploaded files into an isolated temporary directory."""
    if len(uploaded_files) > max_files:
        raise ValueError(f"最多允许上传 {max_files} 个文件")
    input_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: list[Path] = []
    warnings: list[str] = []
    seen_names: dict[str, int] = {}
    for uploaded_file in uploaded_files:
        if uploaded_file.size > max_bytes:
            warnings.append(f"{uploaded_file.name} 超过大小限制，已跳过。")
            continue
        safe_name = _safe_upload_name(uploaded_file.name, seen_names)
        destination = input_dir / safe_name
        destination.write_bytes(uploaded_file.getvalue())
        staged_paths.append(destination)
    return PreparedWebInputs(input_dir=input_dir, input_paths=staged_paths, warnings=warnings)


def _safe_upload_name(name: str, seen_names: dict[str, int]) -> str:
    raw_name = Path(name).name.strip() or "uploaded"
    if raw_name.startswith("."):
        raw_name = raw_name.lstrip(".") or "uploaded"
    count = seen_names.get(raw_name, 0) + 1
    seen_names[raw_name] = count
    if count == 1:
        return raw_name
    path = Path(raw_name)
    return f"{path.stem}_{count}{path.suffix}"
