"""Streamlit-facing service helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from cpicann_xrd.catalog.elements import VALID_ELEMENT_SYMBOLS
from cpicann_xrd.decomposition.capabilities import Capabilities
from cpicann_xrd.decomposition.schemas import DecomposedComponent, XDecomposerPreprocessingMetadata

MAX_UPLOAD_FILES = 20
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
WebMode = Literal["single_phase", "decompose", "decompose_and_identify"]


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


@dataclass(frozen=True)
class WebModeOption:
    """One mode option exposed by the Streamlit UI."""

    key: WebMode
    label: str
    enabled: bool
    reason: str | None = None


def available_elements() -> list[str]:
    """Return user-selectable real element symbols."""
    return sorted(element for element in VALID_ELEMENT_SYMBOLS if element != "X")


def web_mode_options(capabilities: Capabilities) -> list[WebModeOption]:
    """Return mode options without duplicating decomposition business logic."""
    xdecomposer_ready = capabilities.xdecomposer.available
    reason = None if xdecomposer_ready else capabilities.xdecomposer.reason
    return [
        WebModeOption(
            key="single_phase",
            label="单相物相识别",
            enabled=True,
        ),
        WebModeOption(
            key="decompose",
            label="多相谱图分解",
            enabled=xdecomposer_ready,
            reason=reason,
        ),
        WebModeOption(
            key="decompose_and_identify",
            label="多相分解并识别",
            enabled=xdecomposer_ready,
            reason=reason,
        ),
    ]


def component_pattern_csv(
    component: DecomposedComponent,
    preprocessing: XDecomposerPreprocessingMetadata,
) -> bytes:
    """Return one decomposed component pattern as CSV bytes for browser download."""
    if component.pattern is None:
        raise ValueError("component pattern is not available")
    if len(component.pattern) != preprocessing.output_points:
        raise ValueError("component pattern length does not match preprocessing metadata")
    if preprocessing.output_points == 1:
        angles = [preprocessing.two_theta_min]
    else:
        step = (preprocessing.two_theta_max - preprocessing.two_theta_min) / (
            preprocessing.output_points - 1
        )
        angles = [
            preprocessing.two_theta_min + step * index
            for index in range(preprocessing.output_points)
        ]
    lines = ["two_theta,intensity"]
    lines.extend(
        f"{angle:.8g},{intensity:.8g}"
        for angle, intensity in zip(angles, component.pattern, strict=True)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


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
