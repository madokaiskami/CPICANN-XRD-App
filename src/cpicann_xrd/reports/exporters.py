"""Atomic CSV, JSON, PNG and ZIP exporters."""

from __future__ import annotations

import csv
import json
import struct
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cpicann_xrd.core.preprocessing import PreprocessedSpectrum
from cpicann_xrd.schemas import SamplePrediction, SpectrumData


def write_text_atomic(path: Path, text: str) -> None:
    """Write text through a temporary file and replace atomically."""
    tmp_path = _tmp_path(path)
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write bytes through a temporary file and replace atomically."""
    tmp_path = _tmp_path(path)
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write pretty JSON atomically."""
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write CSV atomically."""
    tmp_path = _tmp_path(path)
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def prediction_rows(prediction: SamplePrediction) -> list[dict[str, object]]:
    """Return prediction_top5.csv rows."""
    return [
        {
            "sample_id": prediction.sample_id,
            "filtered_rank": item.filtered_rank,
            "global_rank": item.global_rank,
            "class_index": item.class_index,
            "cod_id": item.phase.cod_id,
            "formula": item.phase.formula,
            "reduced_formula": item.phase.reduced_formula,
            "elements": " ".join(sorted(item.phase.elements)),
            "space_group": item.phase.space_group or "",
            "space_group_number": item.phase.space_group_number or "",
            "raw_logit": item.raw_logit,
            "unfiltered_probability": item.unfiltered_probability,
            "filtered_confidence": item.filtered_confidence,
            "candidate_count_after_filter": prediction.candidate_count_after_filter,
        }
        for item in prediction.predictions
    ]


def write_preprocessed_csv(path: Path, preprocessed: PreprocessedSpectrum) -> None:
    """Write preprocessed XRD points."""
    write_csv_atomic(
        path,
        ["two_theta", "intensity"],
        preprocessed.export_table(),
    )


def write_observed_png(path: Path, spectrum: SpectrumData) -> None:
    """Write a small observed XRD line plot as a PNG using only stdlib encoders."""
    width = 640
    height = 360
    margin_left = 48
    margin_right = 16
    margin_top = 20
    margin_bottom = 42
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_line(
        image,
        margin_left,
        height - margin_bottom,
        width - margin_right,
        height - margin_bottom,
    )
    _draw_line(image, margin_left, margin_top, margin_left, height - margin_bottom)

    angles = np.asarray(spectrum.two_theta, dtype=np.float64)
    intensities = np.asarray(spectrum.intensity, dtype=np.float64)
    if angles.size >= 2 and intensities.size >= 2:
        x_min = float(np.min(angles))
        x_max = float(np.max(angles))
        y_min = float(np.min(intensities))
        y_max = float(np.max(intensities))
        if x_max > x_min and y_max > y_min:
            points = []
            for angle, intensity in zip(angles, intensities, strict=True):
                x = margin_left + int(
                    (float(angle) - x_min) / (x_max - x_min) * (width - margin_left - margin_right)
                )
                y = height - margin_bottom - int(
                    (float(intensity) - y_min)
                    / (y_max - y_min)
                    * (height - margin_top - margin_bottom)
                )
                points.append((x, y))
            for start, end in pairwise(points):
                _draw_line(image, start[0], start[1], end[0], end[1], color=(24, 96, 180))
    write_bytes_atomic(path, _encode_png(image))


def write_zip_bundle(run_dir: Path, zip_path: Path) -> None:
    """Create a ZIP archive for the run directory."""
    tmp_path = _tmp_path(zip_path)
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path in (tmp_path, zip_path):
                continue
            archive.write(path, path.relative_to(run_dir))
    tmp_path.replace(zip_path)


def _tmp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp")


def _draw_line(
    image: npt.NDArray[np.uint8],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    height, width, _ = image.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += sx
        if doubled_error <= dx:
            error += dx
            y0 += sy


def _encode_png(image: npt.NDArray[np.uint8]) -> bytes:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG encoder expects RGB images")
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    chunks = [
        _png_chunk(
            b"IHDR",
            struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0),
        ),
        _png_chunk(b"IDAT", zlib.compress(raw_rows)),
        _png_chunk(b"IEND", b""),
    ]
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", crc)
