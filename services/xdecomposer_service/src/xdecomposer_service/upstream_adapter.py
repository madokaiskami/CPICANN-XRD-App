"""Optional import checks for upstream XDecomposer code."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from xdecomposer_service.schemas import UpstreamStatus


def inspect_upstream(source_dir: Path | None) -> UpstreamStatus:
    """Check whether upstream XDecomposer modules are importable."""
    if source_dir is None:
        return UpstreamStatus(importable=False, message="source_dir_not_configured")
    if not source_dir.exists():
        return UpstreamStatus(
            source_dir=str(source_dir),
            importable=False,
            message="source_dir_not_found",
        )

    source_dir_str = str(source_dir)
    if source_dir_str not in sys.path:
        sys.path.insert(0, source_dir_str)

    for module_name in ("src.models.xdecomposer", "models.xdecomposer"):
        try:
            importlib.import_module(module_name)
        except Exception:
            continue
        return UpstreamStatus(
            source_dir=str(source_dir),
            importable=True,
            module=module_name,
            message="ok",
        )
    return UpstreamStatus(
        source_dir=str(source_dir),
        importable=False,
        message="xdecomposer_module_not_importable",
    )
