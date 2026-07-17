"""Optional import checks for upstream XDecomposer code."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

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
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if getattr(module, "__file__", None) is None or not hasattr(module, "XDecomposer"):
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


def import_upstream_module(source_dir: Path | None) -> ModuleType:
    """Import the reviewed upstream XDecomposer module or raise RuntimeError."""
    status = inspect_upstream(source_dir)
    if not status.importable or status.module is None:
        raise RuntimeError(status.message)
    return importlib.import_module(status.module)
