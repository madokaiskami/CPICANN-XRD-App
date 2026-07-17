"""Environment-driven settings for the isolated XDecomposer service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings for health and asset checks."""

    manifest_path: Path = Path("/app/models/xdecomposer/manifest.yaml")
    upstream_source_dir: Path | None = None
    require_upstream_import: bool = False
    expected_python_minor: str = "3.10"


def load_settings() -> Settings:
    """Load settings from environment variables."""
    source_dir = os.environ.get("XDECOMPOSER_SOURCE_DIR")
    return Settings(
        manifest_path=Path(
            os.environ.get(
                "XDECOMPOSER_MANIFEST_PATH",
                "/app/models/xdecomposer/manifest.yaml",
            )
        ),
        upstream_source_dir=None if not source_dir else Path(source_dir),
        require_upstream_import=_env_bool("XDECOMPOSER_REQUIRE_UPSTREAM_IMPORT", default=False),
        expected_python_minor=os.environ.get("XDECOMPOSER_EXPECTED_PYTHON", "3.10"),
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
